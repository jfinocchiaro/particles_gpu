#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <algorithm>
#include "common.h"


//taken from common.cu
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

#define NUM_THREADS 256
#define MAXITEM 4 //At most MAXITEM particles in one bin. Change depends on binSize
#define CUTOFF_SCALE 4

extern double size;

double binSize;
int binNum;

//
//  benchmarking program
//


__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  //r2 = fmax( r2, min_r*min_r );
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}

//direction to check 8 neighboring boxes and self
__constant__ const int dir[9][2]={{0,0},{-1,-1},{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0}};

__global__ void compute_forces_gpu(particle_t * particles, int*cnt,int n,double binSize,int binNum)
{
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //increments for loop
    int offset = gridDim.x*blockDim.x;

    for(int ii=tid;ii<n;ii+=offset)
    {
        particle_t p = particles[ii];
        p.ax=p.ay=0;
        int i = int(p.x / binSize);
        int j = int(p.y / binSize);
        for(int t=0;t<9;t++)
        {
            //which box to check
            int x = i + dir[t][0];
            int y = j + dir[t][1];
            //if we're actually in the grid
            if (x >= 0 && x < binNum && y >= 0 && y < binNum)
            {
                int id = x*binNum+y;
                int start = cnt[id-1],end = cnt[id];
                for (int k = start; k < end; k++)
                    apply_force_gpu(p, particles[k]);
            }
        }
        particles[ii].ax = p.ax;
        particles[ii].ay = p.ay;
    }
}

__global__ void move_gpu (particle_t * __restrict__ particles, int n, double size)
{

  // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x*blockDim.x;
    for(int i=tid;i<n;i+=offset)
    {

        particle_t * p = &particles[i];
        //
        //  slightly simplified Velocity Verlet integration
        //  conserves energy better than explicit Euler method
        //
        p->vx += p->ax * dt;
        p->vy += p->ay * dt;
        p->x  += p->vx * dt;
        p->y  += p->vy * dt;

        //
        //  bounce from walls
        //
        while( p->x < 0 || p->x > size )
        {
            p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
            p->vx = -(p->vx);
        }
        while( p->y < 0 || p->y > size )
        {
            p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
            p->vy = -(p->vy);
        }
    }

}

__global__ void getCount(particle_t* particles, int* count,int n,double binSize,int binNum)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x*blockDim.x;
    for(int i=tid;i<n;i+=offset)
    {
        int x = int(particles[i].x / binSize); //find x-coord of current bin
        int y = int(particles[i].y / binSize); //find y-coord of bin
        atomicAdd(count+x*binNum+y,1);
    }
}

__global__ void buildBins(particle_t* particles,particle_t* tmp,int* count,int n,double binSize,int binNum)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x*blockDim.x;
    for(int i=tid;i<n;i+=offset)
    {
        int x = int(particles[i].x / binSize);//find x-coord of current bin
        int y = int(particles[i].y / binSize); //find y-coord of bin
        int id = atomicSub(count+x*binNum+y,1);
        tmp[id-1] = particles[i];
    }
}

int main( int argc, char **argv )
{
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize();

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * d_particles,*tmp;
    //allocate memory for CUDA
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));
    cudaMalloc((void **) &tmp, n * sizeof(particle_t));

    //set size of box
    set_size( n );

    binSize = cutoff*CUTOFF_SCALE;
    binNum = int(size / binSize)+1; // Should be around sqrt(N/2)
    printf("Grid Size: %.4lf\n",size);
    printf("Number of Bins: %d*%d\n",binNum,binNum);
    printf("Bin Size: %.2lf\n",binSize);

    //pointer to an int array
    int* cnt;
    //allocate CUDA memory
    cudaMalloc((void **) &cnt, (binNum*binNum+1) * sizeof(int));
    cudaMemset(cnt,0,(binNum*binNum+1)*sizeof(int));
    cnt+=1; //Add one therefore cnt[-1]==0
    int* count = (int*) malloc(binNum*binNum * sizeof(int));

    //initialize particles in the box
    init_particles( n, particles );

    //wait for everything to be on the same page
    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);
    //wait for everything to be on the same page
    cudaThreadSynchronize();
    //get the run time of copying the data
    copy_time = read_timer( ) - copy_time;

    //
    //  simulate a number of time steps
    //wait for everything to sync
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    for( int step = 0; step < NSTEPS; step++ )
    {
        //
        //  compute forces
        int threadNum = NUM_THREADS;
        int blks = min(1024,(n + NUM_THREADS - 1) / NUM_THREADS);
    	  int blockNum = blks;//min(512,(n+threadNum-1)/threadNum);


        //Old methods that don't assume maxitems for each bin, but just calculate everything
        cudaMemset(cnt,0,binNum*binNum*sizeof(int));
        getCount<<<blockNum,threadNum>>>(d_particles,cnt,n,binSize,binNum);

        cudaMemcpy(count, cnt, binNum*binNum * sizeof(int), cudaMemcpyDeviceToHost);
        for(int i=1;i<binNum*binNum;i++)  //Prefix sum  could be faster using parallel....
            count[i]+=count[i-1];
        cudaMemcpy(cnt, count, binNum*binNum * sizeof(int), cudaMemcpyHostToDevice);
        //build subbins to calculate forces later
        buildBins<<<blockNum,threadNum>>>(d_particles,tmp,cnt,n,binSize,binNum);
        std::swap(d_particles,tmp);
        cudaMemcpy(cnt, count, binNum*binNum * sizeof(int), cudaMemcpyHostToDevice);


        //only computes forces with nearby particles
    	  compute_forces_gpu<<<blks, NUM_THREADS>>> (d_particles,cnt,n,binSize,binNum);

        //
        //  move particles
      	move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);


        //  save if necessary
        if( fsave && (step%SAVEFREQ) == 0 )
        {
	        // Copy the particles back to the CPU
          cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
          save( fsave, n, particles);
	      }
    }
    //wait for everything to catch up
    cudaThreadSynchronize();
    //final run time
    simulation_time = read_timer( ) - simulation_time;

    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );

    free( particles );
    cudaFree(d_particles);
    if( fsave )
        fclose( fsave );

    return 0;
}
