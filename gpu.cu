#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"


//repeated from common.cu
#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

#define NUM_THREADS 256

extern double size;
extern double gridSize, binSize;
extern int binNum;

//  benchmarking program

__host__ static int grid_coord(double c)
{
    return (int)floor(c / cutoff);
}

__host__ static int grid_coord_flat(int size, double x, double y)
{
    return grid_coord(x) * size + grid_coord(y);
}

__host__ void grid_init(grid_t & grid, int size)
{
    grid.size = size;

    // Initialize grid
    grid.grid = (linkedlist**) malloc(sizeof(linkedlist*) * size * size);

    if (grid.grid == NULL)
    {
        fprintf(stderr, "Error: Could not allocate memory for the grid!\n");
        exit(1);
    }
}
__host__ void grid_add (grid_t & grid, particle_t * p)
{
    int gridCoord = grid_coord_flat(grid.size, p->x, p->y);

    linkedlist_t * newElement = (linkedlist_t *) malloc(sizeof(linkedlist));
    newElement->value = p;

    newElement->next = grid.grid[gridCoord];

    grid.grid[gridCoord] = newElement;


}

__host__ void buildGrids(grid_t grid , particle_t* particles, int n)
{


    int gridSize = (size/cutoff) + 1;

    //initialize the grid with given params
    grid_init(grid, gridSize);
    //add all particles to grid
    for (int i = 0; i < n; ++i)
    {
        grid_add(grid, &particles[i]);
    }

}

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
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}

__global__ void compute_forces_gpu( grid_t grid, particle_t * particles, int n) //n is number of particles
{
  // Get thread (particle) ID (one row to represent entire block)
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  int particlesInBin = grid.size;

  particles[tid].ax = particles[tid].ay = 0; //initialize acceleration to 0
  for(int j = 0 ; j < particlesInBin ; j++) //for every particle
  //while(neighbor != NULL)
  {
    //double distance = (particles[tid].x - particles[j].x) * (particles[tid].x - particles[j].x) + (particles[tid].y - particles[j].y) *(particles[tid].y - particles[j].y);
    //if(distance < cutoff)
      apply_force_gpu(particles[tid], grid[j]); //apply force to every other particle
      //neighbor = neighbor->next;
  }


}

__global__ void move_gpu (particle_t * particles, int n, double size)
{

  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particle_t * p = &particles[tid];

    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method

    p->vx += p->ax * dt; //time interval, 0.0005 ish declared in common.cu
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;  //bounce off the wall
        p->vx = -(p->vx); //change direction of velocity
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y; // bounce off the wall
        p->vy = -(p->vy); //change direction of velocity
    }

}



int main( int argc, char **argv )
{

    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize();  // Blocks until the device has completed all preceding requested tasks. Error if one of the preceding tasks fails.

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 ); //number of particles

    char *savename = read_string( argc, argv, "-o", NULL ); //output filename

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) ); //linked list of n particles (pointer to the top)
    grid_t grid;


    // GPU particle data structure
    particle_t * d_particles; //destination of CUDAmemcpy, used on GPU
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t)); //cuda stuff... memory allocation

    set_size( n ); // sets the double size equal to sqrt( density * n )
    init_particles( n, particles ); //initialize all 1000 particles or whatever
    buildGrids(grid, particles, n); //builds bins for force to only be applied to nearby particles

    cudaThreadSynchronize(); // Blocks until the device has completed all preceding requested tasks. Error if one of the preceding tasks fails.
    double copy_time = read_timer( ); //gets the current time

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice); //destination, source, number of bytes to copy, type of transfer

    cudaThreadSynchronize(); // Blocks until the device has completed all preceding requested tasks. Error if one of the preceding tasks fails.
    copy_time = read_timer( ) - copy_time; //retrieves how long the copy tool

    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize(); // Blocks until the device has completed all preceding requested tasks. Error if one of the preceding tasks fails.
    double simulation_time = read_timer( ); //starts timer for simulation

    for( int step = 0; step < NSTEPS; step++ ) //for designated number of steps
    {

         //  compute forces
	       int blks = (n + NUM_THREADS - 1) / NUM_THREADS; //blocks? see how this gets used
	       compute_forces_gpu <<< blks, NUM_THREADS >>> (grid, d_particles, n); // call compute_forces_gpu , execution configuration, params


        //  move particles
	       move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size); //call move_gpu, execution, params

        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
	         // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost); //brings particles back to CPU
            save( fsave, n, particles); //save progress
	         }
    }
    cudaThreadSynchronize(); //Blocks until the device has completed all preceding requested tasks. Error if one of the preceding tasks fails.
    simulation_time = read_timer( ) - simulation_time;

    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );

    free( particles );
    cudaFree(d_particles);
    if( fsave )
        fclose( fsave );

    return 0;
}
