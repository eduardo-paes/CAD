
#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


int main(int argc, char ** argv ) {
    MPI_Init( &argc, & argv );

    // Get the rank and size in the original communicator
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the group of processes in MPI_COMM_WORLD
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    int n = 6;
    const int prime_ranks[6] = {2, 3, 5, 7, 11, 13};

    // Construct a group containing all of the prime ranks in world_group
    MPI_Group prime_group;
    MPI_Group non_prime_group;
    
    MPI_Group_incl(world_group, 6, prime_ranks, &prime_group);
    MPI_Group_excl(world_group, 6, prime_ranks, &non_prime_group);

    // Create a new communicator based on the group
    MPI_Comm prime_comm;
    MPI_Comm non_prime_comm;
    MPI_Comm_create_group(MPI_COMM_WORLD, prime_group, 0, &prime_comm);
    MPI_Comm_create_group(MPI_COMM_WORLD, non_prime_group, 0, &non_prime_comm);
    
    
    int prime_rank = -1, prime_size = -1;
    int non_prime_rank = -1, non_prime_size = -1;
    // If this rank isn't in the new communicator, it will be
    // MPI_COMM_NULL. Using MPI_COMM_NULL for MPI_Comm_rank or
    // MPI_Comm_size is erroneous
    if (MPI_COMM_NULL != prime_comm) {
            MPI_Comm_rank(prime_comm, &prime_rank);
            MPI_Comm_size(prime_comm, &prime_size);
    } else if( MPI_COMM_NULL != non_prime_comm ) {
            MPI_Comm_rank(non_prime_comm, &non_prime_rank);
            MPI_Comm_size(non_prime_comm, &non_prime_size);
    }

    printf("WORLD RANK/SIZE: %d/%d \t PRIME RANK/SIZE: %d/%d\n",
            world_rank, world_size, prime_rank, prime_size);
    
    
    printf("WORLD RANK/SIZE: %d/%d \t NON PRIME RANK/SIZE: %d/%d\n",
            world_rank, world_size, non_prime_rank, non_prime_size);
            
            
    MPI_Group_free(&world_group);
    MPI_Group_free(&prime_group);
    MPI_Group_free(&non_prime_group);
    
    if (MPI_COMM_NULL != prime_comm) 
        MPI_Comm_free(&prime_comm);
    else if(MPI_COMM_NULL != non_prime_comm) 
        MPI_Comm_free(&non_prime_comm);
    MPI_Finalize();
    return 0;
}    
