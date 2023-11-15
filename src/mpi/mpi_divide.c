#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int r;
    MPI_Comm_size(MPI_COMM_WORLD, &r);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int n = 8;
    
    float * v = (float *) malloc( sizeof( float) * n );
    
    MPI_Request * requests = (MPI_Request *) malloc( sizeof( MPI_Request) * r ); 
    
    if( rank == 0 ) {
        for( int i = 0; i < n; i++ ) {
            v[i] = i;
        }
    
        for( int i = 0; i < r; i++ ) {
            
            int initial_pos = 0;
            int size_send = 0;
            if ( i < (n % r)) {  
                initial_pos = i * (n/r+1);
                size_send = n/r + 1;
            } else {
                initial_pos = (n%r) + (n/r)*i;
                size_send = n/r;
            }    
            
            MPI_Isend(&v[initial_pos], size_send , MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requests[i] );
        
        }
    }
    
    int size_recv = 0;
    
    if (rank < (n % r)) {  
        size_recv = n/r + 1;
    } else {
        size_recv = n/r;
    } 
    
    float * v_recv = (float *) malloc( sizeof( float ) * size_recv );
    
    MPI_Request request;
    
    MPI_Irecv( v_recv, size_recv , MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &request );
    
    
    int flags = 0;
    while( flags == 0 ) {
           MPI_Test( &request, &flags, NULL );
    }
    
    for( int i = 0; i < size_recv; i ++ ) {
        printf("[%d] %f\n", rank, v_recv[i] );
    
    }
    

    MPI_Finalize();
}
