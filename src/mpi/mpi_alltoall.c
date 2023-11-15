/*int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm comm)
               */
               


#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>


int main(int argc, char ** argv ) {
    MPI_Init( &argc, & argv );

    int rank;
    int size;
    
    MPI_Comm_rank(MPI_COMM_WORLD,  &rank );
    MPI_Comm_size(MPI_COMM_WORLD,  &size );   
    
    int send[4];
    for( int i = 0; i < 4; i++ ) {
        send[i] = rank*4 + i;
    }
    
    int recv[4];

     
    MPI_Alltoall(send, 1, MPI_INT, recv, 1, MPI_INT, MPI_COMM_WORLD );

/*    
    for( int r = 0; r < size; r++ ) {
	if( r == rank ) {
    		for( int i = 0; i < 4; i++ ) {
    			printf("[%d] %d\n", rank, recv[i]);
		}
	}
    	MPI_Barrier( MPI_COMM_WORLD );
    }
*/    
    
    int final_recv[16];
    
    
    MPI_Gather( recv, 4, MPI_INT, final_recv, 4, MPI_INT, 0, MPI_COMM_WORLD );              
    
                  
    if( rank == 0 ) {           
        int m[4][4]; 
        for( int i = 0; i < 4; i++ ) {
            for( int j = 0; j < 4; j++ ) {
                m[i][j] = final_recv[j*4+i];
                printf("%d\t", m[i][j] );
            }
            printf("\n");
        }
    }
    MPI_Finalize();
    return 0;
}    
                   
