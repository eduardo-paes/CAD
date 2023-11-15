#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
int main(int argc, char ** argv ) {
    MPI_Init( &argc, &argv );
    
    int rank;
    int var = 123;    
    
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
    
    
    if( rank == 0 ) {
        var = 10;
    }

    if( rank == 1 ) {    
        MPI_Request request;
        MPI_Irecv( &var,1,MPI_INT, 0, 0,MPI_COMM_WORLD, &request );
        
        /*
        int flag = 0;
        do {
            
            usleep( 100 );
            MPI_Test( &request, &flag, NULL );
            
        } while( flag == 0 );
        */
        printf("Waiting...\n"); 
   	MPI_Status status;
   	MPI_Wait( &request, &status ); 
        
        printf("%d\n", var );
        
    } else {
        sleep( 2 );
        MPI_Request request;
        MPI_Isend(&var,1,MPI_INT, 1, 0,MPI_COMM_WORLD, &request );
        
        
   	MPI_Status status;
   	MPI_Wait( &request, &status ); 
    }
    
    MPI_Finalize();
    return 0;
}
