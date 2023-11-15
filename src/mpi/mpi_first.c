#include <mpi.h>
#include <stdio.h>

int main(int argc, char ** argv ) {
    MPI_Init( &argc, &argv );
    
    int rank;
    int size;
    
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    
       
    
    if( rank == 0 ) {
        // 0 -> 1
        char * str = "CEFET";
        MPI_Send(           str,   // buf
                              6,   // count
                        MPI_CHAR,  // data_type
                               1,  // dest
                               0,  // tag
                  MPI_COMM_WORLD );// communicator
    }
    
    
    if( rank == 1 ) {
        char str[6];
        MPI_Recv(    str,      // buffer
                       6,      // count
                MPI_CHAR,      // data_type
                       0,      // source 
                       0,      // tag
          MPI_COMM_WORLD,      // communicator
                    NULL );    // status
        printf("%s\n", str );
    }
    
    
    
    
    
    
    MPI_Finalize();
    return 0;
}

