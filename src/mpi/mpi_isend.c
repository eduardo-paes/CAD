/**
 * Comunicação não bloqueante: isend e irecv
 *                              
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main( int argc, char ** argv ) {
    
    MPI_Init( &argc, &argv ); // Inicializa o ambiente MPI
    

    int rank;
    int size;
    char name[32];
    int resultlength;
    MPI_Get_processor_name(name, &resultlength);
    printf("name = %s\n", name );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);            
    
    if( rank == 1 ) {
    
        MPI_Request request;   
        sleep( 10 );    
        MPI_Isend(      &rank, 
                        1, 
                        MPI_INT, 
                        0, 
                        0,
                        MPI_COMM_WORLD, 
                        &request);
     } else if( rank == 0 ) {
            double t0 = MPI_Wtime ();
            //MPI_Request request;   
            int buf;
            MPI_Recv(  &buf, 
                        1, 
                        MPI_INT, 
                        1, 
                        0,
                        MPI_COMM_WORLD,
                        NULL);
            int flag = 0;
            /**
            printf("Terminou IRECV\n");
            while( flag == 0 ) {
                MPI_Test( &request, &flag, NULL );
                usleep( 100 );
                //printf("FLAG = %d\n", flag );
            }
            */
            double t1 = MPI_Wtime ();
            printf("%lf\n", t1 - t0);
        /*do {
            printf("Tentando testar...\n");
            sleep( 1 );
        } while( !MPI_Test( request ) );
        */
     }                          
                                
   MPI_Finalize();
   return 0;
}                             
