#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main( int argc, char ** argv ) {
    MPI_Init( &argc, &argv );    
    
    int rank;
    int size;
    
    MPI_Comm_rank( MPI_COMM_WORLD, &rank  );
    MPI_Comm_size( MPI_COMM_WORLD, &size  );
    
    // Rank[0] envia a mensagem "CEFET" para Rank[1]
    
    int N = 100;
    int chunk = N/size;
    if( rank == 0 ) {
        double * u = (double *) malloc( sizeof( double ) * N );
        double * v = (double *) malloc( sizeof( double ) * N );
        double * w = (double *) malloc( sizeof( double ) * N );
        
        for( int i = 0; i < N; i++ ) {
            u[i] = 1;
            v[i] = 1;
        }
        
        for( int i = 1; i < size; i++ ) {
            MPI_Send( (const void *) &u[ i * chunk ], 
                     chunk,
                     MPI_DOUBLE, 
                     i,
                     0, 
                     MPI_COMM_WORLD );
            MPI_Send( (const void *) &v[ i * chunk ], 
                     chunk,
                     MPI_DOUBLE, 
                     i,
                     0, 
                     MPI_COMM_WORLD );
        }
        
        for( int i = 0; i < chunk; i++ ) {
            w[i] = v[i] + u[i];
        }
        
        for( int i = 1; i < size; i++ ) {
            MPI_Recv((void *) &w[i*chunk], 
                 chunk, 
                 MPI_DOUBLE,
                 i, 
                 0, 
                 MPI_COMM_WORLD, 
                 NULL );
        }
        
        for( int i = 0; i < N; i++ ) {
            printf("%lf\n", w[i] );
        
        }

    } else {
        double * u = (double *) malloc( sizeof( double ) * chunk );
        double * v = (double *) malloc( sizeof( double ) * chunk );
        
        double * w = (double *) malloc( sizeof( double ) * chunk );
        // Recebe u do Rank[0]
        MPI_Recv((void *) u, 
                 chunk, 
                 MPI_DOUBLE,
                 0, 
                 0, 
                 MPI_COMM_WORLD, 
                 NULL );
       // Recebe v do Rank[0]          
       MPI_Recv((void *) v, 
                 chunk, 
                 MPI_DOUBLE,
                 0, 
                 0, 
                 MPI_COMM_WORLD, 
                 NULL );
        // Processa
        for( int i = 0; i < chunk; i++ ) {
            w[i] = u[i] + v[i];
        }
        // Envia o resultado w para Rank[0]
        MPI_Send( (const void *) w, 
                     chunk,
                     MPI_DOUBLE, 
                     0,
                     0, 
                     MPI_COMM_WORLD );
 
    }
    
    MPI_Finalize();
}
