#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char ** argv ) {
    MPI_Init( &argc, &argv );
    
    int rank;
    int size;
    
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    
    
    
    int N = atoi( argv[1] );
    int chunk = N / size;
    
    double h = 1./N;
    
    double cpi = 0.;
       
    for( int i = (rank * chunk); i < (rank+1) * chunk; i++ ) {
        double x = (i+1)*h;
        double fx = 4./(1+x*x);
        cpi += h * fx;
    }
    
    if( rank != 0 )
        MPI_Send( &cpi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD );
    else {
        double pi = cpi;
        for( int i = 1; i < size; i++ ) {
            double ranks_cpi;
	    MPI_Status status;
            MPI_Recv( &ranks_cpi, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status );
            pi += ranks_cpi; 
        }
        printf("pi = %.10lf\n", pi );
    }
    
    
    MPI_Finalize();
    return 0;
}

