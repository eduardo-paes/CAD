#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main(int argc, char ** argv ) {
    MPI_Init( &argc, &argv );
    int rank;
    int size;

    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    int N = 10000;
    double * X; 
    double * Y; 
    
    double * X_local = (double *) malloc( sizeof( double ) * (N / size) );
    double * Y_local = (double *) malloc( sizeof( double ) * (N / size) );
    
    if( rank == 0 ) {
        X =  (double *) malloc( sizeof( double ) * N );
        Y =  (double *) malloc( sizeof( double ) * N );

        for( int i = 0; i < N; i++ ) {
            X[i] = i/100.;
            
            Y[i] = 2.39 * X[i] + 1.87;// + ((rand() % 1000)/1000. - 0.5)*10;
            //printf( "%lf %lf\n", X[i], Y[i] );
        } 
    
    }
    MPI_Request request_y, request_x;
    
    MPI_Iscatter( X, N/size, MPI_DOUBLE, X_local, N/size,MPI_DOUBLE, 0, MPI_COMM_WORLD, &request_x );
    MPI_Iscatter( Y, N/size, MPI_DOUBLE, Y_local, N/size,MPI_DOUBLE, 0, MPI_COMM_WORLD, &request_y );
    
    MPI_Wait( &request_x, NULL );
    
    double sum_local_x = 0.;
    for( int i = 0; i < N/size; i++ ) {
        sum_local_x += X_local[i];
    }
    
    MPI_Wait( &request_y, NULL );
    
    double sum_local_y = 0.;
    for( int i = 0; i < N/size; i++ ) {
        sum_local_y += Y_local[i];
    }
    
    
    double sum_global_x, sum_global_y;
    
    MPI_Reduce( &sum_local_x, &sum_global_x, 1,MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
    MPI_Reduce( &sum_local_y, &sum_global_y, 1,MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
    

    double mean_x =       sum_global_x / N;
    double mean_y =       sum_global_y / N;
    
    MPI_Bcast(&mean_x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    MPI_Bcast(&mean_y, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    
    
    double sum_local_xx = 0.;
    for( int i = 0; i < N/size; i++ ) {
        sum_local_xx += (X_local[i] - mean_x )*(X_local[i] - mean_x);
    }
    
    double sum_local_xy = 0.;
    for( int i = 0; i < N/size; i++ ) {
        sum_local_xy += (X_local[i] - mean_x )*(Y_local[i] - mean_y);
    }     
    
    double sum_global_xx, sum_global_xy;
    
    MPI_Reduce( &sum_local_xx, &sum_global_xx, 1,MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
    MPI_Reduce( &sum_local_xy, &sum_global_xy, 1,MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );        
    
    if( rank == 0 ) {
        double beta  =   sum_global_xy /   sum_global_xx;
        double alpha =   mean_y - beta * mean_x;
        printf("alpha = %lf beta = %lf\n", alpha, beta );
    }
     
    MPI_Finalize();
    return 0;
}

