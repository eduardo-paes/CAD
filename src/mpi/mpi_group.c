#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXNP 10



int isIn( int rank, int n, int * ranks ) {
    for( int i = 0; i < n; i++ ) {
        if( rank == ranks[i] )
            return 1;
    }
    return 0;
}

int showmsg( char * msg ) {
    int rank;  
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if( rank == 0 )
        printf( msg );
}
int main( int argc, char ** argv ) {

    MPI_Init( &argc, &argv ); // Inicializa o ambiente MPI
    
    int rank;
    int size;

    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    
    if( size < 10 ) {
        showmsg("Passe np maior que 10\n");
        goto RELEASE;
    }
    
    MPI_Group world_group;
    MPI_Group local_group;
    
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);


    int ranks [] = { 1, 3, 5, 7, 9 };
    int size_group = 5;
    
    if( isIn( rank, size_group, ranks ) ) 
        MPI_Group_incl( world_group, size_group,  ranks, &local_group ); 
    else              
        MPI_Group_excl( world_group, size_group,  ranks, &local_group ); 
    
    
    MPI_Comm  newcomm;
    
    MPI_Comm_create_group (
	    MPI_COMM_WORLD,
	    local_group,
	    0,
	    &newcomm );
	    
    int rank_local;
    if( isIn( rank, size_group, ranks )  ) {
        MPI_Comm_rank( newcomm, &rank_local);
        printf("[1] %d/%d\n", rank_local, rank );   
    } else {
        MPI_Comm_rank( newcomm, &rank_local);
        printf("[2] %d/%d\n", rank_local, rank );   
    }
    
    MPI_Group_free(&local_group);
    MPI_Group_free(&world_group);
    MPI_Comm_free(&newcomm);
    	
	
RELEASE:	

	MPI_Finalize();    
	return 0;    
}	



