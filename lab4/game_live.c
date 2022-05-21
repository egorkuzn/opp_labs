#include <stdio.h>
#include <mpi/mpi.h>
#include <stdlib.h>

const int ITERATIONS_COUNT = 10000;

struct grid{
    int width,
        height;
    unsigned int size;
} grid;


void Grid_Init(char ** argv){

}

void General_cycle(){
    int current_iteration = 0;
    while(current_iteration < ITERATIONS_COUNT){
        
    }
}

void Print_Picture(){
    
}

int main(int argc, char* argv[]){    
    int size, rank;
    MPI_Init(&argc, &argv);
    Grid_Init(argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Print_Picture();
    General_cycle();


    
    MPI_Finalize();
    return 0;
}