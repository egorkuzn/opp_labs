#include <stdio.h>
#include <stdlib.h>

void GenMatrix(double* matrix, int lines, int columns){
    srand(1);

    for(int i = 0; i < lines; ++i)
        for(int j = 0; j < columns; ++j)
            matrix[i * columns + j] = rand() / (double)RAND_MAX * 50 - 25;
}

void WriteMatrix(double* matrix, int lines, int columns, FILE* out){
    for(int i = 0; i < lines; ++i){
        for(int j = 0; j < columns; ++j)
            fprintf(out, "%lf ", matrix[i * columns + j]);
        
        fprintf(out, "\n");
    }
}

//     <----M---->                            <-K->
//  ^  +----------+         <-K->             +---+ ^
//  |  |          |         +---+ ^           |   | |
//  |  |          |         |   | |    ____   |   | |
//  N  |    A     |    X    | B | | M  ____   | C | | N
//  |  |          |         |   | |           |   | |
//  |  |          |         +---+ v           |   | |
//  v  +----------+                           +---+ v

int main(int argc, char* argv[]){
// Getting of matrixes sizes params: 
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    int K = atoi(argv[3]);
// Memory allocation by zero:
    double* A = (double*)calloc(N * M, sizeof(double));
    double* B = (double*)calloc(M * K, sizeof(double));
// Out file preparation:
    FILE* out = fopen("matrix.txt", "w");
    fprintf(out, "%d %d %d\n", N, M, K);
// Generate matrix:
    GenMatrix(A, N, M);
    GenMatrix(B, M, K);
// Writing matrix to out.txt file:
    WriteMatrix(A, N, M, out);
    WriteMatrix(B, M, K, out);
// File closing, memory clearing and job finishing:
    fclose(out);
    free(A);
    free(B);
}