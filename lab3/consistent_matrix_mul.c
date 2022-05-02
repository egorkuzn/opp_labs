#include <stdio.h>
#include <stdlib.h>

void MatrixRead(double* matrix, int lines, int columns, FILE* in){
    for(int i = 0; i < lines * columns; ++i)
        fscanf(in, "%lf", &matrix[i]);
}

void MatrixMul(double* A, double* B, double* C, int N, int M, int K){
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < M; ++j)
            for(int k = 0; k < K; ++k)
                C[i * K + k] += A[i * M + j] * B[j * N + k];
}

void MatrixPrint(double* matrix, int lines, int columns){
    for(int i = 0; i < lines; ++i){
        for(int j = 0; j < columns; ++j)
            printf("%lf ", matrix[i * columns + j]);
        
        printf("\n");
    }
}

void MatrixFree(double* A, double* B, double* C){
    free(A);
    free(B);
    free(C);
}

//     <----M---->                            <-K->
//  ^  +----------+         <-K->             +---+ ^
//  |  |          |         +---+ ^           |   | |
//  |  |          |         |   | |    ____   |   | |
//  N  |    A     |    X    | B | | M  ____   | C | | N
//  |  |          |         |   | |           |   | |
//  |  |          |         +---+ v           |   | |
//  v  +----------+                           +---+ v

int main(){
    int N, M, K;
// Input file praparation:
    FILE* in = fopen("matrix.txt", "r");
// Taking params:
    fscanf(in, "%d %d %d", &N, &M, &K);
// Memory allocation block:
    double* A = (double*)calloc(N * M, sizeof(double));
    double* B = (double*)calloc(M * K, sizeof(double));
    double* C = (double*)calloc(N * K, sizeof(double));
// Taking generated matrixes:
    MatrixRead(A, N, M, in);
    MatrixRead(B, M, K, in);
// Finishing reading from file:
    fclose(in);
// Calculating C = A x B:
    MatrixMul(A, B, C, N, M, K);
// Results demonstration:
    MatrixPrint(C, N, K);
// Free block and finishing:    
    MatrixFree(A, B, C);
    return 0;
}