#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>

const long int N = 3200;
const double ε = 1e-5;
float τ = 1e-4;
const clock_t timeLimit = 600;

double EuclideanNorm(const double* vector){
    double norm = 0;

    for (int i = 0; i < N; ++i)
        norm += vector[i] * vector[i];

    return norm;
}

void sub(const double* from, const double* what, double* result){
    for (int i = 0; i < N; ++i)
        result[i] = from[i] - what[i];
}

void mul(double* matrix, double* vector, double* result) {
    memset(result, 0, N * sizeof(double));

    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            result[j] += matrix[i * N + j] * vector[i];  
}

void scalMulTau(double* A){
    for (int i = 0; i < N; ++i)
        A[i] = A[i] * τ;
}

double drand(double low, double high) {
    double f = (double)rand() / RAND_MAX;
    return low + f * (high - low);
}

double randDouble(){
    return (double)rand() / RAND_MAX * 4.0 - 2.0;
}

void generateMatrix(double* matrix) {
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < i; j++)
            matrix[i * N + j] = matrix[j*N + i];

        for(int j = i; j < N; j++){
            matrix[i * N + j] = randDouble();

            if(i == j)
                matrix[i * N + j] = fabs(matrix[i * N + j]) + 400.0; 
        }
    }
}

void final_out(bool timeOut, double res, int countIt, clock_t end, clock_t start){
    if (!timeOut){
        printf("%ld*%ld matrix error coefficient is %lf, iterations: %d\n", N, N, res , countIt);
        printf("That took %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    } else 
        printf("it took more than %ld seconds so I killed the process\n", timeLimit); 
}

void final_free(double* Ax, double* X, double* b, double* A){
    free(Ax);
    free(X);
    free(b);
    free(A);
}

int main(int argc, char** argv) {
    srand(time(0));

    double* Ax = (double*)malloc(N * sizeof(double));
    double* b = (double*)malloc(N * sizeof(double));
    double* X = (double*)malloc(N * sizeof(double));
    double* A = (double*)malloc(N * N * sizeof(double));  

    int countIt = 1;
    bool timeOut = false;
    clock_t start, currentTime, end;
    double normAxb = 0, normb = 0, εSquart_mult_normb = 0, last_norm = INFINITY;

    start = clock();
    for(long i = 0; i < N; i++) {
        X[i] = drand(1, 5);
        b[i] = drand(1, 5);
    }

    generateMatrix(A);
    normAxb = EuclideanNorm(Ax); // ||A*xn - b||
    normb = EuclideanNorm(b);  

    εSquart_mult_normb = normb * ε * ε;

    do {
        mul(A, X, Ax); //A*xn
        sub(Ax, b, Ax); //A*xn - b
        normAxb = EuclideanNorm(Ax); // ||A*xn - b||
        scalMulTau(Ax); // TAU*(A*xn - b)
        sub(X, Ax, X); // xn - TAU * (A*xn - b)
        ++countIt;

        if ((countIt > 100000 && last_norm > normAxb) || normAxb == INFINITY){
                printf("Does not converge\n");
                final_free(Ax, X, b, A);
                return 0;
        }         

        last_norm = normAxb;
        currentTime = clock();

        if ((double)(currentTime - start) / CLOCKS_PER_SEC > timeLimit){
            normAxb = 0;
            timeOut = true;
        }
    } while (normAxb > εSquart_mult_normb && !timeOut);

    end = clock();
    final_out(timeOut, ε, countIt, end, start);
    final_free(Ax, X, b, A);
    return 0;
}
