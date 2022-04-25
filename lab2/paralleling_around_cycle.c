#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <omp.h>

const long int N = 4800;
const double e = 1e-5;
float t = 1e-4;
const double timeLimit = 600;
short OMP_NUM_THREADS = 1;

double EuclideanNorm(const double* vector){
    double norm = 0;

    #pragma omp parallel for reduction(+: norm)
    for (int i = 0; i < N; ++i)
        norm += vector[i] * vector[i];

    return norm;
}

void sub(const double* from, const double* what, double* result){
    #pragma omp parallel for
    for (int i = 0; i < N; ++i)
        result[i] = from[i] - what[i];   
    
}

void mul(double* matrix, double* vector, double* result) {
    #pragma omp parallel for
    for(int i = 0; i < N; ++i){
        result[i] = 0;
        for(int j = 0; j < N; ++j)
            result[i] += matrix[i * N + j] * vector[j]; 
    }
}

void scalMulTau(double* A){
    #pragma omp parallel for
    for (int i = 0; i < N; ++i)
        A[i] *= t;
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

void generateVector(double* vector){
    for(long i = 0; i < N; ++i)
        vector[i] = drand(1,5);
}

void final_out(bool doesNotCoverege, bool timeOut, double res, int countIt, double end, double start){
    if(doesNotCoverege)
        printf("Please, take another sign for tau\n");
    else if (timeOut)
        printf("it took more than %lf seconds so I killed the process\n", timeLimit); 
    else{
        printf("%ld*%ld matrix error coefficient is %lf, iterations: %d\n", N, N, res , countIt);
        printf("That took %lf seconds\n", end - start);
    }
}

void final_free(double* Ax, double* X, double* b, double* A){
    free(Ax);
    free(X);
    free(b);
    free(A);
}

void initBlock(double* A, double* X, double* b){
    generateMatrix(A);
    generateVector(X);
    generateVector(b);
}

int main(int argc, char** argv) {
    srand(time(0));

    double* Ax = (double*)malloc(N * sizeof(double));
    double* b = (double*)malloc(N * sizeof(double));
    double* X = (double*)malloc(N * sizeof(double));
    double* A = (double*)malloc(N * N * sizeof(double));  

    bool timeOut = false, doesNotCoverage = false;
    unsigned int countIt = 1;
    double start, currentTime, end;
    double normAxb = 0, normb = 0, eSquart_mult_normb = 0, last_norm = INFINITY;

    initBlock(A, X, b);  
      
    OMP_NUM_THREADS = atoi(argv[1]);
    omp_set_num_threads(OMP_NUM_THREADS);
    normb = EuclideanNorm(b);  
    eSquart_mult_normb = normb * e * e;

    start = omp_get_wtime();

    do {
        mul(A, X, Ax); //A*xn
        sub(Ax, b, Ax); //A*xn - b
        normAxb = EuclideanNorm(Ax); // ||A*xn - b||
        scalMulTau(Ax); // TAU*(A*xn - b)
        sub(X, Ax, X); // xn - TAU * (A*xn - b)

        if ((countIt > 100000 && last_norm > normAxb) || normAxb == INFINITY){
                printf("Does not converge\n");
                final_free(Ax, X, b, A);
                doesNotCoverage = true;
        }         

        countIt++;
        last_norm = normAxb;        
        currentTime = omp_get_wtime();

        if (currentTime - start > timeLimit){
            {
                normAxb = 0;
                timeOut = true;
            }
        }

    } while (normAxb > eSquart_mult_normb && !timeOut && !doesNotCoverage);

    end = omp_get_wtime();
    final_out(doesNotCoverage, timeOut, e, countIt, end, start);

    
    final_free(Ax, X, b, A);
    return 0;
}
