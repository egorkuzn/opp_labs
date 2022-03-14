#include <iostream>
#include <cmath>

const long int N = 6400;
const double ε = 1e-5;
float τ = 1e-4;
const clock_t timeLimit = 600;

double EuclideanNorm(const double* u){
    double norm = 0;

    for (int i = 0; i < N; i++)
        norm += u[i]*u[i];

    return norm;
}

void sub(double* a, double* b, double* c){
    for (int i = 0; i < N; i++)
        c[i] = a[i] - b[i];
}

void mul(double* A, double* b, double* result) {
    for(unsigned int i = 0; i < N; i++) {
        result[i] = 0;

        for(unsigned int j = 0; j < N; j++) 
            result[i] += A[i * N + j] * b[j];
    }
}

void scalMul(double* A){
    for (int i = 0; i < N; ++i)
        A[i] = A[i] * τ;
}

void printMatrix(double* A){
    printf("\n");

    for(unsigned int i = 0; i < N; i++) {
        for(unsigned int j = 0; j < N; j++)
            printf("%lf ", A[i * N + j]);

        printf("\n");
    }

    printf("\n");
}

void printVec(double* A, int n){

    printf("\n\n");

    for (int i = 0; i < n; ++i)
        printf("%lf ", A[i]);

    printf("\n\n");
}

double drand(double low, double high) {
    double f = (double)rand() / RAND_MAX;
    return low + f * (high - low);
}

double rand_double(){
    return (double)rand()/RAND_MAX*4.0 - 2.0;
}

void generate_matrix(double* matrix) {
    for(int i = 0; i < N; i++){
        for(int j = 0; j < i; j++)
            matrix[i*N + j] = matrix[j*N + i];

        for(int j = i; j < N; j++){
            matrix[i*N + j] = rand_double();

            if(i == j)
                matrix[i*N + j] = fabs(matrix[i*N + j]) + 400.0;            // make smaller 
        }
    }
}

void final_out(bool timeOut, double res, int countIt, clock_t end, clock_t start){
    if (!timeOut){
        printf("%ld*%ld matrix error coefficient is %lf, iterations: %d\n", N, N, sqrt(res) , countIt);
        printf("That took %lf seconds\n", double(end - start) / CLOCKS_PER_SEC);
    } else 
        printf("it took more than %ld seconds so I killed the process\n", timeLimit); 
}

void free_block(double* Ax, double* nextX, double* prevX, double* b, double* A){
    free(Ax);
    free(nextX);
    free(prevX);
    free(b);
    free(A);
}

int main(int argc, char** argv) {
    srand(time(0));

    double* prevX = (double*)malloc(N * sizeof(double));
    double* Ax = (double*)malloc(N * sizeof(double));
    double* b = (double*)malloc(N * sizeof(double));
    double* nextX = (double*)malloc(N * sizeof(double));
    double* A = (double*)malloc(N * N * sizeof(double));

    double normAxb = 0, normb = 0, saveRes = 0, res = 0;
    double lastres = 0; bool timeOut = false; int countIt = 1;
    clock_t start, currentTime, end;

    start = clock();
    for(long i = 0; i < N; i++) {
        prevX[i] = drand(1, 5);
        b[i] = drand(1, 5);
    }

    generate_matrix(A);
    mul(A, prevX, Ax); // A*xn
    sub(Ax, b, Ax); // A*xn - b 
    normAxb = EuclideanNorm(Ax); // ||A*xn - b||
    normb = EuclideanNorm(b);  
    scalMul(Ax); // TAU*(A*xn - b)
    sub(prevX, Ax, nextX); // xn - TAU * (A*xn - b)

    saveRes = normAxb / normb;
    res = normAxb / normb;
    lastres = res;    

    while (res > ε * ε && !timeOut) {
        for (long i = 0; i < N; i++)
            prevX[i] = nextX[i];        

        mul(A, prevX, Ax); //A*xn
        sub(Ax, b, Ax); //A*xn - b
        normAxb = EuclideanNorm(Ax); // ||A*xn - b||
        scalMul(Ax); // TAU*(A*xn - b)
        sub(prevX, Ax, nextX); // xn - TAU * (A*xn - b)
        res = normAxb / normb;
        countIt++;

        if ((countIt > 100000 && lastres < res) || res == INFINITY){
                printf("Does not converge\n");
                free_block(Ax, nextX, prevX, b, A);
                return 0;
        } 
        

        lastres = res;
        currentTime = clock();

        if ( double(currentTime - start) / CLOCKS_PER_SEC > timeLimit){
            res = 0;
            timeOut = true;
        }
    }

    end = clock();
    final_out(timeOut, res, countIt, end, start);
    free_block(Ax, nextX, prevX, b, A);
    return 0;
}
