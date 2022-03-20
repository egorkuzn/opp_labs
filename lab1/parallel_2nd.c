#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi/mpi.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>

const int N = 3200;
const double timeLimit = 120.0;
const double ε = 1e-5;
const double εSquard = ε * ε;
const double τ = 1e-4;

double EuclideanNorm(const double* vector, const int size){
    double norm = 0;

    for (int i = 0; i < size; ++i)
        norm += vector[i] * vector[i];

    return norm;
}

void sub(const double* from, const double* what, double* result, const int size){
    for (int i = 0; i < size; ++i)
        result[i] = from[i] - what[i];
} 

void mul(double* matrix, double* vector, double* result, const int size) { 
    memset(result, 0, N * sizeof(double));

    for(int i = 0; i < size / N; ++i)
        for(int j = 0; j < N; ++j)
            result[j] += matrix[i * N + j] * vector[i];  
}

void scalMulTau(double* A, const int size){
    for (int i = 0; i < size; ++i)
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
       for(int j = 0; j < i; ++j)
           matrix[i * N + j] = matrix[j * N + i];        
       
       for(int j = i; j < N; ++j){ 
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

void final_out(bool timeOut, double res, int countIt, double end, double start){
    if (!timeOut){ 
        printf("%d*%d matrix error coefficient is %lf, iterations: %d\n",
                                                             N, N, res, countIt); 
        printf("That took %lf seconds\n", end - start); 
    } else 
        printf("it took more than %lf seconds so I killed the process," 
                            "error coefficient was %lf\n", timeLimit, res); 
}

void final_free(double* A, double* X, double* b, double* final_vect_res,
                double* ABuf, double* XBuf, double* final_vect_resBuf, 
                double* piece, double* bBuf, double* Axn_minus_b_buffer){
    free(A);
    free(X);
    free(b);                
    free(final_vect_res);
    free(ABuf);
    free(XBuf);
    free(final_vect_resBuf);
    free(piece);
    free(bBuf);
    free(Axn_minus_b_buffer);
}

void initBlock(double* A, double* X, double* b){
    generateMatrix(A);
    generateVector(X);
    generateVector(b);
}

int main(int argc, char** argv){
    srand(time(0));
    int processRank, sizeOfCluster;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &sizeOfCluster);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    bool timeOut = false;
    int countIt = 0, fixedCutMatrixSize = N * N / sizeOfCluster, fixedCutVectorSize = N / sizeOfCluster;
    double start = 0, end = 0, currentTime = 0, norm_Axn_minus_b = 0, pieceOfNorm = 0,
                                     εSquard_mult_normb = 0, last_norm = INFINITY, normbBuf = 0;
    double* A = (double*)malloc(N * N * sizeof(double));
    double* ABuf = (double*)malloc(fixedCutMatrixSize * sizeof(double));
    double* X = (double*)malloc(N * sizeof(double));
    double* XBuf = (double*)malloc(fixedCutVectorSize * sizeof(double));
    double* final_vect_res = (double*)malloc(N * sizeof(double));
    double* final_vect_resBuf = (double*)malloc(fixedCutVectorSize * sizeof(double));
    double* piece = (double*)malloc(N * sizeof(double));
    double* b = (double*)malloc(N * sizeof(double));
    double* bBuf = (double*)malloc(fixedCutVectorSize* sizeof(double));
    double* Axn_minus_b_buffer = (double*)malloc(fixedCutVectorSize * sizeof(double));

    if(processRank == 0)
        initBlock(A, X, b);    
/* 
Example: sizeOfCluster == 4:

        +-------+   +-------+ 
        |___1___|   | | | | | 
        |___2___|   | | | | |   - when we use MPI_Scatter().
        |___3___| ~ |1|2|3|4|     1st picture is real affect of function but in cause of symmetry,
        |___4___|   | | | | |     result will be equal.
        +-------+   +-------+ 
                           
                        1 process piece
                          <---------->
        +-------+   +-+     +-+                    +-+                        +---------+
        | | | | |   |A|     | |                    | |                        |         |
        | | | | |   |B|     | |   +-+              | |   +-+    AllReduce()   |final_   |
        |1|2|3|4| X |C|   = |1| X |A|   +  ... +   |4| X |D|   =============  |vect_res |
        | | | | |   |D|     | |   +-+              | |   +-+                  |         |
        +-------+   +-+     +-+                    +-+                        +---------+

        so, other operations, like sum, sub, scalmul - independent from order. So final result just summ 
        of gotten from each part.
 */ 

    start = MPI_Wtime();

    MPI_Scatter(b, fixedCutVectorSize, MPI_DOUBLE, bBuf,
                   fixedCutVectorSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    normbBuf = EuclideanNorm(bBuf, fixedCutVectorSize) * εSquard;
    MPI_Allreduce(&normbBuf, &εSquard_mult_normb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


    MPI_Scatter(A, fixedCutMatrixSize, MPI_DOUBLE, ABuf, 
                   fixedCutMatrixSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(X, fixedCutVectorSize, MPI_DOUBLE, XBuf,
                   fixedCutVectorSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    do {
        // MPI_Scatter(); - logically. But why use that if we don't gather xn 
        mul(ABuf, XBuf, piece, fixedCutMatrixSize); // mul for 1 piece calculating
        MPI_Allreduce(piece, final_vect_res, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // for final vect res taking         
        MPI_Scatter(final_vect_res, fixedCutVectorSize, MPI_DOUBLE, final_vect_resBuf,
                                    fixedCutVectorSize, MPI_DOUBLE, 0, MPI_COMM_WORLD); // final vect res cutting
        sub(final_vect_resBuf, bBuf, Axn_minus_b_buffer, fixedCutVectorSize); // parallel diff calculating for [A*xn - b] (we getting pieces of that) 
        /*~~~~~~~~~~~~~~~~~~~ ACCENT PAUSE FOR NORM CALCULATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
        pieceOfNorm = EuclideanNorm(Axn_minus_b_buffer, fixedCutVectorSize); // taking norm of piece
        MPI_Allreduce(&pieceOfNorm, &norm_Axn_minus_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // sum by each piece => norm_Axn_minus_b 
        /*~~~~~~~~~~~~~~~~~~~~~~ CONTINUE TO CALCULATE X_{n + 1} ~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
        // since final vect res was cut we make it work parallel
        scalMulTau(Axn_minus_b_buffer, fixedCutVectorSize); // final vect res scalar multiplication
        sub(XBuf, Axn_minus_b_buffer, XBuf, fixedCutVectorSize); // [xn - τ * (A*xn - b)] -- xn was cut, τ * (final vect res) - also was cut
        /*~~~~~~~~~~~~~~~~~~~~ CONGRATULATIONS - WE COUNTED X_{n + 1} ~~~~~~~~~~~~~~~~~~~~~~*/
        /*vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv CHECKPOINT vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv*/     
        if(processRank == 0){
            ++countIt;
            if ((countIt > 100000 && last_norm > norm_Axn_minus_b) 
                                                ||  (norm_Axn_minus_b == INFINITY)){ 
                printf("Does not converge\n"); 
                timeOut = true; 
            }
        } 

        memset(final_vect_res, 0, N * sizeof(double)),
        last_norm = norm_Axn_minus_b;
        norm_Axn_minus_b = 0;

        currentTime = MPI_Wtime();

        if((currentTime - start) > timeLimit)
            timeOut = true;      

        MPI_Bcast(&countIt, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&timeOut, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    } while (last_norm > εSquard_mult_normb && !timeOut /* && countIt < 10 */);

    MPI_Allgather(XBuf, fixedCutVectorSize, MPI_DOUBLE, X,
                        fixedCutVectorSize, MPI_DOUBLE, MPI_COMM_WORLD); // gathering of X
    end = MPI_Wtime();

    if(processRank == 0){
        final_out(timeOut, ε, countIt, end, start);  // outputing of the statistic log
    }
    
    final_free(A, X, b, final_vect_res, ABuf,
               XBuf, final_vect_resBuf,
               piece, bBuf, Axn_minus_b_buffer);
    MPI_Finalize(); 
    return 0;
<<<<<<< HEAD
}
=======
}
>>>>>>> af1b601513f20eb7f6dd0e73aabf1a65dfe0a7da
