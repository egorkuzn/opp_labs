#include <iostream>
#include <cstring>
#include <cmath>
#include <mpi/mpi.h>

const int N = 6400;
const double timeLimit = 120.0;
const double εSquard = 1e-5 * 1e-5;
const double τ = 1e-4;

double EuclideanNorm(const double* vector, const int size){
    double norm = 0;

    for (int i = 0; i < size; i++)
        norm += vector[i] * vector[i];

    return norm;
}

void sub(const double* from, const double* what, double* result, const int size){
    for (int i = 0; i < size; i++)
        result[i] = from[i] - what[i];
} 

void mul(double* matrix, double* vector, double* result, int size) {
    for(unsigned int i = 0; i < N; i++) {
        result[i] = 0; // init of vector elem

        for(unsigned int j = 0; j < size; j++) 
            result[i] += matrix[i * N + j] * vector[j];
    }
}

void scalMulTau(double* A){
    for (int i = 0; i < N; ++i)
        A[i] = A[i] * τ;
}

double randDouble(){ 
   return (double)rand() / RAND_MAX * 4.0 - 2.0; 
} 

void generateMatrix(double* matrix) { 
   for(int i = 0; i < N; i++){ 
       for(int j = 0; j < i; j++)
           matrix[i * N + j] = matrix[j * N + i];        

       for(int j = i; j < N; j++){ 
           matrix[i * N + j] = randDouble(); 
           if(i == j)
               matrix[i * N + j] = fabs(matrix[i * N + j]) + 200.0;   
       } 
   } 
}  

void final_out(bool timeOut, double res, int countIt, double end, double start){
    if (!timeOut){ 
        printf("%ld*%ld matrix error coefficient is %lf, iterations: %d\n",
                                                             N, N, sqrt(res), countIt); 
        printf("That took %lf seconds\n", end - start); 
    } else 
        printf("it took more than %d seconds so I killed the process," 
                            "error coefficient was %lf\n", timeLimit, res); 
}

void final_free(double* A, double* ABuf, double* X, double* XBuf, 
                double* final_vect_res, double* final_vect_resBuf,
                double* piece, double* b, double* bBuf, double* Axn_minus_b_buffer    ){
    free(A);
    free(ABuf);
    free(X);
    free(XBuf);
    free(final_vect_res);
    free(final_vect_resBuf);
    free(piece);
    free(b);
    free(bBuf);
    free(Axn_minus_b_buffer);
}

int main(int argc, char** argv){
    srand(time(0));
    int processRank, sizeOfCluster;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &sizeOfCluster);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    bool timeOut = false;
    int countIt = 0;
    double start = 0, end = 0, currentTime = 0, norm_Axn_minus_b, pieceOfNorm,
                                     εSquard_mult_normb, last_norm, normb, normbBuf;
    double* A = (double*)malloc(N * N * sizeof(double));
    double* ABuf = (double*)malloc(N * N / sizeOfCluster * sizeof(double));
    double* X = (double*)malloc(N * sizeof(double));
    double* XBuf = (double*)malloc(N / sizeOfCluster * sizeof(double));
    double* final_vect_res = (double*)malloc(N * sizeof(double));
    double* final_vect_resBuf = (double*)malloc(N / sizeOfCluster * sizeof(double));
    double* piece = (double*)malloc(N * sizeof(double));
    double* b = (double*)malloc(N * sizeof(double));
    double* bBuf = (double*)malloc(N / sizeOfCluster* sizeof(double));
    double* Axn_minus_b_buffer = (double*)malloc(N / sizeOfCluster * sizeof(double));

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

    MPI_Scatter(b, N / sizeOfCluster, MPI_DOUBLE, bBuf,
                   N / sizeOfCluster, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    normbBuf = EuclideanNorm(bBuf, N / sizeOfCluster) * εSquard;
    MPI_Allreduce(&normbBuf, &εSquard_mult_normb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


    MPI_Scatter(A, N * N / sizeOfCluster, MPI_DOUBLE, ABuf, 
                   N * N / sizeOfCluster, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(X, N / sizeOfCluster, MPI_DOUBLE, XBuf,
                   N / sizeOfCluster, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    while (last_norm > εSquard_mult_normb && !timeOut){
        // MPI_Scatter(); - logically. But why use that if we don't gather xn 
        mul(ABuf, XBuf, piece, N / sizeOfCluster); // mul for 1 piece calculating
        MPI_Allreduce(piece, final_vect_res, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // for final vect res taking 
        MPI_Scatter(final_vect_res, N / sizeOfCluster, MPI_DOUBLE, final_vect_resBuf,
                                    N / sizeOfCluster, MPI_DOUBLE, 0, MPI_COMM_WORLD); // final vect res cutting
        sub(final_vect_resBuf, bBuf, Axn_minus_b_buffer, N / sizeOfCluster); // parallel diff calculating for [A*xn - b] (we getting pieces of that) 
        /*~~~~~~~~~~~~~~~~~~~ ACCENT PAUSE FOR NORM CALCULATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
        pieceOfNorm = EuclideanNorm(Axn_minus_b_buffer, N / sizeOfCluster); // taking norm of piece
        MPI_Allreduce(&pieceOfNorm, &norm_Axn_minus_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // sum by each piece => norm_Axn_minus_b 
        /*~~~~~~~~~~~~~~~~~~~~~~ CONTINUE TO CALCULATE X_{n + 1} ~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
        // since final vect res was cut we make it work parallel
        scalMulTau(final_vect_resBuf); // final vect res scalar multiplication
        sub(XBuf, final_vect_resBuf, XBuf, N / sizeOfCluster); // [xn - τ * (A*xn - b)] -- xn was cut, τ * (final vect res) - also was cut
        /*~~~~~~~~~~~~~~~~~~~~ CONGRATULATIONS - WE COUNTED X_{n + 1} ~~~~~~~~~~~~~~~~~~~~~~*/
        /*vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv CHECKPOINT vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv*/     
        if(processRank == 0){
            if ( (++countIt > 100000 && last_norm > norm_Axn_minus_b) 
                                                || (norm_Axn_minus_b == INFINITY) ){ 
                printf("Does not converge\n"); 
                timeOut = true; 
            }

            last_norm = norm_Axn_minus_b;
            norm_Axn_minus_b = 0;
            final_vect_res = 0;
        } 

        currentTime = MPI_Wtime();

        if((currentTime - start) > timeLimit)
            timeOut = true;
        
        MPI_Bcast(&norm_Axn_minus_b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&final_vect_res, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Allgather(XBuf, N / sizeOfCluster, MPI_DOUBLE, X,
                        N / sizeOfCluster, MPI_DOUBLE, MPI_COMM_WORLD); // gathering of X
    end = MPI_Wtime();

    final_out(timeOut, norm_Axn_minus_b / normb, countIt, end, start);  // outputing of the statistic log
    final_free(A, ABuf, X, XBuf, final_vect_res,
                 final_vect_resBuf, piece, b, bBuf, Axn_minus_b_buffer); // memory free
    MPI_Finalize(); 
    return 0;
}