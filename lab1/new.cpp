#include <iostream>
#include <cmath>
#include <mpi/mpi.h>

const int N = 6400;
const double timeLimit = 120.0;
const double εSquard = 1e-5 * 1e-5;
const double τ = 1e-4;

double EuclideanNorm(const double* vector){
    double norm = 0;

    for (int i = 0; i < N; i++)
        norm += vector[i] * vector[i];

    return norm;
}

void sub(double* from, double* what, double* result, int size){
    for (int i = 0; i < size; i++)
        result[i] = from[i] - what[i];
} 

void mul(double* matrix, double* vector, double* result, int size) {
    for(unsigned int i = 0; i < N; i++) {
        result[i] = 0;

        for(unsigned int j = 0; j < size; j++) 
            result[i] += matrix[i * N + j] * vector[j];
    }
}

void scalMulTau(double* A){
    for (int i = 0; i < N; ++i)
        A[i] = A[i] * τ;
}

double randDouble(){ 
   return (double)rand()/RAND_MAX * 4.0 - 2.0; 
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

int main(int argc, char** argv){
    srand(time(0));
    int processRank, sizeOfCluster;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &sizeOfCluster);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    bool timeOut = false;
    int countIt = 0;
    double normDivision, lastNormDivision, savedNormDivision, normVector, normTotal, currentTime, startTime, normb;
    double  *array = (double*)malloc(sizeOfCluster * sizeof(double));
    double *vector = (double*)malloc(N / sizeOfCluster * sizeof(double)), // already have cut
            *xPrev = (double*)malloc(N / sizeOfCluster * sizeof(double)), // already have cut
            *xNext = (double*)malloc(N / sizeOfCluster * sizeof(double)),
                *A = (double*)malloc(N * N * sizeof(double)),
             *ABuf = (double*)malloc(N * N / sizeOfCluster * sizeof(double));

    start = MPI_Wtime();

    if(processRank == 0){
        generateMatrix(A);              
    }  

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
        | | | | |   |B|     | |   +-+              | |   +-+    AllReduce()   |final    |
        |1|2|3|4| X |C|   = |1| X |A|   +  ... +   |4| X |D|   =============  |vect res |
        | | | | |   |D|     | |   +-+              | |   +-+                  |         |
        +-------+   +-+     +-+                    +-+                        +---------+

        so, other operations, like sum, sub, scalmul - independent from order. So final result just summ 
        of gotten from each part.
        Keeping 
 */ 

    MPI_Scatter(A, N * N / sizeOfCluster, MPI_DOUBLE, ABuf, 
                   N * N / sizeOfCluster, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter();

    while (norm_Axn_minus_b > εSquard_mult_normb && !timeOut){
        // MPI_Scatter(); - logically. But why use that if we don't gather xn 
        mul(); // mul for 1 piece calculating
        MPI_Allreduce(); // for final vect res taking 
        MPI_Scatter(); // final vect res cutting
        MPI_Scatter(); // scattering of b vect
        sub(); // parallel diff calculating for [A*xn - b] (we getting pieces of that) 
        /*~~~~~~~~~~~~~~~~~~~ ACCENT PAUSE FOR NORM CALCULATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
        EuclideanNorm(); // taking norm of each pieces
        MPI_Allreduce(); // sum by each piece => norm_Axn_minus_b 
        /*~~~~~~~~~~~~~~~~~~~~~~ CONTINUE TO CALCULATE X_{n + 1} ~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
        // since final vect res was cut we make it work parallel
        scalMulTau(); // final vect res scalar multiplication
        sub(); // [xn - τ * (A*xn - b)] -- xn was cut, τ * (final vect res) - also was cut
        /*~~~~~~~~~~~~~~~~~~~~ CONGRATULATIONS - WE COUNTED X_{n + 1} ~~~~~~~~~~~~~~~~~~~~~~*/
        /*vvvvvvvvvvvvvvvvvvvvvvvv CHECKPOINT vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv*/      
        if ((process_Rank == 0) && ( (++countIt > 100000 && last_norm > norm_Axn_minus_b) || norm_Axn_minus_b == INFINITY) ){ 
            printf("Does not converge\n"); 
            timeOut = true; 
        }
    }
    MPI_Allgather(); // gathering of X_{n + 1} in one full answer
    end = MPI_Wtime();

    final_out();  // outputing of the statistic log
    final_free(); // memory free
    MPI_Finalize(); 
    return 0;
}