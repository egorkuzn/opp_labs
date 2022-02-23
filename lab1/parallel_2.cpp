#include <iostream> 
#include <cmath> 
#include <mpich/mpi.h> 

const long int N = 6400; 
const double ε = 1e-5; 
double τ = 1e-4;  
const int timeLimit = 70;

double EuclideanNorm(const double* u){ 
   double norm = 0; 

   for (int i = 0; i < N; i++) 
       norm += u[i]*u[i]; 

   return norm; 
} 

void sub(double* a, double* b, double* c, int n){ 
   for (int i = 0; i < n; ++i)
       c[i] = a[i] - b[i]; 
} 

void InitialMul(double* A, double* b, double* result, int n) { 
   for(unsigned int i = 0; i < n; ++i){ 
       result[i] = 0; 

       for(unsigned int j = 0; j < N; ++j)
           result[i] += A[i * N + j] * b[j]; 
   } 
} 

void mul(double* A, double* vec, double* result, int rowcount) { 
   for (int i = 0; i < N; ++i){ 
       result[i] = 0; 

       for (int j = 0; j < rowcount; ++j) 
           result[i] += A[j * N + i] * vec[j]; 
   } 
} 

void scalMul(double* A, int n){ 
   for (int i = 0; i < n; ++i)  
       A[i] = A[i] * τ; 
} 

double drand(double low, double high) { 
   double f = double(rand()) / RAND_MAX; 
   return low + f * (high - low); 
} 

double rand_double(){ 
   return (double)rand() / RAND_MAX * 4.0 - 2.0; 
} 

void generate_matrix(double* matrix) { 
   for(int i = 0; i < N; ++i){ 
       for(int j = 0; j < i; ++j) 
           matrix[i * N + j] = matrix[j * N + i]; 

       for(int j = i; j < N; ++j){ 
           matrix[i * N + j] = rand_double(); 

           if(i == j) 
               matrix[i * N + j] = fabs(matrix[i * N + j]) + 400.0; 
       } 
   } 
} 

void printVector(const double* B, const char* name,
                                 int procRank, int procNum, int n){ 
   for (int numProc = 0; numProc < procNum; ++numProc) { 
       if (procRank == numProc) { 
           printf("%s in rank %d:\n", name, procRank); 

           for (int i = 0; i < n; ++i)  
               printf("%lf\n", B[i]); 

           printf("\n"); 
       } 
   } 
} 

void buf_free(double* nextXBuf, double* xBuf, double* ABuf,
                        double* AxBuf, double* AxMulRes, double* bBuf){
   free(nextXBuf); 
   free(xBuf); 
   free(ABuf); 
   free(AxBuf); 
   free(AxMulRes); 
   free(bBuf); 
} 

void final_free(double* A, double* b, double* nextX,
                                     double* prevX, double* Ax){
    free(A); 
    free(b); 
    free(nextX); 
    free(prevX); 
    free(Ax);
}

int main(int argc, char** argv) { 
   srand(time(0)); 
   int process_Rank, size_Of_Cluster; 

   MPI_Init(&argc, &argv); 
   MPI_Comm_size(MPI_COMM_WORLD, &size_Of_Cluster); 
   MPI_Comm_rank(MPI_COMM_WORLD, &process_Rank); 
   // process shared buffers 
   double* ABuf  = (double*)malloc(N*N / size_Of_Cluster * sizeof(double)); 
   double* xBuf  = (double*)malloc(N / size_Of_Cluster * sizeof(double)); 
   double* nextXBuf  = (double*)malloc(N / size_Of_Cluster * sizeof(double)); 
   double* AxBuf = (double*)malloc(N / size_Of_Cluster * sizeof(double)); 
   double* AxMulRes = (double*)malloc(N * sizeof(double));
   double* bBuf =  (double*)malloc(N / size_Of_Cluster * sizeof(double)); 
   double* Ax = (double*)malloc(N * sizeof(double)); 

   double* prevX = nullptr, *nextX = nullptr, *A = nullptr, *b = nullptr; 
   double startTime = 0, endTime = 0, currentTime = 0, normAxb = 0; 
   double normb = 0, saveRes = 1, res = 1, lastres = 1; 

   bool timeOut = false; 

   if (process_Rank == 0){ 
       b = (double*)malloc(N * sizeof(double)); 
       prevX = (double*)malloc(N * sizeof(double)); 
       nextX = (double*)malloc(N * sizeof(double)); 

       for (long i = 0; i < N; ++i) { 
           prevX[i] = drand(1,5); 
           nextX[i] = drand(1,5); 
           b[i] = drand(1,N); 
       } 

       A = (double*)malloc(N * N * sizeof(double)); 

       generate_matrix(A); 
       InitialMul(A, prevX, Ax, N); // A*xn 
       normb = EuclideanNorm(b); 
   } 

   MPI_Bcast(&res,     1, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
   MPI_Bcast(&normb,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
   MPI_Bcast(&saveRes, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
   MPI_Scatter(A, N * N / size_Of_Cluster, MPI_DOUBLE, ABuf, 
                    N * N / size_Of_Cluster, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
   MPI_Scatter(b, N / size_Of_Cluster, MPI_DOUBLE, bBuf, 
                    N / size_Of_Cluster, MPI_DOUBLE, 0, MPI_COMM_WORLD); 

   int countIt = 1; 
   startTime = MPI_Wtime(); 

   while (res > ε * ε && !timeOut){ 
       MPI_Scatter(prevX, N / size_Of_Cluster, MPI_DOUBLE, xBuf, 
                   N / size_Of_Cluster, MPI_DOUBLE, 0, MPI_COMM_WORLD); 

       mul(ABuf, xBuf, AxMulRes, N / size_Of_Cluster); // A*xn 

       MPI_Allreduce(AxMulRes, Ax, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
       MPI_Scatter(Ax, N / size_Of_Cluster, MPI_DOUBLE, AxBuf, 
                   N / size_Of_Cluster, MPI_DOUBLE, 0, MPI_COMM_WORLD);

       sub(AxBuf, bBuf, AxBuf, N / size_Of_Cluster); // A*xn - b 
       scalMul(AxBuf, N / size_Of_Cluster); // t*(A*xn - b) 
       sub(xBuf, AxBuf, nextXBuf, N / size_Of_Cluster); // xn - t * (A*xn - b) 

       MPI_Gather(nextXBuf, N/size_Of_Cluster, MPI_DOUBLE, prevX, 
                    N/size_Of_Cluster, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
       MPI_Gather(AxBuf, N/size_Of_Cluster, MPI_DOUBLE, Ax, 
                  N/size_Of_Cluster, MPI_DOUBLE, 0, MPI_COMM_WORLD); 

       if(process_Rank == 0){ 
           normAxb = EuclideanNorm(Ax); // ||A*xn - b|| 
           res = normAxb / normb; 
           countIt++; 

           if ((countIt > 100000 && lastres > res) || res == INFINITY)
               if (τ < 0){ 
                   printf("Does not converge\n"); 
                   saveRes = res; 
                   res = 0; 
               } else{ 
                   τ *= -1; 
                   countIt = 0; 
               } 

           lastres = res; 
       } 

       currentTime = MPI_Wtime(); 

       if ((currentTime - startTime) > timeLimit) 
           timeOut = true; 

       MPI_Bcast(&countIt, 1, MPI_INT, 0, MPI_COMM_WORLD); 
       MPI_Bcast(&res,     1, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
       MPI_Bcast(&lastres, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
       MPI_Bcast(&τ,     1, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
       MPI_Bcast(&saveRes, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
   } 

   endTime = MPI_Wtime(); 

   if (process_Rank == 0){ 
       if (res == 0) 
           res = saveRes; 

       if (!timeOut){ 
           printf("%ld*%ld matrix error coefficient is %lf, iterations: %d\n",N, N, sqrt(res), countIt); 
           printf("That took %lf seconds\n",endTime-startTime); 
       } else 
           printf("it took more than %d seconds so I killed the proccess," 
                  "error coefficient was %lf\n", timeLimit, res); 

       final_free(A, b, nextX, prevX, Ax); 
   } 

   buf_free(nextXBuf, xBuf, ABuf, AxBuf, AxMulRes, bBuf);
   MPI_Finalize(); 
   return 0; 
} 
