#include <iostream> 
#include <cmath> 
#include <mpich/mpi.h> 

const long int N = 6400;
const double e = 1e-5;
float t = 1e-4;
const int timeLimit = 120; 
int size_Of_Cluster, process_Rank;

double EuclideanNorm(const double* u, const long int n){
    double norm = 0;

    for (int i = 0; i < n; i++)
        norm += u[i] * u[i];    
    
    return norm;
}  

void printVector(const double* B, const char* name,
                         int procRank, int procNum, int n){ 
   for (int numProc = 0; numProc < procNum; ++numProc) 
       if (procRank == numProc) { 
           printf("%s in rank %d:\n", name, procRank); 

           for (int i = 0; i < n; ++i)
               printf("%9lf ", B[i]);

           printf("\n"); 
       }     
}

void sub(double* a, double* b, double* c, int n){
    for (int i = 0; i < n; i++)
        c[i] = a[i] - b[i];
} 

void mul(double* A, double* b, double* result, int n) { 
   for(unsigned int i = 0; i < n; i++){ 
       result[i] = 0; 

       for(unsigned int j = 0; j < N; j++){ 
           result[i] += A[i * N + j] * b[j];
       } 
   } 
}  

void scalMul(double* A, int n){ 
   for (int i = 0; i < n; ++i) { 
       A[i] = A[i] * t; 
   } 
}  

double drand(double low, double high) { 
   double f = (double)rand() / RAND_MAX; 
   return low + f * (high - low); 
}  

double rand_double(){ 
   return (double)rand()/RAND_MAX * 4.0 - 2.0; 
}  

void generate_matrix(double* matrix) { 
   for(int i = 0; i < N; i++){ 
       for(int j = 0; j < i; j++)
           matrix[i * N + j] = matrix[j * N + i];        

       for(int j = i; j < N; j++){ 
           matrix[i * N + j] = rand_double(); 
           if(i == j)
               matrix[i * N + j] = fabs(matrix[i * N + j]) + 200.0;   
       } 
   } 
}  

void free_block(double* ABuf, double* Ax, double* nextX,
                double* prevX, double* b, double* AxBuf){
   free(ABuf);
   free(Ax); 
   free(nextX); 
   free(prevX); 
   free(b); 
   free(AxBuf); 
}

void final_out(bool timeOut, float res, int countIt, double end, double start){
    if (!timeOut){ 
        printf("%ld*%ld matrix error coefficient is %lf, iterations: %d\n",
                                                             N, N, sqrt(res), countIt); 
        printf("That took %lf seconds\n", end - start);         
    } else 
        printf("it took more than %d seconds so I killed the process," 
                            "error coefficient was %lf\n", timeLimit, res); 
}

int main(int argc, char** argv) { 
   srand(time(0));  

   MPI_Init(&argc, &argv); 
   MPI_Comm_size(MPI_COMM_WORLD, &size_Of_Cluster); 
   MPI_Comm_rank(MPI_COMM_WORLD, &process_Rank); 
   // process shared buffers 
   double* ABuf  = (double*)malloc(N*N / size_Of_Cluster * sizeof(double)); // free 
   double* AxBuf = (double*)malloc(N / size_Of_Cluster * sizeof(double)); // free 

   double* b = (double*)malloc(N * sizeof(double));
   double* prevX = (double*)malloc(N * sizeof(double));// free 
   double* Ax = (double*)malloc(N * sizeof(double));// free 
   double* A = nullptr;
   double* nextX = (double*)malloc(N * sizeof(double));// free 

   double normAxb = 0, normb = 0, saveRes = 1, res = 1;
   double lastres = 1; bool timeOut = false; int countIt = 1;
   double start = 0, currentTime = 0, end = 0; 

   start = MPI_Wtime();

   for (long i = 0; i < N; ++i) {
       prevX[i] = drand(1, 5); 
       nextX[i] = drand(1, 5); 
       b[i] = drand(1, N); 
   }  

    if (process_Rank == 0){ 
        A = (double*)malloc(N * N * sizeof(double)); // free
        generate_matrix(A); 
        mul(A, prevX, Ax, N); // A*xn        
        normb = EuclideanNorm(b, N); 
    }  

   MPI_Bcast(b,        N, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
   MPI_Bcast(Ax,       N, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
   MPI_Bcast(&res,     1, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
   MPI_Bcast(&normb,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
   MPI_Bcast(&saveRes, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); 

   MPI_Scatter(A, N * N / size_Of_Cluster, MPI_DOUBLE, ABuf, 
               N * N / size_Of_Cluster, MPI_DOUBLE, 0, MPI_COMM_WORLD);    

   while (res > e * e && !timeOut){ 
        if(process_Rank == 0){
            for (long i = 0; i < N; ++i)
                prevX[i] = nextX[i];
            
            normAxb = 0;
        }

        MPI_Bcast(prevX,    N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&normAxb, 1,    MPI_INT, 0, MPI_COMM_WORLD);
        //MPI_Bcast(nextX,    N, MPI_DOUBLE, 0, MPI_COMM_WORLD); 

        MPI_Scatter(Ax, N / size_Of_Cluster, MPI_DOUBLE, AxBuf,
                   N / size_Of_Cluster, MPI_DOUBLE, 0, MPI_COMM_WORLD);  
        /////////////////////////////////////////////////////////////////////
        mul(ABuf, prevX, AxBuf, N / size_Of_Cluster); // A*xn 
        sub(AxBuf, b, AxBuf, N / size_Of_Cluster); // A*xn - b - rewrite
        scalMul(AxBuf, N / size_Of_Cluster); // t*(A*xn - b)  
        sub(prevX, AxBuf, nextX, N / size_Of_Cluster); // xn - t * (A*xn - b) 
        normAxb += EuclideanNorm(AxBuf, N / size_Of_Cluster);
        /////////////////////////////////////////////////////////////////////
        MPI_Allgather(AxBuf, N / size_Of_Cluster, MPI_DOUBLE, Ax,  // подумать
                        N / size_Of_Cluster, MPI_DOUBLE, MPI_COMM_WORLD); 


        if(process_Rank == 0){ 
                countIt++; 
                res = normAxb / normb; 

                if ((countIt > 100000 && lastres > res) || res == INFINITY) { 
                    printf("Does not converge\n"); 
                    saveRes = res; 
                    res = 0; 
                }
        }        

        lastres = res; 
        currentTime = MPI_Wtime(); 

        if ((currentTime - start) > timeLimit)
            timeOut = true;   

        MPI_Bcast(&res,     1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&lastres, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
        MPI_Bcast(&saveRes, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); 

   } 

   end = MPI_Wtime(); 

   if (process_Rank == 0){ 
       if (res == 0)
           res = saveRes;    

       final_out(timeOut, res, countIt, end, start);
       free(A); 
   } 

   free_block(ABuf, Ax, nextX, prevX, b, AxBuf);
   MPI_Finalize(); 
   return 0; 
} 
