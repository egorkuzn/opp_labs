#include <mpi/mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

//Mask arrays:
const int HORIZONTAL[2] = {0, 1},
           VERTICAL [2] = {1, 0};
//Grid parameters box:
struct grid{
    int height,
        width,
        array[2];
} grid;
//Matrixes and their sizes:
struct matrixes{
    int N,
        M,
        K;
    double* A,
          * B,
          * C;
} matrixes;

void Grid_Init(char* height, char* width){
    grid.height = atoi(height);
    grid.width  = atoi(width);
    grid.array[1] = grid.height;
    grid.array[2] = grid.width;
}

void MatrixRead(double* matrix, int lines, int columns, FILE* in){
    for(int i = 0; i < lines * columns; ++i)
        fscanf(in, "%lf", &matrix[i]);
}

void Matrixes_Init_NMK(FILE* in){
    fscanf(in, "%d %d %d", &matrixes.N,
                           &matrixes.M,
                           &matrixes.K); 
}

void Matrixes_Init_AB(FILE* in, int* coords){
    if(coords[0] == 0 && coords[1] == 0){
        MatrixRead(matrixes.A, matrixes.N, matrixes.M, in);
        MatrixRead(matrixes.B, matrixes.N, matrixes.M, in);
    }
}

void Matrix_B_Scatter(MPI_Comm* COMM_2D, int* coords, int* root, double* B_part){
    int rank_recieve,
        coords_to_receive[2] = {0, 0};

    MPI_Datatype B_TYPE;
    MPI_Type_vector(matrixes.M, matrixes.K / grid.width, matrixes.K, MPI_DOUBLE, &B_TYPE);
    MPI_Type_commit(&B_TYPE);
    MPI_Cart_rank(*COMM_2D, coords_to_receive, root);

    for(int i = 1; i < grid.width; ++i){
        coords_to_receive[1] = i;
        MPI_Cart_rank(*COMM_2D, coords_to_receive, &rank_recieve);

        if(coords[0] == 0 && coords[1] == 0)
            MPI_Send(&matrixes.B[i * matrixes.K / grid.width], 1, B_TYPE, rank_recieve, i, *COMM_2D);
    }

    if (coords[0] == 0 && coords[1] != 0)
        MPI_Recv(B_part, matrixes.M * matrixes.K / grid.width, MPI_DOUBLE, *root, coords[1], *COMM_2D, MPI_STATUS_IGNORE);
    
    if (coords[0] == 0 && coords[1] == 0)
        for (int i = 0; i < matrixes.M; ++i) 
            memcpy(&B_part[i * matrixes.K / grid.width], &matrixes.B[i * matrixes.K], matrixes.K / grid.width * sizeof(double));  

    MPI_Type_free(B_TYPE);    
}

void Matrix_C_Gather(int root, int processRank, double* C_part, MPI_Comm* COMM_2D){
    MPI_Datatype C_TYPE;
    MPI_Type_vector(matrixes.N / grid.height, matrixes.K / grid.width, matrixes.K, MPI_DOUBLE, &C_TYPE);
    MPI_Type_commit(&C_TYPE);

    if (processRank != root)
        MPI_Send(C_part, matrixes.N / grid.height * matrixes.K / grid.width, MPI_DOUBLE, root, processRank, *COMM_2D);

    int coords_src[2] = {0, 0};

    for (int i = 0; i < grid.height; ++i) 
        for (int j = 0; j < grid.width; ++j) 
            if ((i != 0 || j != 0) && processRank == root) {
                int rank_src;
                coords_src[0] = i;
                coords_src[1] = j;
                MPI_Cart_rank(*COMM_2D, coords_src, &rank_src);
                MPI_Recv(&matrixes.C[i * matrixes.N / grid.height * matrixes.K + j * matrixes.K / grid.width], 1, C_TYPE, rank_src, rank_src, *COMM_2D, MPI_STATUS_IGNORE);
            }

    if (processRank == root)
        for (int i = 0; i < matrixes.N / grid.height; ++i)
            memcpy(&matrixes.C[i * matrixes.N], &C_part[i], matrixes.K / grid.width  * sizeof(double));
        
    
    MPI_Type_free(&C_TYPE);
}

void MatrixMul(double* A, double* B, double* C, int N, int M, int K){
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < M; ++j)
            for(int k = 0; k < K; ++k)
                C[i * K + k] += A[i * M + j] * B[j * N + k];
}

void final_free(double* A_part, double* B_part, double* C_part, MPI_Comm* COMM_HORIZONTAL, MPI_Comm* COMM_VERTICAL, MPI_Comm* COMM_2D){
    free(matrixes.A);
    free(matrixes.B);
    free(matrixes.C);

    free(A_part);
    free(B_part);
    free(C_part);

    MPI_Comm_free(COMM_HORIZONTAL);
    MPI_Comm_free(COMM_VERTICAL);
    MPI_Comm_free(COMM_2D);
}

void final_printf(double start, double end){
    printf("N:%d\tcolumns%lf\n\
            M:%d\tlines%lf\n\
            K:%d\ttime:%lf\n", matrixes.N, grid.width,
                               matrixes.M, grid.height,
                               matrixes.K, start - end);
}

//    A                 B             C
//
// <---M--->          <-K->         <-K->
//  ______             ___           ___
// |______| ^         | |X| ^       | | | ^
// |______| |N        | |X| |       | | | |
// |XXXXXX| |    X    | |X| |   =   | |X| | N
// |______| V         | |X| | M     | | | |
//                    | |X| |        ___  V
//                    | |X| |
//                     ___  V
                              
int main(int argc, char* argv[]){
    if(argc != 3){
        perror("Bad count of parameters");
        return 0;
    }

    int sizeOfCluster, processRank;
    Grid_Init(argv[1], argv[2]);
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &sizeOfCluster);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
//                    START
    double start = MPI_Wtime();
//New communicators creation:
    int periods[2] = {0, 0},
        coords [2];
    MPI_Comm COMM_2D, COMM_HORIZONTAL, COMM_VERTICAL;
    MPI_Cart_create(MPI_COMM_WORLD, 2, grid.array, periods, false, &COMM_2D);
    MPI_Cart_coords(COMM_2D, processRank, 2, coords);
    MPI_Cart_sub(COMM_2D, HORIZONTAL, &COMM_HORIZONTAL);
    MPI_Cart_sub(COMM_2D, VERTICAL, &COMM_VERTICAL);
//Opening resource file:
    FILE* in = fopen("matrix.txt", "r");
//Checking opening of file:    
    if(!in)
        perror("File didn't open");
//Taking matrixes from resources:    
    Matrixes_Init_NMK(in);    
    matrixes.A = (double*)calloc(matrixes.N * matrixes.M, sizeof(double));
    matrixes.B = (double*)calloc(matrixes.M * matrixes.K, sizeof(double));
    matrixes.C = (double*)calloc(matrixes.N * matrixes.K, sizeof(double));
    Matrixes_Init_AB(in, coords);
    fclose(in);
//Taking part of matrix A:
    double* A_part = (double*)calloc(matrixes.N / grid.height * matrixes.M, sizeof(double));
    MPI_Scatter(matrixes.A, matrixes.M * matrixes.N / grid.height, MPI_DOUBLE,
                    A_part, matrixes.M * matrixes.N / grid.height, MPI_DOUBLE, 0, COMM_VERTICAL);
    MPI_Bcast(A_part, matrixes.M * matrixes.N / grid.height, MPI_DOUBLE, 0, COMM_HORIZONTAL);
//Taking part of matrix B:
    double* B_part = (double*)calloc(matrixes.K / grid.width  * matrixes.M, sizeof(double));
    int root;
    Matrix_B_Scatter(&COMM_2D, coords, &root, B_part);
    MPI_Bcast(B_part, matrixes.M * matrixes.K / grid.width, MPI_DOUBLE, 0, COMM_VERTICAL);
//Taking part of matrix C:
    double* C_part = (double*)calloc(matrixes.N / grid.height * matrixes.K / grid.width, sizeof(double));
//Getting multiplication part of C:
    MatrixMul(A_part, B_part, C_part, matrixes.N / grid.height, matrixes.M, matrixes.K / grid.width);
//C gathering part:
    Matrix_C_Gather(root, processRank, C_part, &COMM_2D);
//                     END
    double end = MPI_Wtime();
//Memory deallocation:
    final_free(A_part, B_part, C_part, &COMM_HORIZONTAL, &COMM_VERTICAL, &COMM_2D);
//Final output:
    if(processRank == 0)
        final_printf(start, end);

    MPI_Finalize();   
    return 0;
}
