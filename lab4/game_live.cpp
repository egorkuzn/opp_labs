#include <iostream>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi/mpi.h>
#include <malloc.h>
#include <string.h>
#include <limits.h>

struct NUM{
    int LINES;
    int COLUMNS;
    int UINTS;
} NUM;

const int MAX_ITERATION = 100000;

typedef unsigned int u_int;
const size_t UINT_BIT_SIZE = sizeof(u_int) * 8;

u_int getBit(u_int line, u_int index) {
    return (line >> index) & 1;
}

/*
 * 0 0 -> 0
 * 0 1 -> 1
 * 1 0 -> 0
 * 1 1 -> 1
 */

u_int setBit(u_int line, u_int index, u_int val) {
    return (line & (UINT_MAX - (1 << index))) + (val << index);
}

void printUINT(u_int num) {
    for (int i = 0; i < sizeof(u_int) * 8; i++)
        perror((num >> i) & 1);
}

void Size_INIT(char** argv){
    NUM.LINES = atoi(argv[1]);
    NUM.COLUMNS = atoi(argv[2]);
    NUM.UINTS = NUM.COLUMNS / (UINT_BIT_SIZE) + (NUM.COLUMNS % (UINT_BIT_SIZE) != 0);
}

void init(u_int *stage) {
    stage[0] = setBit(stage[0], 1, 1);
    stage[NUM.UINTS] = setBit(stage[NUM.UINTS], 2, 1);
    stage[2 * NUM.UINTS] = setBit(stage[2 * NUM.UINTS], 0, 7);
}

int takingCentralNumAlive(u_int* stage, int i, int j){
    /*
             *      order of checking neighbours
             *
             *      3 4 5
             *      2 c 1
             *      6 7 8
    */
    u_int num_alive = 0;

    num_alive += getBit(stage[i       * NUM.UINTS + (j + 1) * (j != NUM.COLUMNS - 1) / UINT_BIT_SIZE], (j + 1) * (j != NUM.COLUMNS - 1) % UINT_BIT_SIZE);
    num_alive += getBit(stage[i       * NUM.UINTS + (j - 1 + (!j) * NUM.COLUMNS)     / UINT_BIT_SIZE], (j - 1 + (!j) * NUM.COLUMNS)     % UINT_BIT_SIZE);
    num_alive += getBit(stage[(i - 1) * NUM.UINTS + (j - 1 + (!j) * NUM.COLUMNS)     / UINT_BIT_SIZE], (j - 1 + (!j) * NUM.COLUMNS)     % UINT_BIT_SIZE);
    num_alive += getBit(stage[(i - 1) * NUM.UINTS +  j                               / UINT_BIT_SIZE],  j                               % UINT_BIT_SIZE);
    num_alive += getBit(stage[(i - 1) * NUM.UINTS + (j + 1) * (j != NUM.COLUMNS - 1) / UINT_BIT_SIZE], (j + 1) * (j != NUM.COLUMNS - 1) % UINT_BIT_SIZE);
    num_alive += getBit(stage[(i + 1) * NUM.UINTS + (j - 1 + (!j) * NUM.COLUMNS)     / UINT_BIT_SIZE], (j - 1 + (!j) * NUM.COLUMNS)     % UINT_BIT_SIZE);
    num_alive += getBit(stage[(i + 1) * NUM.UINTS +  j                               / UINT_BIT_SIZE],  j                               % UINT_BIT_SIZE);
    num_alive += getBit(stage[(i + 1) * NUM.UINTS + (j + 1) * (j != NUM.COLUMNS - 1) / UINT_BIT_SIZE], (j + 1) * (j != NUM.COLUMNS - 1) % UINT_BIT_SIZE);
    
    return num_alive;
}

int takingBoarderUpNumAlive(u_int* stage, int j, u_int* cells_up){
    int num_alive = 0;
    
    num_alive += getBit(stage[             (j + 1) * (j != NUM.COLUMNS - 1) / UINT_BIT_SIZE],                 (j + 1) * (j != NUM.COLUMNS - 1) % UINT_BIT_SIZE);
    num_alive += getBit(stage[             (j - 1 + (!j) * NUM.COLUMNS)     / UINT_BIT_SIZE],                 (j - 1 + (!j) * NUM.COLUMNS)     % UINT_BIT_SIZE);
    num_alive += getBit(cells_up[          (j - 1 + (!j) * NUM.COLUMNS)     / UINT_BIT_SIZE],                 (j - 1 + (!j) * NUM.COLUMNS)     % UINT_BIT_SIZE);
    num_alive += getBit(cells_up[           j                               / UINT_BIT_SIZE],                  j                               % UINT_BIT_SIZE);
    num_alive += getBit(cells_up[(j + 1) * (j     !=       NUM.COLUMNS - 1) / UINT_BIT_SIZE], (j + 1)    *    (j       !=     NUM.COLUMNS - 1) % UINT_BIT_SIZE);
    num_alive += getBit(stage[NUM.UINTS  + (j - 1 + (!j) * NUM.COLUMNS)     / UINT_BIT_SIZE], (j + UINT_BIT_SIZE - 1 + (!j) * NUM.COLUMNS)     % UINT_BIT_SIZE);
    num_alive += getBit(stage[NUM.UINTS  +  j                               / UINT_BIT_SIZE],  j                                               % UINT_BIT_SIZE);
    num_alive += getBit(stage[NUM.UINTS  + (j + 1) * (j != NUM.COLUMNS - 1) / UINT_BIT_SIZE], (j + 1)    *    (j       !=     NUM.COLUMNS - 1) % UINT_BIT_SIZE);

    return num_alive;
}

int takingBoarderDownNumAlive(u_int* stage, int j, int num_lines, u_int* cells_down){
    int num_alive = 0;

    num_alive += getBit(stage[     (num_lines - 1) * NUM.UINTS + (j + 1) * (j != NUM.COLUMNS - 1) / UINT_BIT_SIZE], (j + 1) * (j != NUM.COLUMNS - 1) % UINT_BIT_SIZE);
    num_alive += getBit(stage[     (num_lines - 1) * NUM.UINTS + (j - 1 + (!j) * NUM.COLUMNS)     / UINT_BIT_SIZE], (j - 1 + (!j) * NUM.COLUMNS)     % UINT_BIT_SIZE);
    num_alive += getBit(stage[     (num_lines - 2) * NUM.UINTS + (j - 1 + (!j) * NUM.COLUMNS)     / UINT_BIT_SIZE], (j - 1 + (!j) * NUM.COLUMNS)     % UINT_BIT_SIZE);
    num_alive += getBit(stage[     (num_lines - 2) * NUM.UINTS +  j                               / UINT_BIT_SIZE],  j                               % UINT_BIT_SIZE);
    num_alive += getBit(stage[     (num_lines - 2) * NUM.UINTS + (j + 1) * (j != NUM.COLUMNS - 1) / UINT_BIT_SIZE], (j + 1) * (j != NUM.COLUMNS - 1) % UINT_BIT_SIZE);
    num_alive += getBit(cells_down[(j - 1 + (!j)   *                             NUM.COLUMNS)     / UINT_BIT_SIZE], (j - 1 + (!j) * NUM.COLUMNS)     % UINT_BIT_SIZE);
    num_alive += getBit(cells_down[ j                                                             / UINT_BIT_SIZE],  j                               % UINT_BIT_SIZE);
    num_alive += getBit(cells_down[(j + 1)         *                       (j != NUM.COLUMNS - 1) / UINT_BIT_SIZE], (j + 1) * (j != NUM.COLUMNS - 1) % UINT_BIT_SIZE);

    return num_alive;
}

void calcNextStageCenter(u_int *stage, u_int *next_stage, int num_lines) {
    for (int i = 1; i < num_lines - 1; i++) {
        for (int j = 0; j < NUM.COLUMNS; j++) {            
            u_int num_alive = takingCentralNumAlive(stage, i, j);
            u_int bit = getBit(stage[i * NUM.UINTS + j / UINT_BIT_SIZE], j % UINT_BIT_SIZE);

            if ((num_alive < 2 || num_alive > 3) && bit == 1)
                next_stage[i * NUM.UINTS + j / UINT_BIT_SIZE] = setBit(next_stage[i * NUM.UINTS + j / UINT_BIT_SIZE], j % UINT_BIT_SIZE, 0);
            else if (num_alive == 3 && bit == 0)
                next_stage[i * NUM.UINTS + j / UINT_BIT_SIZE] = setBit(next_stage[i * NUM.UINTS + j / UINT_BIT_SIZE], j % UINT_BIT_SIZE, 1);
            else
                next_stage[i * NUM.UINTS + j / UINT_BIT_SIZE] = setBit(next_stage[i * NUM.UINTS + j / UINT_BIT_SIZE], j % UINT_BIT_SIZE, bit);
        }
    }
}

void calcNextStageUp(u_int *stage, u_int *next_stage, u_int *cells_up) {
    for (int j = 0; j < NUM.COLUMNS; j++) {
        u_int num_alive = takingBoarderUpNumAlive(stage, j, cells_up);
        u_int bit = getBit(stage[j / UINT_BIT_SIZE], j % UINT_BIT_SIZE);

        if ((num_alive < 2 || num_alive > 3) && bit == 1)
            next_stage[j / UINT_BIT_SIZE] = setBit(next_stage[j / UINT_BIT_SIZE], j % UINT_BIT_SIZE, 0);
        else if (num_alive == 3 && bit == 0)
            next_stage[j / UINT_BIT_SIZE] = setBit(next_stage[j / UINT_BIT_SIZE], j % UINT_BIT_SIZE, 1);
        else
            next_stage[j / UINT_BIT_SIZE] = setBit(next_stage[j / UINT_BIT_SIZE], j % UINT_BIT_SIZE, bit);
    }
}

void calcNextStageDown(u_int *stage, u_int *next_stage, int num_lines, u_int *cells_down) {
    for (int j = 0; j < NUM.COLUMNS; j++) {
        u_int num_alive = takingBoarderDownNumAlive(stage, j, num_lines, cells_down);
        

        u_int bit = getBit(stage[(num_lines - 1) * NUM.UINTS + j / UINT_BIT_SIZE], j % UINT_BIT_SIZE);
        if ((num_alive < 2 || num_alive > 3) && bit == 1)
            next_stage[(num_lines - 1) * NUM.UINTS + j / UINT_BIT_SIZE] = setBit(next_stage[(num_lines - 1) * NUM.UINTS + j / UINT_BIT_SIZE], j % UINT_BIT_SIZE, 0);
        else if (num_alive == 3 && bit == 0)
            next_stage[(num_lines - 1) * NUM.UINTS + j / UINT_BIT_SIZE] = setBit(next_stage[(num_lines - 1) * NUM.UINTS + j / UINT_BIT_SIZE], j % UINT_BIT_SIZE, 1);
        else
            next_stage[(num_lines - 1) * NUM.UINTS + j / UINT_BIT_SIZE] = setBit(next_stage[(num_lines - 1) * NUM.UINTS + j / UINT_BIT_SIZE], j % UINT_BIT_SIZE, bit);
    }
}


bool checkPrevStage(u_int *prev_stage, u_int *stage, int num_lines) {
    for (int i = 0; i < num_lines; i++)
        for (int j = 0; j < NUM.UINTS; j++)
            if (stage[i * NUM.UINTS + j] != prev_stage[i * NUM.UINTS + j])
                return false;

    return true;
}

void initArrays(int* strings_in_tasks, int* shifts, int* to_send, int sizeOfCluster){
    for (int i = 0; i < sizeOfCluster; i++) {
        strings_in_tasks[i] = NUM.LINES / sizeOfCluster;

        if (i < NUM.LINES % sizeOfCluster) {
            strings_in_tasks[i] += 1;
        }
    }

    int start_index = 0;
    for (int i = 0; i < sizeOfCluster; i++) {
        shifts[i] = start_index;
        start_index += strings_in_tasks[i] * NUM.UINTS;
    }

    for (int i = 0; i < sizeOfCluster; i++) {
        to_send[i] = NUM.UINTS * strings_in_tasks[i];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        perror("Wrong amount of arguments: expected 3");
        return 1;
    }

    int sizeOfCluster, rank;
    MPI_Init(&argc, &argv);
    Size_INIT(argv);
    MPI_Comm_size(MPI_COMM_WORLD, &sizeOfCluster);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    int* strings_in_tasks = (int*)calloc(sizeOfCluster, sizeof(int)),
       * shifts           = (int*)calloc(sizeOfCluster, sizeof(int)),
       * to_send          = (int*)calloc(sizeOfCluster, sizeof(int));

    initArrays(strings_in_tasks, shifts, to_send, sizeOfCluster);

    u_int* prev_stages[MAX_ITERATION];
    size_t full_field_size = NUM.LINES * NUM.UINTS;
    size_t field_size = strings_in_tasks[rank] * NUM.UINTS;
    u_int* stage      = (u_int*)calloc(field_size, sizeof(u_int));
    u_int* next_stage = (u_int*)calloc(field_size, sizeof(u_int));

    double start_time = MPI_Wtime();
    u_int *full_field;
    if (rank == 0 ) {
        full_field = (u_int *)calloc(full_field_size, sizeof(u_int));
        init(full_field);
    }

    MPI_Scatterv(full_field, to_send, shifts, MPI_UNSIGNED, stage, NUM.UINTS * strings_in_tasks[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (rank == 0)
        free(full_field);

    u_int *cells_up = (u_int *)calloc(NUM.UINTS, sizeof(u_int));
    u_int *cells_down = (u_int *)calloc(NUM.UINTS, sizeof(u_int));
    int period[MAX_ITERATION];
    int global_period[MAX_ITERATION];

    int current_iteration = 0;
    int is_global_period = 0;
    while (current_iteration < MAX_ITERATION) {
        MPI_Request req_recv_up, req_recv_down;
        MPI_Irecv(cells_up, NUM.UINTS, MPI_UNSIGNED, (rank - 1 + sizeOfCluster) % sizeOfCluster, 0, MPI_COMM_WORLD, &req_recv_up);
        MPI_Irecv(cells_down, NUM.UINTS, MPI_UNSIGNED, (rank + 1) % sizeOfCluster, 1, MPI_COMM_WORLD, &req_recv_down);

        MPI_Request req_send[2];
        MPI_Isend(stage, NUM.UINTS, MPI_UNSIGNED, (rank - 1 + sizeOfCluster) % sizeOfCluster, 1, MPI_COMM_WORLD, req_send);
        MPI_Isend(&stage[(strings_in_tasks[rank] - 1) * NUM.UINTS], NUM.UINTS, MPI_UNSIGNED, (rank + 1) % sizeOfCluster, 0, MPI_COMM_WORLD, &req_send[1]);

        prev_stages[current_iteration] = (u_int *) calloc(field_size, sizeof(u_int));
        memcpy(prev_stages[current_iteration], stage, field_size * sizeof(u_int));

        calcNextStageCenter(stage, next_stage, strings_in_tasks[rank]);
        bool up_done = false;
        bool down_done = false;
        int suc_up;
        int suc_down;

        while (!up_done || !down_done) {
            MPI_Test(&req_recv_up, &suc_up, MPI_STATUS_IGNORE);
            if (suc_up && !up_done) {
                calcNextStageUp(stage, next_stage, cells_up);
                up_done = true;
            }

            MPI_Test(&req_recv_down, &suc_down, MPI_STATUS_IGNORE);

            if (suc_down && !down_done) {
                calcNextStageDown(stage, next_stage, strings_in_tasks[rank], cells_down);
                down_done = true;
            }
        }

        memcpy(stage, next_stage, field_size * sizeof(u_int));
        MPI_Waitall(2, req_send, MPI_STATUS_IGNORE);

        for (int i = current_iteration; i > -1; i--) {
            if (checkPrevStage(prev_stages[i], stage, strings_in_tasks[rank]))
                period[i] = 1;
            else
                period[i] = 0;
        }

        is_global_period = 0;
        MPI_Allreduce(&period, &global_period, current_iteration + 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        for (int i = current_iteration; i >= 0; i--)
            if (global_period[i] == sizeOfCluster) {
                is_global_period = global_period[i];
                break;
            }

        if (is_global_period == sizeOfCluster)
            break;

        current_iteration += 1;
    }

    double end_time = MPI_Wtime();

    if (is_global_period && rank == 0) {
        std::cout << "size = " << sizeOfCluster << std::endl;
        std::cout << "Periodic" << std::endl;
        std::cout << "Time taken: " << end_time - start_time << std::endl << std::endl;
    }

    for (int i = 0; i < current_iteration; i++)
        free(prev_stages[i]);

    free(cells_up);
    free(cells_down);

    free(stage);
    free(next_stage);
    free(strings_in_tasks);
    free(shifts);
    free(to_send);
    
    MPI_Finalize();

    return 0;
}