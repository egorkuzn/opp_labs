#include <mpi/mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>

#define LIST_AMOUNT 1000
#define TASK_AMOUNT 1024

int taskSend, taskLeft;
int threadRank, threadSize;

pthread_mutex_t mutex, sendThreadMutex, executeThreadMutex;
pthread_cond_t finishTasks, newTasksAvailable;
pthread_t sendThread, receiveThread, executeThread;

int *list;
int executedLists, executedThreadTasks, tasks;

bool sendThreadGetSignal = false;
bool executeThreadGetSignal = false;

void* sendRequest(void* r) {
    MPI_Status status;
    pthread_mutex_lock(&sendThreadMutex);

    while (true) {
        pthread_cond_wait(&finishTasks, &sendThreadMutex);
        sendThreadGetSignal = true;

        if (executedLists > LIST_AMOUNT) {
            pthread_mutex_unlock(&sendThreadMutex);
            pthread_exit(NULL);
        }

        bool sendRequestFlag = true;
        int recvTaskAmount;

        for (int i = 0; i < threadSize; i++) {
            if (i == threadRank)
                continue;

            MPI_Send(&sendRequestFlag, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Recv(&recvTaskAmount, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);

            if (recvTaskAmount == 0)
                continue;

            MPI_Recv(&(list[0]), recvTaskAmount, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
            break;
        }

        pthread_mutex_lock(&mutex);
        tasks = recvTaskAmount;
        executedThreadTasks = 0;

        while (executeThreadGetSignal == false)
            pthread_cond_signal(&newTasksAvailable);

        executeThreadGetSignal = false;
        pthread_mutex_unlock(&mutex);
    }
}

void* recvRequest(void* r) {
    MPI_Status status;

    while (true) {
        bool recvRequestFlag = false;
        MPI_Recv(&recvRequestFlag, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

        if (recvRequestFlag == 0) 
            pthread_exit(NULL);

        pthread_mutex_lock(&mutex);
        taskSend = (tasks - executedThreadTasks) / 2;
        taskLeft = (tasks - executedThreadTasks + 1) / 2;
        MPI_Send(&taskSend, 1, MPI_INT, status.MPI_SOURCE, 2, MPI_COMM_WORLD);

        if (taskSend != 0) {
            MPI_Send(&(list[executedThreadTasks + taskLeft]), taskSend, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
            tasks = executedThreadTasks + taskLeft;
        }

        pthread_mutex_unlock(&mutex);
    }
}

void* execute(void* r) {
    int tasksPerProc = TASK_AMOUNT / threadSize;
    list = (int*)malloc(tasksPerProc * sizeof(int));
    executedLists = 0;
    double iterTime = 0;
    double globalRes = 0;

    char threadRankPath[5];
    threadRankPath[0] = threadRank + '0';
    memcpy(threadRankPath + 1, ".txt", 4); 

    FILE* in = fopen(threadRankPath, "w");

    for (int listId = 0; listId < LIST_AMOUNT; listId++) {
        double start = MPI_Wtime();

        for (int i = 0; i < tasksPerProc; ++i) 
            list[i] = i * abs(threadRank - (executedLists % threadSize)) * 64;

        tasks = tasksPerProc;
        executedThreadTasks = 0;
        int totalExecutedTasks = 0;

        while(true) {
            if (tasks == 0)
                break;

            for (int taskId = 0; taskId < tasks; taskId++) {
                pthread_mutex_lock(&mutex);
                executedThreadTasks++;
                pthread_mutex_unlock(&mutex);

                for (int i = 0; i < list[taskId]; i++)
                    globalRes += cos(i);
            }

            totalExecutedTasks += executedThreadTasks;

            pthread_mutex_lock(&executeThreadMutex);

            while (sendThreadGetSignal == false)
                pthread_cond_signal(&finishTasks);

            sendThreadGetSignal = false;
            pthread_cond_wait(&newTasksAvailable, &executeThreadMutex);
            executeThreadGetSignal = true;
            pthread_mutex_unlock(&executeThreadMutex);
        }

        double end = MPI_Wtime();
        iterTime = end - start;
        int iterCounter = executedLists;

        fprintf(in, "%d - %d - %d - %f - %f\n", threadRank, iterCounter, totalExecutedTasks, globalRes, iterTime);
        executedLists++;

        MPI_Barrier(MPI_COMM_WORLD);
    }

    pthread_mutex_lock(&mutex);
    int recvRequestFlag = false;
    MPI_Send(&recvRequestFlag, 1, MPI_INT, threadRank, 0, MPI_COMM_WORLD);
    executedLists++;
    pthread_cond_signal(&finishTasks);
    pthread_mutex_unlock(&mutex);

    free(list);
    fclose(in);

    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    int providedLvl;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &providedLvl);

    if (providedLvl != MPI_THREAD_MULTIPLE) {
        MPI_Finalize();
        return -1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &threadRank);
    MPI_Comm_size(MPI_COMM_WORLD, &threadSize);

    pthread_mutex_init(&mutex, NULL);
    pthread_mutex_init(&sendThreadMutex, NULL);
    pthread_mutex_init(&executeThreadMutex, NULL);

    pthread_cond_init(&finishTasks, NULL);
    pthread_cond_init(&newTasksAvailable, NULL);

    double start = MPI_Wtime();

    pthread_attr_t attrs;
    pthread_attr_init(&attrs);

    pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE);

    pthread_create(&sendThread, &attrs, sendRequest, NULL);
    pthread_create(&receiveThread, &attrs, recvRequest, NULL);
    pthread_create(&executeThread, &attrs, execute, NULL);

    pthread_attr_destroy(&attrs);

    pthread_join(sendThread, NULL);
    pthread_join(receiveThread, NULL);
    pthread_join(executeThread, NULL);

    double end = MPI_Wtime();

    if (threadRank == 0) {
        printf("\nTIME: %lf\n", end - start);
    }

    pthread_mutex_destroy(&mutex);
    pthread_mutex_destroy(&sendThreadMutex);
    pthread_mutex_destroy(&executeThreadMutex);
    pthread_cond_destroy(&finishTasks);
    pthread_cond_destroy(&newTasksAvailable);

    MPI_Finalize();
    return 0;
}
