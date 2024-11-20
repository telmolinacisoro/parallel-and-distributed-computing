#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <mpi.h>

#define MAX_NUM_ELEMENTS 10000000

int main()
{
    int rank, n, rep, repetitions;
    MPI_Init(0, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Request request;
    MPI_Status stat;

    double start_time, end_time, total_time, mean_time, rate, mean_rate;
    double data[MAX_NUM_ELEMENTS];

    memset(data, 1.0, MAX_NUM_ELEMENTS);

    if (rank == 0){
        printf("n\tTime(sec)\tRate(MB/sec)\n");
    }

    for (n = 1; n < MAX_NUM_ELEMENTS; n *= 2)
    {
        if (n > 1000) repetitions = 1;
        else repetitions = 1000/n;
        
        mean_time = 0.0; // Initialize mean_time for each iteration
        mean_rate = 0.0; // Initialize mean_rate for each iteration
        
        for (rep = 1; rep <= repetitions; rep++){
            start_time = MPI_Wtime();
            if (rank == 0)
            {
                MPI_Isend(&data, n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, &stat);
                MPI_Irecv(&data, n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, &stat);
            }
            else
            {
                MPI_Irecv(&data, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, &stat);
                MPI_Isend(&data, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, &stat);
            }

            end_time = MPI_Wtime();
            total_time = end_time - start_time;
            
            mean_time += total_time;
            rate = (n * sizeof(double)) / (total_time * (1024 * 1024));
            mean_rate += rate;
        }
        mean_time /= repetitions;
        mean_rate /= repetitions; // We multiply by 2 because the time's computed as the difference between sending and receiving all the data, hence for the transfer we must consider twice the data size
        
        if (rank == 0) {
            mean_rate *= 2;
            printf("%d\t%f\t%f\n", n, mean_time, mean_rate);
        }
    }
    
    MPI_Finalize();
    return 0;
}
