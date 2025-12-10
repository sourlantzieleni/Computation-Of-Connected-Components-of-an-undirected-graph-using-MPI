#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include "mmio.h"
#include "converter.h"

struct timespec t_start, t_end;

int world_size;
int world_rank;

int main(int argc, char *argv[]) {
    int *global_indexes = NULL;
    int *global_indices = NULL;
    int N = 0;
    
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <graph_file.mtx>\n", argv[0]);
        return 1;
    }
    
    FILE *test = fopen(argv[1], "r");
    if (test == NULL) {
        fprintf(stderr, "Error: Cannot open file %s\n", argv[1]);
        return 1;
    }
    fclose(test);
    
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    char *str = argv[1];
    
    if (world_rank == 0) {
        N = cooReader(str, &global_indexes, &global_indices);
        N = N - 1;
    }
    
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (world_rank != 0) {
        global_indexes = (int*)malloc((N + 1) * sizeof(int));
    }
    MPI_Bcast(global_indexes, N + 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int total_edges = global_indexes[N];
    if (world_rank != 0) {
        global_indices = (int*)malloc(total_edges * sizeof(int));
    }
    MPI_Bcast(global_indices, total_edges, MPI_INT, 0, MPI_COMM_WORLD);
    
    int base_size = N / world_size;
    int remainder = N % world_size;
    int local_start, local_n;
    
    if (world_rank < remainder) {
        local_n = base_size + 1;
        local_start = world_rank * local_n;
    } else {
        local_n = base_size;
        local_start = remainder * (base_size + 1) + (world_rank - remainder) * base_size;
    }
    
    int *colors = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        colors[i] = i;
    }
    
    if (world_rank == 0) {
        clock_gettime(CLOCK_REALTIME, &t_start);
    }
    
    int active = 1;
    int iteration = 0;
    
    while (active) {
        active = 0;
        
        for (int i = local_start; i < local_start + local_n; ++i) {
            for (int j = global_indexes[i]; j < global_indexes[i+1]; ++j) {
                int neighbor = global_indices[j];
                
                int cmin;
                if(colors[i] < colors[neighbor]){
                    cmin = colors[i];
                } else {
                    cmin = colors[neighbor];
                }
                
                if(colors[i] != cmin){
                    colors[i] = cmin;
                    active = 1;
                }
                if(colors[neighbor] != cmin){
                    colors[neighbor] = cmin;
                    active = 1;
                }
            }
        }
        
        int *global_min_colors = (int*)malloc(N * sizeof(int));
        MPI_Allreduce(colors, global_min_colors, N, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        for (int i = 0; i < N; i++) {
            if (colors[i] != global_min_colors[i]) {
                active = 1;
            }
            colors[i] = global_min_colors[i];
        }
        free(global_min_colors);
        
        int global_active;
        MPI_Allreduce(&active, &global_active, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        active = global_active;
        
        iteration++;
    }
    
    if (world_rank == 0) {
        clock_gettime(CLOCK_REALTIME, &t_end);
        
        int numCC = 0;
        int *uniqueFlags = (int*)calloc(N, sizeof(int));
        for (int i = 0; i < N; i++) {
            if (!uniqueFlags[colors[i]]) {
                uniqueFlags[colors[i]] = 1;
                numCC++;
            }
        }
        
        printf("Number of connected components: %d\n", numCC);
        
        double duration = ((t_end.tv_sec - t_start.tv_sec) * 1000000.0 + 
                          (t_end.tv_nsec - t_start.tv_nsec) / 1000.0) / 1000000.0;
        printf("~ CC duration: %lf seconds\n", duration);
        
        FILE *f = fopen("results.csv", "a");
        if (f != NULL) {
            fprintf(f, "mpi,%d,%lf,%s,%d\n", world_size, duration, str, N);
            fclose(f);
        }
        
        free(uniqueFlags);
    }
    
    free(colors);
    free(global_indexes);
    free(global_indices);
    
    MPI_Finalize();
    return 0;
}