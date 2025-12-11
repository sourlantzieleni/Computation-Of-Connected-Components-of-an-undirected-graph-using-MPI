#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include "mmio.h"
#include "converter.h"

struct timespec t_start, t_end;

int world_size;
int world_rank;

void broadcast_large_array(int *array, int size, int root) {
   
    const int MAX_MPI_SIZE = 100000000; 

    if (size <= MAX_MPI_SIZE) {

        MPI_Bcast(array, size, MPI_INT, root, MPI_COMM_WORLD);
        if (world_rank == 0) {
            printf("Broadcasted array (%d elements) in single message\n", size);
            fflush(stdout);
        }
    } else {

        int num_chunks = (size + MAX_MPI_SIZE - 1) / MAX_MPI_SIZE;
        
        
        for (int chunk = 0; chunk < num_chunks; chunk++) {
            int offset = chunk * MAX_MPI_SIZE;
            int chunk_size = (offset + MAX_MPI_SIZE > size) ? (size - offset) : MAX_MPI_SIZE;
            
            MPI_Bcast(array + offset, chunk_size, MPI_INT, root, MPI_COMM_WORLD);
            
            if (world_rank == 0) {
                printf("  Chunk %d/%d: %d elements (%.2f MB)\n", 
                       chunk + 1, num_chunks, chunk_size,
                       (chunk_size * sizeof(int)) / (1024.0 * 1024.0));
                fflush(stdout);
            }
        }
        
        if (world_rank == 0) {
            printf("All chunks broadcasted successfully!\n");
            fflush(stdout);
        }
    }
}

int main(int argc, char *argv[]) {
    int *global_indexes = NULL;
    int *global_indices = NULL;
    int N = 0;
    int total_edges = 0;
    
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <graph_file.mtx>\n", argv[0]);
        return 1;
    }
    
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    char *str = argv[1];
    
    if (world_rank == 0) {
        
        
        FILE *test = fopen(str, "r");
        if (test == NULL) {
            fprintf(stderr, "Rank 0: Cannot open file %s\n", str);
            N = -1;
            MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Finalize();
            return 1;
        }
        fclose(test);
        
        N = cooReader(str, &global_indexes, &global_indices);
        N = N - 1;
        total_edges = global_indexes[N];
        
    }
    
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (N <= 0) {
        MPI_Finalize();
        return 1;
    }
    
        if (world_rank != 0) {
        global_indexes = (int*)malloc((N + 1) * sizeof(int));
        if (global_indexes == NULL) {
            fprintf(stderr, "Rank %d: Failed to allocate indexes\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    
    MPI_Bcast(global_indexes, N + 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    
    total_edges = global_indexes[N];
    
    if (world_rank != 0) {
        printf("Rank %d: Allocating %d elements (%.2f MB)...\n", 
               world_rank, total_edges, (sizeof(int) * total_edges) / (1024.0 * 1024.0));
        fflush(stdout);
        
        global_indices = (int*)malloc(total_edges * sizeof(int));
        if (global_indices == NULL) {
            fprintf(stderr, "Rank %d: Failed to allocate column indices\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    if (world_rank == 0) {
        printf("Rank 0: Broadcasting column indices (%d elements)...\n", total_edges);
        fflush(stdout);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    broadcast_large_array(global_indices, total_edges, 0);
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (world_rank == 0) {
        printf("Rank 0: All data broadcasted!\n");
        fflush(stdout);
    } else {
        printf("Rank %d: All data received!\n", world_rank);
        fflush(stdout);
    }
    
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