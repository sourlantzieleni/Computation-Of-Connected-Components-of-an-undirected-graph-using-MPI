#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include "mmio.h"
#include "converter.h"

static void get_range(int N, int p, int r, int *start, int *count) {
    int base = N / p;
    int rem  = N % p;

    if (r < rem) {
        *count = base + 1;
        *start = r * (*count);
    } else {
        *count = base;
        *start = rem * (base + 1) + (r - rem) * base;
    }
}

int main(int argc, char *argv[]) {
    int world_size, world_rank;
    int N = 0;
    int nnz = 0;

    int *global_rowptr = NULL;
    int *global_colind = NULL;

    int *local_rowptr  = NULL;
    int *local_colind  = NULL;

    int local_start = 0;
    int local_n = 0;
    int local_nnz = 0;

    int *colors = NULL;
    int *global_min_colors = NULL;

    struct timespec t_start, t_end;


    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc < 2) {
        if (world_rank == 0) {
            fprintf(stderr, "Usage: %s <graph_file.mtx>\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    char *filename = argv[1];


    if (world_rank == 0) {
        int Nplus1 = cooReader(filename, &global_rowptr, &global_colind);
        if (Nplus1 <= 0) {
            fprintf(stderr, "Rank 0: cooReader failed for %s\n", filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        N   = Nplus1 - 1;
        nnz = global_rowptr[N];

        printf("Rank 0: Read graph '%s' with N = %d, nnz = %d\n",
               filename, N, nnz);
        fflush(stdout);
    }


    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (N <= 0) {
        if (world_rank == 0) {
            fprintf(stderr, "Invalid N (%d). Exiting.\n", N);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    get_range(N, world_size, world_rank, &local_start, &local_n);

    if (world_rank == 0) {
        printf("World size = %d, N = %d\n", world_size, N);
        for (int r = 0; r < world_size; ++r) {
            int s, c;
            get_range(N, world_size, r, &s, &c);
            printf("  Rank %d: local_start = %d, local_n = %d\n", r, s, c);
        }
        fflush(stdout);
    }


    if (world_rank == 0) {
        for (int r = 0; r < world_size; ++r) {
            int r_start, r_n;
            get_range(N, world_size, r, &r_start, &r_n);

            int row_begin = global_rowptr[r_start];
            int row_end   = (r_n > 0) ? global_rowptr[r_start + r_n] : row_begin;
            int r_nnz     = row_end - row_begin;

            if (r == 0) {
            
                local_start = r_start;
                local_n     = r_n;
                local_nnz   = r_nnz;

                local_rowptr = (int *)malloc((local_n + 1) * sizeof(int));
                if (!local_rowptr) {
                    fprintf(stderr, "Rank 0: Failed to allocate local_rowptr\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            
                for (int i = 0; i <= local_n; ++i) {
                    local_rowptr[i] = global_rowptr[local_start + i] - row_begin;
                }

                local_colind = NULL;
                if (local_nnz > 0) {
                    local_colind = (int *)malloc(local_nnz * sizeof(int));
                    if (!local_colind) {
                        fprintf(stderr, "Rank 0: Failed to allocate local_colind\n");
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    for (int i = 0; i < local_nnz; ++i) {
                        local_colind[i] = global_colind[row_begin + i];
                    }
                }

            } else {
            
                MPI_Send(&r_n,     1, MPI_INT, r, 100, MPI_COMM_WORLD);
                MPI_Send(&r_start, 1, MPI_INT, r, 101, MPI_COMM_WORLD);
                MPI_Send(&r_nnz,   1, MPI_INT, r, 102, MPI_COMM_WORLD);

            
                int *tmp_rowptr = (int *)malloc((r_n + 1) * sizeof(int));
                if (!tmp_rowptr) {
                    fprintf(stderr, "Rank 0: Failed to allocate tmp_rowptr for rank %d\n", r);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                for (int i = 0; i <= r_n; ++i) {
                    tmp_rowptr[i] = global_rowptr[r_start + i] - row_begin;
                }
                MPI_Send(tmp_rowptr, r_n + 1, MPI_INT, r, 103, MPI_COMM_WORLD);
                free(tmp_rowptr);

            
                if (r_nnz > 0) {
                    MPI_Send(&global_colind[row_begin], r_nnz,
                             MPI_INT, r, 104, MPI_COMM_WORLD);
                }
            }
        }

    
        free(global_rowptr);
        free(global_colind);
        global_rowptr = NULL;
        global_colind = NULL;

    } else {
    
        MPI_Recv(&local_n,     1, MPI_INT, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&local_start, 1, MPI_INT, 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&local_nnz,   1, MPI_INT, 0, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        local_rowptr = (int *)malloc((local_n + 1) * sizeof(int));
        if (!local_rowptr) {
            fprintf(stderr, "Rank %d: Failed to allocate local_rowptr\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Recv(local_rowptr, local_n + 1, MPI_INT, 0, 103, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        local_colind = NULL;
        if (local_nnz > 0) {
            local_colind = (int *)malloc(local_nnz * sizeof(int));
            if (!local_colind) {
                fprintf(stderr, "Rank %d: Failed to allocate local_colind\n", world_rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            MPI_Recv(local_colind, local_nnz, MPI_INT, 0, 104, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        printf("CSR distributed. Starting CC computation.\n");
        fflush(stdout);
    }


    colors = (int *)malloc(N * sizeof(int));
    global_min_colors = (int *)malloc(N * sizeof(int));
    if (!colors || !global_min_colors) {
        fprintf(stderr, "Rank %d: Failed to allocate color arrays\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < N; ++i) {
        colors[i] = i;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        clock_gettime(CLOCK_REALTIME, &t_start);
    }


    int iteration     = 0;
    int active_local  = 1;
    int active_global = 1;

    while (active_global) {
        active_local = 0;


        for (int li = 0; li < local_n; ++li) {
            int i_global  = local_start + li;
            int row_begin = local_rowptr[li];
            int row_end   = local_rowptr[li + 1];

            for (int j = row_begin; j < row_end; ++j) {
                int neighbor = local_colind[j];

                int ci = colors[i_global];
                int cn = colors[neighbor];
                int cmin = (ci < cn) ? ci : cn;

                if (colors[i_global] != cmin) {
                    colors[i_global] = cmin;
                    active_local = 1;
                }
                if (colors[neighbor] != cmin) {
                    colors[neighbor] = cmin;
                    active_local = 1;
                }
            }
        }

    
        MPI_Allreduce(colors, global_min_colors, N, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    
        for (int i = 0; i < N; ++i) {
            if (colors[i] != global_min_colors[i]) {
                colors[i] = global_min_colors[i];
                active_local = 1;
            }
        }

    
        MPI_Allreduce(&active_local, &active_global, 1,
                      MPI_INT, MPI_LOR, MPI_COMM_WORLD);

        iteration++;
    }

    if (world_rank == 0) {
        clock_gettime(CLOCK_REALTIME, &t_end);
    }


    if (world_rank == 0) {
        int *uniqueFlags = (int *)calloc(N, sizeof(int));
        if (!uniqueFlags) {
            fprintf(stderr, "Rank 0: Failed to allocate uniqueFlags\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int numCC = 0;
        for (int i = 0; i < N; ++i) {
            int c = colors[i];
            if (!uniqueFlags[c]) {
                uniqueFlags[c] = 1;
                numCC++;
            }
        }

        double duration = ((t_end.tv_sec - t_start.tv_sec) * 1e6 +
                           (t_end.tv_nsec - t_start.tv_nsec) / 1000.0) / 1e6;

        printf("Iterations: %d\n", iteration);
        printf("Number of connected components: %d\n", numCC);
        printf("~ CC duration: %lf seconds\n", duration);
        fflush(stdout);

        FILE *f = fopen("results.csv", "a");
        if (f != NULL) {
            fprintf(f, "mpi,%d,%lf,%s,%d\n", world_size, duration, filename, N);
            fclose(f);
        }

        free(uniqueFlags);
    }


    free(colors);
    free(global_min_colors);
    free(local_rowptr);
    free(local_colind);

    MPI_Finalize();
    return 0;
}
