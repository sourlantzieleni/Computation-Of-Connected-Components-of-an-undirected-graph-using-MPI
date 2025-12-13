#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include "mmio.h"
#include "converter.h"

/* 1D block distribution of [0..N-1] among p ranks */
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

/* Determine which rank owns vertex gid */
static int owner_of(int gid, const int *rank_starts, const int *rank_counts, int p) {
    for (int r = 0; r < p; ++r) {
        int s = rank_starts[r];
        int e = s + rank_counts[r];
        if (gid >= s && gid < e) return r;
    }
    return -1;  /* should not happen */
}

/* Comparator for qsort on int */
static int cmp_int(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return (ia > ib) - (ia < ib);
}

/* Binary search for gid in gids[offset .. offset+count-1] */
static int find_gid_in_segment(int gid, const int *gids, int offset, int count) {
    int lo = offset;
    int hi = offset + count - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int v = gids[mid];
        if (v == gid) return mid;
        if (v < gid) lo = mid + 1;
        else         hi = mid - 1;
    }
    fprintf(stderr, "ERROR: gid %d not found in ghost segment\n", gid);
    return -1;
}

int main(int argc, char *argv[]) {
    int world_size, world_rank;
    int N = 0;      /* number of vertices */
    int nnz = 0;    /* number of edges (CSR col length) */

    /* Global CSR (rank 0 only) */
    int *global_rowptr = NULL;
    int *global_colind = NULL;

    /* Local CSR */
    int *local_rowptr  = NULL;
    int *local_colind  = NULL;

    int local_start = 0;
    int local_n     = 0;
    int local_nnz   = 0;

    /* Local colors (only owned vertices) */
    int *local_colors = NULL;

    /* Distribution metadata */
    int *rank_starts = NULL;
    int *rank_counts = NULL;

    /* Ghost / boundary communication pattern */
    int *remote_gids   = NULL;  /* distinct remote neighbor gids */
    int  total_remote  = 0;

    /* GIDs this rank needs from each other rank (ghosts) */
    int *need_counts   = NULL;
    int *need_displs   = NULL;
    int  total_need    = 0;
    int *need_gids     = NULL;  /* length total_need */

    /* GIDs this rank must provide to others */
    int *give_counts   = NULL;
    int *give_displs   = NULL;
    int  total_give    = 0;
    int *give_gids     = NULL;  /* gids I own that others requested */
    int *give_lidx     = NULL;  /* local indices for give_gids */

    /* Per-iteration color exchange buffers */
    int *give_colors   = NULL;  /* colors to send */
    int *need_colors   = NULL;  /* ghost colors received */

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

    /* ---- 1. Rank 0 reads graph and builds CSR via cooReader ---- */
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

    /* ---- 2. Broadcast N ---- */
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (N <= 0) {
        if (world_rank == 0) {
            fprintf(stderr, "Invalid N (%d). Exiting.\n", N);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ---- 3. Rank ranges ---- */
    rank_starts = (int *)malloc(world_size * sizeof(int));
    rank_counts = (int *)malloc(world_size * sizeof(int));
    if (!rank_starts || !rank_counts) {
        fprintf(stderr, "Rank %d: Failed to allocate rank_starts/counts\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int r = 0; r < world_size; ++r) {
        get_range(N, world_size, r, &rank_starts[r], &rank_counts[r]);
    }
    get_range(N, world_size, world_rank, &local_start, &local_n);

    if (world_rank == 0) {
        printf("World size = %d, N = %d\n", world_size, N);
        for (int r = 0; r < world_size; ++r) {
            printf("  Rank %d: local_start = %d, local_n = %d\n",
                   r, rank_starts[r], rank_counts[r]);
        }
        fflush(stdout);
    }

    /* ---- 4. Distribute CSR by rows from rank 0 ---- */
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

                if (local_nnz > 0) {
                    local_colind = (int *)malloc(local_nnz * sizeof(int));
                    if (!local_colind) {
                        fprintf(stderr, "Rank 0: Failed to allocate local_colind\n");
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    for (int i = 0; i < local_nnz; ++i) {
                        local_colind[i] = global_colind[row_begin + i];
                    }
                } else {
                    local_colind = NULL;
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

        if (local_nnz > 0) {
            local_colind = (int *)malloc(local_nnz * sizeof(int));
            if (!local_colind) {
                fprintf(stderr, "Rank %d: Failed to allocate local_colind\n", world_rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            MPI_Recv(local_colind, local_nnz, MPI_INT, 0, 104, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        } else {
            local_colind = NULL;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        printf("CSR distributed. Building communication pattern...\n");
        fflush(stdout);
    }

    /* ---- 5. Allocate and init local colors (only owned vertices) ---- */
    if (local_n < 1) local_n = 0;
    local_colors = (int *)malloc((local_n > 0 ? local_n : 1) * sizeof(int));
    if (!local_colors) {
        fprintf(stderr, "Rank %d: Failed to allocate local_colors\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int li = 0; li < local_n; ++li) {
        local_colors[li] = local_start + li;  /* initial label = global id */
    }

    /* ---- 6. Build list of distinct remote neighbor vertices ---- */
    int remote_cap = (local_nnz > 0 ? local_nnz : 1);
    remote_gids = (int *)malloc(remote_cap * sizeof(int));
    if (!remote_gids) {
        fprintf(stderr, "Rank %d: Failed to allocate remote_gids\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    total_remote = 0;

    for (int li = 0; li < local_n; ++li) {
        int row_begin = local_rowptr[li];
        int row_end   = local_rowptr[li + 1];
        for (int j = row_begin; j < row_end; ++j) {
            int nb = local_colind[j];  /* neighbor global id */
            if (nb < local_start || nb >= local_start + local_n) {
                /* remote neighbor */
                if (total_remote == remote_cap) {
                    remote_cap *= 2;
                    remote_gids = (int *)realloc(remote_gids, remote_cap * sizeof(int));
                    if (!remote_gids) {
                        fprintf(stderr, "Rank %d: Failed to realloc remote_gids\n", world_rank);
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                }
                remote_gids[total_remote++] = nb;
            }
        }
    }

    /* Remove duplicates, keep sorted */
    if (total_remote > 0) {
        qsort(remote_gids, total_remote, sizeof(int), cmp_int);
        int unique_count = 1;
        for (int i = 1; i < total_remote; ++i) {
            if (remote_gids[i] != remote_gids[unique_count - 1]) {
                remote_gids[unique_count++] = remote_gids[i];
            }
        }
        total_remote = unique_count;
    }

    /* ---- 7. Build need_counts / need_gids (ghosts I need) ---- */
    need_counts = (int *)calloc(world_size, sizeof(int));
    need_displs = (int *)calloc(world_size, sizeof(int));
    if (!need_counts || !need_displs) {
        fprintf(stderr, "Rank %d: Failed to allocate need_counts/displs\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* count how many gids needed from each rank */
    for (int i = 0; i < total_remote; ++i) {
        int gid   = remote_gids[i];
        int owner = owner_of(gid, rank_starts, rank_counts, world_size);
        if (owner < 0) {
            fprintf(stderr, "Rank %d: owner_of(%d) < 0\n", world_rank, gid);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        need_counts[owner]++;
    }

    total_need = 0;
    for (int r = 0; r < world_size; ++r) {
        need_displs[r] = total_need;
        total_need += need_counts[r];
    }

    if (total_need > 0) {
        need_gids = (int *)malloc(total_need * sizeof(int));
        if (!need_gids) {
            fprintf(stderr, "Rank %d: Failed to allocate need_gids\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        /* Fill by owner rank, preserving sorted order per rank */
        int *tmp_off = (int *)malloc(world_size * sizeof(int));
        for (int r = 0; r < world_size; ++r) tmp_off[r] = need_displs[r];
        for (int i = 0; i < total_remote; ++i) {
            int gid   = remote_gids[i];
            int owner = owner_of(gid, rank_starts, rank_counts, world_size);
            int pos   = tmp_off[owner]++;
            need_gids[pos] = gid;
        }
        free(tmp_off);
    } else {
        need_gids = NULL;
    }

    /* ---- 8. Compute give_counts / give_gids with Alltoall/Alltoallv ---- */
    give_counts = (int *)malloc(world_size * sizeof(int));
    give_displs = (int *)malloc(world_size * sizeof(int));
    if (!give_counts || !give_displs) {
        fprintf(stderr, "Rank %d: Failed to allocate give_counts/displs\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Alltoall(need_counts, 1, MPI_INT,
                 give_counts, 1, MPI_INT,
                 MPI_COMM_WORLD);

    total_give = 0;
    for (int r = 0; r < world_size; ++r) {
        give_displs[r] = total_give;
        total_give += give_counts[r];
    }

    if (total_give > 0) {
        give_gids = (int *)malloc(total_give * sizeof(int));
        give_lidx = (int *)malloc(total_give * sizeof(int));
        if (!give_gids || !give_lidx) {
            fprintf(stderr, "Rank %d: Failed to allocate give_gids/lidx\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else {
        give_gids = NULL;
        give_lidx = NULL;
    }

    /* Exchange GID request lists: need_gids -> give_gids */
    if (total_need > 0 || total_give > 0) {
        MPI_Alltoallv(need_gids,
                      need_counts, need_displs, MPI_INT,
                      give_gids,
                      give_counts, give_displs, MPI_INT,
                      MPI_COMM_WORLD);
    }

    /* Convert give_gids to local indices (for vertices I own) */
    int my_start = rank_starts[world_rank];
    int my_end   = my_start + rank_counts[world_rank];

    for (int p = 0; p < total_give; ++p) {
        int gid = give_gids[p];
        if (gid < my_start || gid >= my_end) {
            fprintf(stderr,
                    "Rank %d: give_gids[%d] = %d not in my range [%d,%d)\n",
                    world_rank, p, gid, my_start, my_end);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        give_lidx[p] = gid - my_start;
    }

    /* Allocate per-iteration color exchange buffers */
    if (total_give > 0) {
        give_colors = (int *)malloc(total_give * sizeof(int));
    } else {
        give_colors = (int *)malloc(sizeof(int));  /* dummy */
    }
    if (total_need > 0) {
        need_colors = (int *)malloc(total_need * sizeof(int));
    } else {
        need_colors = (int *)malloc(sizeof(int));  /* dummy */
    }
    if (!give_colors || !need_colors) {
        fprintf(stderr, "Rank %d: Failed to allocate color buffers\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        printf("Communication pattern built. Starting CC computation...\n");
        fflush(stdout);
        clock_gettime(CLOCK_REALTIME, &t_start);
    }

    /* ---- 9. Iterative label propagation with ghost exchange ---- */
    int iteration     = 0;
    int active_local  = 1;
    int active_global = 1;

    while (active_global) {
        /* 9.1 Fill colors to send (for vertices others requested) */
        for (int p = 0; p < total_give; ++p) {
            int li = give_lidx[p];
            give_colors[p] = local_colors[li];
        }

        /* Exchange colors: owners -> requestors */
        if (total_need > 0 || total_give > 0) {
            MPI_Alltoallv(give_colors,
                          give_counts, give_displs, MPI_INT,
                          need_colors,
                          need_counts, need_displs, MPI_INT,
                          MPI_COMM_WORLD);
        }

        /* 9.2 Relax local vertices */
        active_local = 0;

        for (int li = 0; li < local_n; ++li) {
            int row_begin = local_rowptr[li];
            int row_end   = local_rowptr[li + 1];

            int old_color = local_colors[li];
            int new_color = old_color;

            for (int j = row_begin; j < row_end; ++j) {
                int nb = local_colind[j];  /* neighbor global id */

                int cn;
                if (nb >= local_start && nb < local_start + local_n) {
                    /* local neighbor */
                    int li_nb = nb - local_start;
                    cn = local_colors[li_nb];
                } else {
                    /* remote neighbor: find its color in need_colors */
                    int owner = owner_of(nb, rank_starts, rank_counts, world_size);
                    if (owner < 0) {
                        fprintf(stderr, "Rank %d: owner_of(%d) < 0 in iteration\n",
                                world_rank, nb);
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    int off   = need_displs[owner];
                    int count = need_counts[owner];
                    int pos   = find_gid_in_segment(nb, need_gids, off, count);
                    cn = need_colors[pos];
                }

                if (cn < new_color) {
                    new_color = cn;
                }
            }

            if (new_color != old_color) {
                local_colors[li] = new_color;
                active_local = 1;
            }
        }

        /* 9.3 Check global convergence */
        MPI_Allreduce(&active_local, &active_global, 1,
                      MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        iteration++;
    }

    if (world_rank == 0) {
        clock_gettime(CLOCK_REALTIME, &t_end);
    }

    /* ---- 10. Gather colors to rank 0 and count CCs ---- */
    int *global_colors = NULL;
    int *gath_counts   = NULL;
    int *gath_displs   = NULL;

    if (world_rank == 0) {
        global_colors = (int *)malloc(N * sizeof(int));
        gath_counts   = (int *)malloc(world_size * sizeof(int));
        gath_displs   = (int *)malloc(world_size * sizeof(int));
        if (!global_colors || !gath_counts || !gath_displs) {
            fprintf(stderr, "Rank 0: Failed to allocate gather buffers\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int r = 0; r < world_size; ++r) {
            gath_counts[r] = rank_counts[r];
            gath_displs[r] = rank_starts[r];
        }
    }

    MPI_Gatherv(local_colors, local_n, MPI_INT,
                global_colors, gath_counts, gath_displs, MPI_INT,
                0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        int *seen = (int *)calloc(N, sizeof(int));
        if (!seen) {
            fprintf(stderr, "Rank 0: Failed to allocate seen[]\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int numCC = 0;
        for (int i = 0; i < N; ++i) {
            int c = global_colors[i];
            if (!seen[c]) {
                seen[c] = 1;
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
            fprintf(f, "mpi_ghost,%d,%lf,%s,%d\n",
                    world_size, duration, filename, N);
            fclose(f);
        }

        free(seen);
        free(global_colors);
        free(gath_counts);
        free(gath_displs);
    }

    /* ---- 11. Cleanup ---- */
    free(local_colors);
    free(local_rowptr);
    free(local_colind);

    free(rank_starts);
    free(rank_counts);

    free(remote_gids);

    free(need_counts);
    free(need_displs);
    free(need_gids);

    free(give_counts);
    free(give_displs);
    free(give_gids);
    free(give_lidx);

    free(give_colors);
    free(need_colors);

    MPI_Finalize();
    return 0;
}
