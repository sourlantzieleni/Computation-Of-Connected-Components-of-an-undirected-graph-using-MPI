#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <mpi.h>
#include "mmio.h"
#include "converter.h"
#include <limits.h>

/* ----------------- Utility: partition vertices ----------------- */

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
static int find_gid_in_segment(int gid, const int *gids, int offset, int count, int rank) {
    int lo = offset;
    int hi = offset + count - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int v = gids[mid];
        if (v == gid) return mid;
        if (v < gid) lo = mid + 1;
        else         hi = mid - 1;
    }
    fprintf(stderr, "Rank %d: ERROR: gid %d not found in ghost segment.\n", rank, gid);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return -1;
}

/* ----------------- MAIN ----------------- */

int main(int argc, char *argv[]) {
    int world_size, world_rank;
    int N = 0;              /* number of vertices */
    long long nnz = 0;      /* number of edges in CSR (64-bit counts) */

    /* Global CSR (on rank 0 only) */
    long long *global_rowptr = NULL; /* rowptr entries are now 64-bit offsets */
    int       *global_colind = NULL; /* colind stores vertex IDs (32-bit) */

    /* Local CSR */
    long long *local_rowptr  = NULL;
    int       *local_colind  = NULL;

    int local_start = 0;
    int local_n     = 0;
    long long local_nnz   = 0; /* local nnz (may be large) */

    /* Local colors (only owned vertices) */
    int *local_colors = NULL;

    /* Distribution metadata */
    int *rank_starts = NULL;
    int *rank_counts = NULL;

    /* Ghost / boundary communication pattern */
    int *remote_gids   = NULL;  /* distinct remote neighbor gids */
    int  total_remote  = 0;

    int *need_counts   = NULL;
    int *need_displs   = NULL;
    int  total_need    = 0;
    int *need_gids     = NULL;

    int *give_counts   = NULL;
    int *give_displs   = NULL;
    int  total_give    = 0;
    int *give_gids     = NULL;
    int *give_lidx     = NULL;

    int *give_colors   = NULL;
    int *need_colors   = NULL;

    /* Precomputed mapping per edge */
    unsigned char *nbr_is_local = NULL;  /* 1 = local neighbor, 0 = ghost */
    int           *nbr_index    = NULL;  /* local index or index in need_colors */

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

    /* -------- 1. Rank 0 reads graph via cooReader (directed CSR) -------- */
    if (world_rank == 0) {
        int Nplus1 = cooReader(filename, (int**)&global_rowptr, &global_colind);
        if (Nplus1 <= 0) {
            fprintf(stderr, "Rank 0: cooReader failed for %s\n", filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        N   = Nplus1 - 1;
        /* cooReader returned an int* rowptr; we casted it above. Convert to 64-bit values */
        /* allocate 64-bit global_rowptr and copy */
        long long *tmp_rowptr = (long long *)malloc((size_t)(N + 1) * sizeof(long long));
        if (!tmp_rowptr) {
            fprintf(stderr, "Rank 0: Failed to allocate tmp 64-bit global_rowptr\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i <= N; ++i) {
            tmp_rowptr[i] = ((int *)global_rowptr)[i];
        }
        /* free the original int* memory returned by cooReader (it was stored in global_rowptr) */
        free(global_rowptr);
        global_rowptr = tmp_rowptr;

        nnz = global_rowptr[N];

        printf("Rank 0: Read graph '%s' with N = %d, nnz = %lld (directed)\n",
               filename, N, (long long)nnz);
        fflush(stdout);

        /* -------- 1b. Symmetrise: build undirected CSR --------
           Implemented with 64-bit counts for NNZ and rowptr offsets; vertex IDs remain 32-bit.
        */

        /* 1) count degrees after symmetrisation (may include duplicates initially)
           use 64-bit per-vertex degree in case a single vertex has >2^31 neighbors */
        long long *deg = (long long *)calloc((size_t)N, sizeof(long long));
        if (!deg) {
            fprintf(stderr, "Rank 0: Failed to allocate deg array\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int u = 0; u < N; ++u) {
            long long row_begin = global_rowptr[u];
            long long row_end   = global_rowptr[u + 1];
            for (long long idx = row_begin; idx < row_end; ++idx) {
                int v = global_colind[(size_t)idx];
                /* (u,v) */
                deg[u]++;
                /* (v,u) if not self-loop */
                if (u != v) deg[v]++;
            }
        }

        /* compute total symmetric nnz and check allocation fit */
        unsigned long long sym_nnz_ull = 0ULL;
        for (int i = 0; i < N; ++i) sym_nnz_ull += (unsigned long long)deg[i];

        if (sym_nnz_ull > (unsigned long long)SIZE_MAX) {
            fprintf(stderr, "Rank 0: symmetric nnz too large to allocate (%llu entries)\n", sym_nnz_ull);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        long long sym_nnz = (long long)sym_nnz_ull;

        /* 2) allocate symmetric CSR arrays (temporary)
           sym_rowptr is 64-bit offsets; sym_colind stores vertex IDs (int) */
        long long *sym_rowptr = (long long *)malloc((size_t)(N + 1) * sizeof(long long));
        int       *sym_colind = (int *)malloc((size_t)sym_nnz * sizeof(int));
        if (!sym_rowptr || !sym_colind) {
            fprintf(stderr, "Rank 0: Failed to allocate symmetric CSR arrays\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* build rowptr by prefix-sum of deg */
        sym_rowptr[0] = 0;
        for (int i = 0; i < N; ++i) {
            sym_rowptr[i + 1] = sym_rowptr[i] + deg[i];
        }

        /* 3) fill sym_colind using per-row cursor */
        long long *cursor = (long long *)malloc((size_t)N * sizeof(long long));
        if (!cursor) {
            fprintf(stderr, "Rank 0: Failed to allocate cursor array\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < N; ++i) cursor[i] = sym_rowptr[i];

        for (int u = 0; u < N; ++u) {
            long long row_begin = global_rowptr[u];
            long long row_end   = global_rowptr[u + 1];
            for (long long idx = row_begin; idx < row_end; ++idx) {
                int v = global_colind[(size_t)idx];
                /* (u,v) */
                sym_colind[(size_t)cursor[u]++] = v;
                /* (v,u) if not self-loop */
                if (u != v) {
                    sym_colind[(size_t)cursor[v]++] = u;
                }
            }
        }

        /* 4) sort and deduplicate each row, building final CSR (no duplicates) */
        long long *uniq_counts = (long long *)malloc((size_t)N * sizeof(long long));
        if (!uniq_counts) {
            fprintf(stderr, "Rank 0: Failed to allocate uniq_counts\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* sort each row and count unique entries */
        for (int u = 0; u < N; ++u) {
            long long a = sym_rowptr[u];
            long long b = sym_rowptr[u + 1];
            long long len = b - a;
            if (len > 1) {
                /* qsort expects size_t count and element size; cast len to size_t */
                qsort(&sym_colind[(size_t)a], (size_t)len, sizeof(int), cmp_int);
            }
            long long uniq = 0;
            int prev = -1;
            for (long long k = a; k < b; ++k) {
                int val = sym_colind[(size_t)k];
                if (uniq == 0 || val != prev) {
                    uniq++;
                    prev = val;
                }
            }
            uniq_counts[u] = uniq;
        }

        /* compute new rowptr (64-bit) */
        long long *new_rowptr = (long long *)malloc((size_t)(N + 1) * sizeof(long long));
        if (!new_rowptr) {
            fprintf(stderr, "Rank 0: Failed to allocate new_rowptr\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        new_rowptr[0] = 0;
        unsigned long long new_nnz_ull = 0ULL;
        for (int i = 0; i < N; ++i) {
            new_nnz_ull += (unsigned long long)uniq_counts[i];
            new_rowptr[i + 1] = new_rowptr[i] + uniq_counts[i];
        }

        if (new_nnz_ull > (unsigned long long)SIZE_MAX) {
            fprintf(stderr, "Rank 0: deduplicated symmetric nnz too large to allocate (%llu)\n", new_nnz_ull);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        long long new_nnz = (long long)new_nnz_ull;

        int *new_colind = (int *)malloc((size_t)new_nnz * sizeof(int));
        if (!new_colind) {
            fprintf(stderr, "Rank 0: Failed to allocate new_colind\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* copy unique entries into new_colind */
        for (int u = 0; u < N; ++u) {
            long long a = sym_rowptr[u];
            long long b = sym_rowptr[u + 1];
            long long dst = new_rowptr[u];
            int prev = -1;
            for (long long k = a; k < b; ++k) {
                int val = sym_colind[(size_t)k];
                if (dst == new_rowptr[u] || val != prev) {
                    new_colind[(size_t)dst++] = val;
                    prev = val;
                }
            }
            /* sanity: dst should equal new_rowptr[u+1] */
            if (dst != new_rowptr[u + 1]) {
                fprintf(stderr, "Rank 0: copy-unique mismatch for row %d (dst=%lld expected=%lld)\n",
                        u, (long long)dst, (long long)new_rowptr[u + 1]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        /* free temporaries and replace global CSR with symmetric CSR */
        free(deg);
        free(cursor);
        free(sym_colind);
        free(sym_rowptr);
        free(uniq_counts);

        global_rowptr = new_rowptr;
        global_colind = new_colind;
        nnz = new_nnz;

        printf("Rank 0: Symmetrised graph (CSR, 64-bit offsets) built: N = %d, sym_nnz = %lld\n",
               N, (long long)nnz);
        fflush(stdout);
    }

    /* -------- 2. Broadcast N -------- */
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (N <= 0) {
        if (world_rank == 0) {
            fprintf(stderr, "Invalid N (%d). Exiting.\n", N);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* -------- 3. Rank ranges -------- */
    rank_starts = (int *)malloc(world_size * sizeof(int));
    rank_counts = (int *)malloc(world_size * sizeof(int));
    if (!rank_starts || !rank_counts) {
        fprintf(stderr, "Rank %d: Failed to allocate rank_starts/rank_counts\n", world_rank);
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

    /* -------- 4. Distribute symmetric CSR by rows from rank 0 -------- */
    if (world_rank == 0) {
        for (int r = 0; r < world_size; ++r) {
            int r_start, r_n;
            get_range(N, world_size, r, &r_start, &r_n);

            long long row_begin = global_rowptr[r_start];
            long long row_end   = (r_n > 0) ? global_rowptr[r_start + r_n] : row_begin;
            long long r_nnz     = row_end - row_begin;

            if (r == 0) {
                local_start = r_start;
                local_n     = r_n;
                local_nnz   = r_nnz;

                local_rowptr = (long long *)malloc((size_t)(local_n + 1) * sizeof(long long));
                if (!local_rowptr) {
                    fprintf(stderr, "Rank 0: Failed to allocate local_rowptr\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                for (int i = 0; i <= local_n; ++i) {
                    local_rowptr[i] = global_rowptr[local_start + i] - row_begin;
                }

                if (local_nnz > 0) {
                    local_colind = (int *)malloc((size_t)local_nnz * sizeof(int));
                    if (!local_colind) {
                        fprintf(stderr, "Rank 0: Failed to allocate local_colind\n");
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    for (long long i = 0; i < local_nnz; ++i) {
                        local_colind[(size_t)i] = global_colind[(size_t)(row_begin + i)];
                    }
                } else {
                    local_colind = NULL;
                }

            } else {
                /* send r_n, r_start as ints, send r_nnz as long long? we only send counts as ints,
                   but we must send the rowptr array (64-bit offsets) so use MPI_LONG_LONG */
                MPI_Send(&r_n,     1, MPI_INT, r, 100, MPI_COMM_WORLD);
                MPI_Send(&r_start, 1, MPI_INT, r, 101, MPI_COMM_WORLD);
                /* r_nnz may be > INT_MAX; send as long long to be safe */
                long long r_nnz_ll = r_nnz;
                MPI_Send(&r_nnz_ll, 1, MPI_LONG_LONG, r, 102, MPI_COMM_WORLD);

                long long *tmp_rowptr = (long long *)malloc((size_t)(r_n + 1) * sizeof(long long));
                if (!tmp_rowptr) {
                    fprintf(stderr, "Rank 0: Failed to allocate tmp_rowptr for rank %d\n", r);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                for (int i = 0; i <= r_n; ++i) {
                    tmp_rowptr[i] = global_rowptr[r_start + i] - row_begin;
                }
                MPI_Send(tmp_rowptr, r_n + 1, MPI_LONG_LONG, r, 103, MPI_COMM_WORLD);
                free(tmp_rowptr);

                if (r_nnz > 0) {
                    /* send the colind slice as ints */
                    MPI_Send(&global_colind[(size_t)row_begin], (int)r_nnz,
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
        long long tmp_local_nnz = 0;
        MPI_Recv(&tmp_local_nnz, 1, MPI_LONG_LONG, 0, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_nnz = tmp_local_nnz;

        local_rowptr = (long long *)malloc((size_t)(local_n + 1) * sizeof(long long));
        if (!local_rowptr) {
            fprintf(stderr, "Rank %d: Failed to allocate local_rowptr\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Recv(local_rowptr, local_n + 1, MPI_LONG_LONG, 0, 103, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        if (local_nnz > 0) {
            if (local_nnz > INT_MAX) {
                fprintf(stderr, "Rank %d: local_nnz (%lld) exceeds INT_MAX; chunked receive needed\n",
                        world_rank, (long long)local_nnz);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            local_colind = (int *)malloc((size_t)local_nnz * sizeof(int));
            if (!local_colind) {
                fprintf(stderr, "Rank %d: Failed to allocate local_colind\n", world_rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            MPI_Recv(local_colind, (int)local_nnz, MPI_INT, 0, 104, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        } else {
            local_colind = NULL;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        printf("CSR (symmetrised) distributed. Building communication pattern...\n");
        fflush(stdout);
    }

    /* -------- 5. Allocate & init local colors -------- */
    if (local_n < 0) local_n = 0;
    local_colors = (int *)malloc((local_n > 0 ? local_n : 1) * sizeof(int));
    if (!local_colors) {
        fprintf(stderr, "Rank %d: Failed to allocate local_colors\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int li = 0; li < local_n; ++li) {
        local_colors[li] = local_start + li;  /* initial label = global id */
    }

    long long local_end = (long long)local_start + (long long)local_n;

    /* -------- 6. Build list of distinct remote neighbor vertices -------- */
    long long remote_cap = (local_nnz > 0 ? local_nnz : 1);
    remote_gids = (int *)malloc((size_t)remote_cap * sizeof(int));
    if (!remote_gids) {
        fprintf(stderr, "Rank %d: Failed to allocate remote_gids\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    total_remote = 0;

    for (int li = 0; li < local_n; ++li) {
        long long row_begin = local_rowptr[li];
        long long row_end   = local_rowptr[li + 1];
        for (long long j = row_begin; j < row_end; ++j) {
            int nb = local_colind[(size_t)j];  /* neighbor global id */
            if (nb < local_start || nb >= local_end) {
                if (total_remote == remote_cap) {
                    remote_cap *= 2;
                    remote_gids = (int *)realloc(remote_gids, (size_t)remote_cap * sizeof(int));
                    if (!remote_gids) {
                        fprintf(stderr, "Rank %d: Failed to realloc remote_gids\n", world_rank);
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                }
                remote_gids[total_remote++] = nb;
            }
        }
    }

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

    /* -------- 7. Build need_counts / need_gids -------- */
    need_counts = (int *)calloc(world_size, sizeof(int));
    need_displs = (int *)calloc(world_size, sizeof(int));
    if (!need_counts || !need_displs) {
        fprintf(stderr, "Rank %d: Failed to allocate need_counts/displs\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

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
        need_gids = (int *)malloc((size_t)total_need * sizeof(int));
        if (!need_gids) {
            fprintf(stderr, "Rank %d: Failed to allocate need_gids\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int *tmp_off = (int *)malloc(world_size * sizeof(int));
        if (!tmp_off) {
            fprintf(stderr, "Rank %d: Failed to allocate tmp_off\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
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

    /* -------- 8. Build give_counts / give_gids with Alltoallv -------- */
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
        give_gids = (int *)malloc((size_t)total_give * sizeof(int));
        give_lidx = (int *)malloc((size_t)total_give * sizeof(int));
        if (!give_gids || !give_lidx) {
            fprintf(stderr, "Rank %d: Failed to allocate give_gids/lidx\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else {
        give_gids = NULL;
        give_lidx = NULL;
    }

    if (total_need > 0 || total_give > 0) {
        MPI_Alltoallv(need_gids,
                      need_counts, need_displs, MPI_INT,
                      give_gids,
                      give_counts, give_displs, MPI_INT,
                      MPI_COMM_WORLD);
    }

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

    if (total_give > 0) {
        give_colors = (int *)malloc((size_t)total_give * sizeof(int));
    } else {
        give_colors = (int *)malloc(sizeof(int));
    }
    if (total_need > 0) {
        need_colors = (int *)malloc((size_t)total_need * sizeof(int));
    } else {
        need_colors = (int *)malloc(sizeof(int));
    }
    if (!give_colors || !need_colors) {
        fprintf(stderr, "Rank %d: Failed to allocate color buffers\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    /* -------- Precompute neighbor mapping -------- */
    nbr_is_local = (unsigned char *)malloc((size_t)(local_nnz > 0 ? local_nnz : 1) * sizeof(unsigned char));
    nbr_index    = (int *)malloc((size_t)(local_nnz > 0 ? local_nnz : 1) * sizeof(int));
    if (!nbr_is_local || !nbr_index) {
        fprintf(stderr, "Rank %d: Failed to allocate neighbor mapping arrays\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (total_need > 0) {
        for (int li = 0; li < local_n; ++li) {
            long long row_begin = local_rowptr[li];
            long long row_end   = local_rowptr[li + 1];
            for (long long j = row_begin; j < row_end; ++j) {
                int nb = local_colind[(size_t)j];
                if (nb >= local_start && nb < local_end) {
                    nbr_is_local[(size_t)j] = 1;
                    nbr_index[(size_t)j]    = nb - local_start;
                } else {
                    nbr_is_local[(size_t)j] = 0;
                    int owner = owner_of(nb, rank_starts, rank_counts, world_size);
                    if (owner < 0) {
                        fprintf(stderr, "Rank %d: owner_of(%d) < 0 in precompute\n",
                                world_rank, nb);
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    int off   = need_displs[owner];
                    int count = need_counts[owner];
                    int pos   = find_gid_in_segment(nb, need_gids, off, count, world_rank);
                    nbr_index[(size_t)j] = pos;
                }
            }
        }
    } else {
        /* No ghosts: all neighbors are local */
        for (int li = 0; li < local_n; ++li) {
            long long row_begin = local_rowptr[li];
            long long row_end   = local_rowptr[li + 1];
            for (long long j = row_begin; j < row_end; ++j) {
                int nb = local_colind[(size_t)j];
                nbr_is_local[(size_t)j] = 1;
                nbr_index[(size_t)j]    = nb - local_start;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        printf("Communication pattern and neighbor mapping built. Starting CC computation...\n");
        fflush(stdout);
        clock_gettime(CLOCK_REALTIME, &t_start);
    }

    /* -------- 9. Iterative label propagation with ghost exchange -------- */
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
            long long row_begin = local_rowptr[li];
            long long row_end   = local_rowptr[li + 1];

            int old_color = local_colors[li];
            int new_color = old_color;

            for (long long j = row_begin; j < row_end; ++j) {
                int cn;
                if (nbr_is_local[(size_t)j]) {
                    cn = local_colors[nbr_index[(size_t)j]];
                } else {
                    cn = need_colors[nbr_index[(size_t)j]];
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

    /* -------- 10. Gather final colors to rank 0 and count CCs -------- */
    int *global_colors = NULL;
    int *gath_counts   = NULL;
    int *gath_displs   = NULL;

    if (world_rank == 0) {
        global_colors = (int *)malloc((size_t)N * sizeof(int));
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
            fprintf(f, "mpi_ghost_sym_opt,%d,%lf,%s,%d\n",
                    world_size, duration, filename, N);
            fclose(f);
        }

        free(seen);
        free(global_colors);
        free(gath_counts);
        free(gath_displs);
    }

    /* -------- 11. Cleanup -------- */
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

    free(nbr_is_local);
    free(nbr_index);

    MPI_Finalize();
    return 0;
}
