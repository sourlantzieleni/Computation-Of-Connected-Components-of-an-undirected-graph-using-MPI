#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include "mmio.h"
#include "converter.h"


struct timespec t_start, t_end;

int main(int argc, char *argv[])
{
    char *str = argv[1];
    int  *indexes;
    int  *indices;

    int N = cooReader(str,  &indexes, & indices)-1;

    int *colors=(int*)malloc(N*sizeof(int));

    int active = 1;

    for(int i=0; i<N; i++){
        colors[i]=i;
    }

    int thread_num =16;
    omp_set_num_threads(thread_num);

    clock_gettime(CLOCK_REALTIME, &t_start);

    while (active) {
        active = 0;
        
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            for (int j = indexes[i]; j < indexes[i+1]; ++j) {
                
                int cmin;
                if(colors[i]<colors[indices[j]]){
                    cmin = colors[i];
                }
                else{
                    cmin = colors[indices[j]];
                }


                if(colors[i] != cmin){
                    colors[i] = cmin;
                    active = 1;
                }
                if(colors[indices[j]] != cmin){
                    colors[indices[j]] = cmin;
                    active = 1;
                }
            }
        }

    }

    clock_gettime(CLOCK_REALTIME, &t_end);


    int numCC = 0;
    int *uniqueFlags = (int*)calloc(N, sizeof(int));

    for (int i = 0; i < N; i++) {
        int color = colors[i];
        if (!uniqueFlags[color]) {
            uniqueFlags[color] = 1;
            numCC++;
        }
    }

    printf("Number of connected components: %d\n", numCC);

    double duration = ((t_end.tv_sec - t_start.tv_sec) * 1000000 + (t_end.tv_nsec - t_start.tv_nsec) / 1000) / 1000000.0;
    printf("~ CC duration: %lf seconds\n", duration);

    FILE *f = fopen("results.csv", "a");
    fprintf(f, "openmp,%d,%lf,%s,%d\n", thread_num, duration, str, N);
    fclose(f);


    free(colors);
    free(uniqueFlags);

    return 0; 
}

