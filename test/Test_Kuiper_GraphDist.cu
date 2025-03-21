#include "timing.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include "Kuiper_GraphDist.h"
#include "test-common.h"

const char *progname;

#define et uint16_t
#define DEFAULT_DIM   1026

void print_matrix(et *m, size_t size)
{
    printf("\n\n");
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            printf("%d ", m[i * size + j]);
        }
        printf("\n");
    }
}

void print_cuda_matrix(et *m, size_t size)
{
    et *m_cpu = (et *) malloc(size * size * sizeof(et));
    cudaMemcpy(m_cpu, m, size * size * sizeof(et), cudaMemcpyDeviceToHost);
    print_matrix(m_cpu, size);
    free(m_cpu);
}

int main(int argc, char **argv)
{
    int i, j;
    size_t size = DEFAULT_DIM;
    bool check = true;
    progname = argv[0];

    if (argc == 3) {
        size = atoi(argv[1]);
        check = atoi(argv[2]) != 0;
    } else if (argc == 1) {
        /* use defaults */
    } else {
        fprintf(stderr, "Usage: %s [<size> <check>]\n", argv[0]);
        return 1;
    }

    printf("Size = %lu\n", size);
    printf("Check = %d\n", check);

    et *m1_cpu;
    m1_cpu = (et *) malloc(size * size * sizeof(et));

    /* This constructs a path graph: 0 -> 1 -> 2 -> ... -> size-1 */
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            m1_cpu[i * size + j] = j == i + 1 ? 1 : 0;
        }
    }

    et *m1 = (et *) kpr_wait_alloc(sizeof(et), size * size);
    et *m2 = (et *) kpr_wait_alloc(sizeof(et), size * size);

    cudaMemcpy(m1, m1_cpu, size * size * sizeof(et), cudaMemcpyHostToDevice);

    // m2 is swap space
    cudaMemcpy(m2, m1_cpu, size * size * sizeof(et), cudaMemcpyHostToDevice);
    for (int pathlen = 1; pathlen < size; pathlen *= 2) {
        // print_cuda_matrix(m1, size);
        TIME_void(Kuiper_GraphDist_matmul_dist_gpu(size, m1, m2), NULL);
        et *tmp = m1;
        m1 = m2;
        m2 = tmp;
        cudaDeviceSynchronize();
    }
    cudaFree(m2);

    if (check) {
        cudaMemcpy(m1_cpu, m1, size * size * sizeof(et),
                   cudaMemcpyDeviceToHost);

        for (i = 0; i < size; i++) {
            for (j = 0; j < size; j++) {
                if (m1_cpu[i * size + j] != (j > i ? j - i : 0)) {
                    fprintf(stderr, "Error: m1[%d][%d] = %d\n", i, j,
                            m1_cpu[i * size + j]);
                    return 1;
                }
            }
        }
    }

    cudaFree(m1);
    free(m1_cpu);

    printf("OK\n");

    return 0;
}
