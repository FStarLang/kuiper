/*
 * SpMM Benchmark: Kuiper vs Sputnik
 *
 * Compares the performance of Kuiper's verified SpMM kernel (f32)
 * against Google's Sputnik SpMM library. Both use CSR sparse format
 * with float32 values.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>

/* Kuiper SpMM (f32) */
#include "Klas_SPMM.h"

/* Sputnik SpMM */
#include "sputnik/spmm/cuda_spmm.h"

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

/* ------------------------------------------------------------------ */
/* CSR matrix generation                                               */
/* ------------------------------------------------------------------ */

struct CSR {
    int rows, cols, nnz;
    std::vector<float>    values;
    std::vector<uint32_t> col_ind;   /* Kuiper uses uint32_t col indices */
    std::vector<uint32_t> row_off;   /* Kuiper uses uint32_t row offsets */
    std::vector<int>      col_ind_i; /* Sputnik uses int col indices */
    std::vector<int>      row_off_i; /* Sputnik uses int row offsets */
    std::vector<int>      row_indices; /* for sputnik load-balancing */
};

/*
 * Generate a random sparse matrix in CSR format.
 * density_pct is in [0, 100].
 */
static void gen_sparse(int rows, int cols, int density_pct, CSR &csr)
{
    csr.rows = rows;
    csr.cols = cols;
    csr.values.clear();
    csr.col_ind.clear();
    csr.row_off.resize(rows + 1);
    csr.col_ind_i.clear();
    csr.row_off_i.resize(rows + 1);

    csr.row_off[0] = 0;
    csr.row_off_i[0] = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (rand() % 100 < density_pct) {
                float v = 1.0f + (float)(rand() % 99);
                csr.values.push_back(v);
                csr.col_ind.push_back((uint32_t)j);
                csr.col_ind_i.push_back(j);
            }
        }
        csr.row_off[i + 1] = (uint32_t)csr.values.size();
        csr.row_off_i[i + 1] = (int)csr.values.size();
    }
    csr.nnz = (int)csr.values.size();

    /* row_indices for sputnik: sorted by nnz-per-row descending (load balance) */
    csr.row_indices.resize(rows);
    std::iota(csr.row_indices.begin(), csr.row_indices.end(), 0);
    std::sort(csr.row_indices.begin(), csr.row_indices.end(),
              [&](int a, int b) {
                  int nnz_a = csr.row_off[a + 1] - csr.row_off[a];
                  int nnz_b = csr.row_off[b + 1] - csr.row_off[b];
                  return nnz_a > nnz_b;
              });
}

/* ------------------------------------------------------------------ */
/* Timing helpers using CUDA events                                    */
/* ------------------------------------------------------------------ */

static float bench_kuiper(int rows, int shared, int cols,
                          uint32_t *d_row_indices,
                          Kuiper_Sparse_Matrix_smatrix__float dA,
                          float *dB, float *dC,
                          int warmup, int iters)
{
    for (int i = 0; i < warmup; i++) {
        Klas_SPMM_spmm_f32(rows, shared, cols, dA, d_row_indices, dB, dC);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        Klas_SPMM_spmm_f32(rows, shared, cols, dA, d_row_indices, dB, dC);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / iters;
}

static float bench_sputnik(int m, int k, int n, int nnz,
                           int *d_row_indices, float *d_values,
                           int *d_row_offsets, int *d_col_indices,
                           float *d_dense, float *d_out,
                           int warmup, int iters)
{
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    for (int i = 0; i < warmup; i++) {
        sputnik::CudaSpmm(m, k, n, nnz, d_row_indices, d_values,
                           d_row_offsets, d_col_indices, d_dense, d_out, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; i++) {
        sputnik::CudaSpmm(m, k, n, nnz, d_row_indices, d_values,
                           d_row_offsets, d_col_indices, d_dense, d_out, stream);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return ms / iters;
}

/* ------------------------------------------------------------------ */
/* GPU memory helpers                                                  */
/* ------------------------------------------------------------------ */

template <typename T>
static T *to_gpu(const std::vector<T> &v)
{
    T *d;
    size_t bytes = v.size() * sizeof(T);
    if (bytes == 0) bytes = sizeof(T); /* avoid zero-size alloc */
    CHECK_CUDA(cudaMalloc(&d, bytes));
    CHECK_CUDA(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

template <typename T>
static T *gpu_zeros(size_t n)
{
    T *d;
    size_t bytes = n * sizeof(T);
    if (bytes == 0) bytes = sizeof(T);
    CHECK_CUDA(cudaMalloc(&d, bytes));
    CHECK_CUDA(cudaMemset(d, 0, bytes));
    return d;
}

/* ------------------------------------------------------------------ */
/* Run one benchmark configuration                                     */
/* ------------------------------------------------------------------ */

static void run_bench(int rows, int shared, int cols, int density_pct,
                      int warmup, int iters)
{
    /* Kuiper requires cols to be a multiple of blockItemsX=128 */
    if (cols % 128 != 0) {
        fprintf(stderr, "SKIP %dx%dx%d @%d%%: cols must be multiple of 128 for Kuiper\n",
                rows, shared, cols, density_pct);
        return;
    }

    CSR csr;
    gen_sparse(rows, shared, density_pct, csr);

    /* Dense B matrix (random float) */
    std::vector<float> B(shared * cols);
    for (int i = 0; i < shared * cols; i++)
        B[i] = 1.0f + (float)(rand() % 99);

    /* Upload Kuiper data (float values, uint32_t indices) */
    Kuiper_Sparse_Matrix_smatrix__float dA_k;
    dA_k.nnz     = (uint32_t)csr.nnz;
    dA_k.elems   = to_gpu(csr.values);
    dA_k.col_ind = to_gpu(csr.col_ind);
    dA_k.row_off = to_gpu(csr.row_off);
    float *dB_k = to_gpu(B);
    float *dC_k = gpu_zeros<float>(rows * cols);

    /* Upload Sputnik data (int indices) */
    int   *d_row_indices = to_gpu(csr.row_indices);
    float *d_values      = to_gpu(csr.values);
    int   *d_row_offsets = to_gpu(csr.row_off_i);
    int   *d_col_indices = to_gpu(csr.col_ind_i);
    float *d_dense       = to_gpu(B);
    float *d_out         = gpu_zeros<float>(rows * cols);

    /* Correctness check: run both once and compare outputs */
    CHECK_CUDA(cudaMemset(dC_k, 0, sizeof(float) * rows * cols));
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float) * rows * cols));

    Klas_SPMM_spmm_f32(rows, shared, cols, dA_k, (uint32_t*)d_row_indices, dB_k, dC_k);
    CHECK_CUDA(cudaDeviceSynchronize());

    {
        cudaStream_t s;
        CHECK_CUDA(cudaStreamCreate(&s));
        sputnik::CudaSpmm(rows, shared, cols, csr.nnz,
                           d_row_indices, d_values, d_row_offsets,
                           d_col_indices, d_dense, d_out, s);
        CHECK_CUDA(cudaStreamSynchronize(s));
        CHECK_CUDA(cudaStreamDestroy(s));
    }

    std::vector<float> C_kuiper(rows * cols), C_sputnik(rows * cols);
    CHECK_CUDA(cudaMemcpy(C_kuiper.data(), dC_k, sizeof(float) * rows * cols,
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C_sputnik.data(), d_out, sizeof(float) * rows * cols,
                          cudaMemcpyDeviceToHost));

    int mismatches = 0;
    float max_reldiff = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float vk = C_kuiper[i * cols + j];
            float vs = C_sputnik[i * cols + j];
            float diff = fabsf(vk - vs);
            float denom = fmaxf(fabsf(vk), fabsf(vs));
            float rel = (denom > 0) ? diff / denom : diff;
            if (rel > max_reldiff) max_reldiff = rel;
            /* tolerance: fp32 accumulation order differs, allow small error */
            if (rel > 1e-4f && diff > 1e-2f) {
                if (mismatches == 0)
                    fprintf(stderr,
                        "  MISMATCH at (%d,%d): kuiper=%.6f sputnik=%.6f "
                        "(diff=%.6f rel=%.6f)\n", i, j, vk, vs, diff, rel);
                mismatches++;
            }
        }
    }

    const char *status = (mismatches == 0) ? "OK" : "FAIL";

    /* Effective FLOPs: 2 * nnz * cols (one mul + one add per nonzero per output col) */
    double flops = 2.0 * csr.nnz * cols;

    float ms_kuiper  = bench_kuiper(rows, shared, cols, (uint32_t*)d_row_indices, dA_k, dB_k, dC_k, warmup, iters);
    float ms_sputnik = bench_sputnik(rows, shared, cols, csr.nnz,
                                     d_row_indices, d_values, d_row_offsets,
                                     d_col_indices, d_dense, d_out,
                                     warmup, iters);

    double gflops_kuiper  = flops / (ms_kuiper  * 1e6);
    double gflops_sputnik = flops / (ms_sputnik * 1e6);
    double speedup = ms_kuiper / ms_sputnik; /* >1 means sputnik is faster */

    printf("%-6d %-6d %-6d %3d%%  %-8d  "
           "%8.3f ms (%6.1f GFLOP/s)  "
           "%8.3f ms (%6.1f GFLOP/s)  "
           "%.2fx  %s (maxrel=%.1e)\n",
           rows, shared, cols, density_pct, csr.nnz,
           ms_kuiper, gflops_kuiper,
           ms_sputnik, gflops_sputnik,
           speedup, status, max_reldiff);

    /* Cleanup */
    cudaFree(dA_k.elems); cudaFree(dA_k.col_ind); cudaFree(dA_k.row_off);
    cudaFree(dB_k); cudaFree(dC_k);
    cudaFree(d_row_indices); cudaFree(d_values);
    cudaFree(d_row_offsets); cudaFree(d_col_indices);
    cudaFree(d_dense); cudaFree(d_out);
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

static void print_gpu_info()
{
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, dev);
    printf("GPU: %s (SM %d.%d, %d SMs, %.0f MHz)\n\n",
           props.name, props.major, props.minor,
           props.multiProcessorCount,
           props.clockRate / 1000.0);
}

int main(int argc, char **argv)
{
    int warmup = 5;
    int iters  = 20;

    /* Parse optional --warmup and --iters */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
            iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--warmup N] [--iters N]\n", argv[0]);
            return 0;
        }
    }

    srand(42);
    print_gpu_info();

    printf("%-6s %-6s %-6s %-5s %-8s  %-31s  %-31s  %-5s  %s\n",
           "rows", "K", "cols", "dens", "nnz",
           "Kuiper (float32)", "Sputnik (float32)", "K/S", "check");
    printf("%s\n", std::string(140, '-').c_str());

    /* Square matrices at various sizes and densities */
    int sizes[]     = { 256, 512, 1024, 2048 };
    int densities[] = { 1, 5, 10, 30, 50 };

    for (int s : sizes)
        for (int d : densities)
            run_bench(s, s, s < 128 ? 128 : s, d, warmup, iters);

    printf("\n--- Non-square matrices ---\n");
    run_bench(512,  1024, 256,  10, warmup, iters);
    run_bench(1024, 512,  256,  10, warmup, iters);
    run_bench(2048, 256,  128,  10, warmup, iters);
    run_bench(256,  256,  1024, 10, warmup, iters);
    run_bench(1024, 1024, 128,  5,  warmup, iters);

    return 0;
}
