/*
 * SpMM Benchmark: Kuiper vs Sputnik
 *
 * Compares the performance of Kuiper's verified SpMM kernel (f32)
 * against Google's Sputnik SpMM library. Both use CSR sparse format
 * with float32 values.
 *
 * Neither side goes through a dispatcher: both run one fixed tile
 * configuration, chosen so the two kernels are parameterized identically.
 * The configuration and the correspondence between Kuiper's and Sputnik's
 * template parameters are documented in spmm_bench_config.h.
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
#include <nvml.h>

/* The one tile configuration both kernels are pinned to. */
#include "spmm_bench_config.h"

/* Kuiper SpMM (f32) */
#include "Klas_SPMM.h"

/* Sputnik SpMM */
#include "sputnik/spmm/cuda_spmm.h"

/*
 * Kuiper: the generated instance for this tile shape, e.g.
 * Klas_SPMM_g_spmm_f32_128x128x32.  NOT the dispatcher.
 */
#define Klas_f KLAS_SPMM_FN(CFG_BLOCK_ITEMS_K, CFG_BLOCK_ITEMS_X, CFG_BLOCK_WIDTH)

/*
 * Asynchronous variant, taking a stream. Used for the timing loop so that
 * Kuiper and Sputnik are measured under the same launch model.
 */
#define Klas_f_on KLAS_SPMM_FN_ON(CFG_BLOCK_ITEMS_K, CFG_BLOCK_ITEMS_X, CFG_BLOCK_WIDTH)

/* Sputnik: the matching config, instantiated in sputnik_spmm_ex.cu. */
typedef BENCH_SPUTNIK_CONFIG BenchConfig;

/*
 * Sputnik entry point for the benchmark. Calls the fixed kernel directly,
 * bypassing CudaSpmm()'s kernel-selection heuristic.
 */
static cudaError_t sputnik_spmm(int m, int k, int n, int nonzeros,
                                const int *row_indices, const float *values,
                                const int *row_offsets,
                                const int *column_indices, const float *dense,
                                float *out, cudaStream_t stream)
{
#ifdef BENCH_SPUTNIK_DISPATCH
    /* Sputnik's own kernel-selection heuristic, i.e. Sputnik as shipped. */
    return sputnik::CudaSpmm(m, k, n, nonzeros, row_indices, values,
                             row_offsets, column_indices, dense, out, stream);
#else
    return sputnik::CudaSpmmEx<BenchConfig>(m, k, n, nonzeros, row_indices,
                                            values, row_offsets, column_indices,
                                            dense, /* bias = */ nullptr, out,
                                            stream);
#endif
}

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
 * Uniform random float in [0, 1).
 *
 * Deliberately NOT small integers: with integer operands every partial sum is
 * exactly representable in fp32, so any accumulation order gives bit-identical
 * results and the EXACT check below would be vacuous. Random mantissas make
 * bit-exactness actually mean the two kernels accumulated in the same order.
 */
static float rand_unit(void)
{
    return (float)rand() / ((float)RAND_MAX + 1.0f);
}

/*
 * Build CSR from per-row density percentages.
 * Helper used by both uniform and non-uniform generators.
 */
static void gen_sparse_from_row_densities(int rows, int cols,
                                          const std::vector<int> &row_density_pct,
                                          CSR &csr)
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
        int d = row_density_pct[i];
        for (int j = 0; j < cols; j++) {
            if (rand() % 100 < d) {
                csr.values.push_back(rand_unit());
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

/*
 * Generate a random sparse matrix in CSR format.
 * density_pct is in [0, 100].  (Uniform: every row has the same density.)
 */
static void gen_sparse(int rows, int cols, int density_pct, CSR &csr)
{
    std::vector<int> d(rows, density_pct);
    gen_sparse_from_row_densities(rows, cols, d, csr);
}

/*
 * Sparsity distribution shapes for non-uniform row densities.
 *
 *   powerlaw  – density ∝ 1/rank^alpha  (a few very dense rows, long tail of sparse ones)
 *   bimodal   – half the rows at high density, half at low density
 *   linear    – density decreases linearly from max to min across rows
 */
enum class SparsityShape { powerlaw, bimodal, linear };

static const char *shape_name(SparsityShape s)
{
    switch (s) {
        case SparsityShape::powerlaw: return "powerlaw";
        case SparsityShape::bimodal:  return "bimodal";
        case SparsityShape::linear:   return "linear";
    }
    return "?";
}

/*
 * Generate a sparse matrix whose per-row density varies according to `shape`.
 * `avg_density_pct` controls the overall density (0–100).
 */
static void gen_sparse_nonuniform(int rows, int cols, int avg_density_pct,
                                  SparsityShape shape, CSR &csr)
{
    std::vector<int> d(rows);

    switch (shape) {
    case SparsityShape::powerlaw: {
        /* Power-law: rank-based density, d_i ∝ 1/(i+1)^0.8 */
        double alpha = 0.8;
        std::vector<double> raw(rows);
        double sum = 0;
        for (int i = 0; i < rows; i++) {
            raw[i] = 1.0 / pow(i + 1.0, alpha);
            sum += raw[i];
        }
        /* Scale so that mean density = avg_density_pct */
        double scale = (double)avg_density_pct * rows / sum;
        for (int i = 0; i < rows; i++)
            d[i] = std::min(100, std::max(1, (int)(raw[i] * scale + 0.5)));
        /* Shuffle so dense rows aren't all at the top */
        for (int i = rows - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            std::swap(d[i], d[j]);
        }
        break;
    }
    case SparsityShape::bimodal: {
        /* Half the rows at 3× avg density, half at ~0 */
        int hi = std::min(100, avg_density_pct * 3);
        /* Solve: (rows/2)*hi + (rows/2)*lo = rows*avg  =>  lo = 2*avg - hi */
        int lo = std::max(1, 2 * avg_density_pct - hi);
        for (int i = 0; i < rows; i++)
            d[i] = (i < rows / 2) ? hi : lo;
        /* Shuffle */
        for (int i = rows - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            std::swap(d[i], d[j]);
        }
        break;
    }
    case SparsityShape::linear: {
        /* Linearly from 2*avg down to ~0 */
        int hi = std::min(100, avg_density_pct * 2);
        for (int i = 0; i < rows; i++) {
            double frac = (rows > 1) ? (double)i / (rows - 1) : 0;
            d[i] = std::max(1, (int)(hi * (1.0 - frac) + 0.5));
        }
        /* Shuffle */
        for (int i = rows - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            std::swap(d[i], d[j]);
        }
        break;
    }
    }

    gen_sparse_from_row_densities(rows, cols, d, csr);
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
    /*
     * Structurally identical to bench_sputnik below: one stream for the whole
     * measurement, asynchronous launches, a single synchronize at the end.
     * Using the synchronous entry point here would instead charge Kuiper a
     * cudaStreamCreate/Synchronize/Destroy per iteration, which is not what
     * Sputnik is being asked to do.
     */
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    for (int i = 0; i < warmup; i++) {
        Klas_f_on(rows, shared, cols, dA, d_row_indices, dB, dC, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; i++) {
        Klas_f_on(rows, shared, cols, dA, d_row_indices, dB, dC, stream);
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

static float bench_sputnik(int m, int k, int n, int nnz,
                           int *d_row_indices, float *d_values,
                           int *d_row_offsets, int *d_col_indices,
                           float *d_dense, float *d_out,
                           int warmup, int iters)
{
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    for (int i = 0; i < warmup; i++) {
        CHECK_CUDA(sputnik_spmm(m, k, n, nnz, d_row_indices, d_values,
                                d_row_offsets, d_col_indices, d_dense, d_out,
                                stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; i++) {
        CHECK_CUDA(sputnik_spmm(m, k, n, nnz, d_row_indices, d_values,
                                d_row_offsets, d_col_indices, d_dense, d_out,
                                stream));
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
/* Output comparison                                                   */
/* ------------------------------------------------------------------ */

/*
 * Raw bit pattern of a float. Used to check for bit-identical output rather
 * than IEEE equality: == would call +0.0 and -0.0 equal (they are not the same
 * bit pattern, and tell us the two kernels accumulated differently) and would
 * call a NaN different from itself.
 */
static inline uint32_t float_bits(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

/* ------------------------------------------------------------------ */
/* Run one benchmark configuration                                     */
/* ------------------------------------------------------------------ */

/*
 * Core benchmark: takes a pre-built CSR and runs both kernels.
 * `label` is a free-form string printed in the density column.
 */
static void run_bench_csr(CSR &csr, int cols, const char *label,
                          int warmup, int iters, int rounds)
{
    int rows   = csr.rows;
    int shared = csr.cols;

    /*
     * Kuiper does not predicate the column tail, so the n-dimension tile must
     * divide cols exactly. Sputnik is run with kPredicateLoads = false for the
     * same reason.
     */
    if (cols % CFG_BLOCK_ITEMS_X != 0) {
        fprintf(stderr, "SKIP %dx%dx%d @%s: cols must be a multiple of "
                "blockItemsX (%d)\n",
                rows, shared, cols, label, CFG_BLOCK_ITEMS_X);
        return;
    }

    /* Dense B matrix (random float) */
    std::vector<float> B(shared * cols);
    for (int i = 0; i < shared * cols; i++)
        B[i] = rand_unit();

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

    Klas_f(rows, shared, cols, dA_k, (uint32_t*)d_row_indices, dB_k, dC_k);
    CHECK_CUDA(cudaDeviceSynchronize());

    {
        cudaStream_t s;
        CHECK_CUDA(cudaStreamCreate(&s));
        CHECK_CUDA(sputnik_spmm(rows, shared, cols, csr.nnz,
                                d_row_indices, d_values, d_row_offsets,
                                d_col_indices, d_dense, d_out, s));
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
    bool bit_exact = true;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float vk = C_kuiper[i * cols + j];
            float vs = C_sputnik[i * cols + j];
            if (float_bits(vk) != float_bits(vs))
                bit_exact = false;
            float diff = fabsf(vk - vs);
            float denom = fmaxf(fabsf(vk), fabsf(vs));
            float rel = (denom > 0) ? diff / denom : diff;
            if (rel > max_reldiff) max_reldiff = rel;
            /* tolerance: fp32 accumulation order may differ between kernels */
            if (rel > 1e-4f && diff > 1e-5f) {
                if (mismatches == 0)
                    fprintf(stderr,
                        "  MISMATCH at (%d,%d): kuiper=%.6f sputnik=%.6f "
                        "(diff=%.6f rel=%.6f)\n", i, j, vk, vs, diff, rel);
                mismatches++;
            }
        }
    }

    /*
     * EXACT means every output word is bit-identical. With random-mantissa
     * operands fp32 addition is not associative, so this genuinely implies the
     * two kernels accumulated in the same order -- which is what we want when
     * Kuiper is meant to reproduce Sputnik's kernel exactly. OK means they
     * agree only within tolerance, i.e. same result, different order.
     */
    char check[64];
    if (mismatches != 0)
        snprintf(check, sizeof(check), "FAIL (maxrel=%.1e)", max_reldiff);
    else if (bit_exact)
        snprintf(check, sizeof(check), "EXACT");
    else
        snprintf(check, sizeof(check), "OK (maxrel=%.1e)", max_reldiff);

    /* Effective FLOPs: 2 * nnz * cols (one mul + one add per nonzero per output col) */
    double flops = 2.0 * csr.nnz * cols;

    /*
     * This GPU power-caps under load (nvidia-smi reports SW_POWER_CAP active,
     * SM clock swinging 1740-2047 MHz), so whichever kernel is timed second
     * systematically runs on a hotter, slower GPU. Alternate the order across
     * rounds and keep the minimum, which is the throttling-robust estimator:
     * the fastest observed run is the one least perturbed by clock droop.
     *
     * Lock the clocks for cleaner numbers still:
     *   sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 1500,1500
     */
    float ms_kuiper = INFINITY, ms_sputnik = INFINITY;
    for (int r = 0; r < rounds; r++) {
        if (r % 2 == 0) {
            ms_kuiper  = fminf(ms_kuiper,  bench_kuiper(rows, shared, cols,
                               (uint32_t*)d_row_indices, dA_k, dB_k, dC_k, warmup, iters));
            ms_sputnik = fminf(ms_sputnik, bench_sputnik(rows, shared, cols, csr.nnz,
                               d_row_indices, d_values, d_row_offsets,
                               d_col_indices, d_dense, d_out, warmup, iters));
        } else {
            ms_sputnik = fminf(ms_sputnik, bench_sputnik(rows, shared, cols, csr.nnz,
                               d_row_indices, d_values, d_row_offsets,
                               d_col_indices, d_dense, d_out, warmup, iters));
            ms_kuiper  = fminf(ms_kuiper,  bench_kuiper(rows, shared, cols,
                               (uint32_t*)d_row_indices, dA_k, dB_k, dC_k, warmup, iters));
        }
    }

    double gflops_kuiper  = flops / (ms_kuiper  * 1e6);
    double gflops_sputnik = flops / (ms_sputnik * 1e6);
    double speedup = ms_kuiper / ms_sputnik; /* >1 means sputnik is faster */

    printf("%-6d %-6d %-6d %-16s %-8d  "
           "%8.3f ms (%6.1f GFLOP/s)  "
           "%8.3f ms (%6.1f GFLOP/s)  "
           "%6.2fx  %s\n",
           rows, shared, cols, label, csr.nnz,
           ms_kuiper, gflops_kuiper,
           ms_sputnik, gflops_sputnik,
           speedup, check);

    /* Cleanup */
    cudaFree(dA_k.elems); cudaFree(dA_k.col_ind); cudaFree(dA_k.row_off);
    cudaFree(dB_k); cudaFree(dC_k);
    cudaFree(d_row_indices); cudaFree(d_values);
    cudaFree(d_row_offsets); cudaFree(d_col_indices);
    cudaFree(d_dense); cudaFree(d_out);
}

/* Uniform-density convenience wrapper (original interface) */
static void run_bench(int rows, int shared, int cols, int density_pct,
                      int warmup, int iters, int rounds)
{
    CSR csr;
    gen_sparse(rows, shared, density_pct, csr);
    char label[16];
    snprintf(label, sizeof(label), "%d%%", density_pct);
    run_bench_csr(csr, cols, label, warmup, iters, rounds);
}

/* Non-uniform-density convenience wrapper */
static void run_bench_nonuniform(int rows, int shared, int cols,
                                 int avg_density_pct, SparsityShape shape,
                                 int warmup, int iters, int rounds)
{
    CSR csr;
    gen_sparse_nonuniform(rows, shared, avg_density_pct, shape, csr);
    char label[32];
    snprintf(label, sizeof(label), "~%d%% %s", avg_density_pct, shape_name(shape));
    run_bench_csr(csr, cols, label, warmup, iters, rounds);
}

/* ------------------------------------------------------------------ */
/* Swizzle effect: compare sorted row_indices vs identity             */
/* ------------------------------------------------------------------ */

/*
 * Time a single kernel variant with a given row_indices permutation.
 * Returns elapsed ms per iteration (after warmup).
 */
static float bench_kuiper_with_perm(int rows, int shared, int cols,
                                    const std::vector<uint32_t> &perm,
                                    Kuiper_Sparse_Matrix_smatrix__float dA,
                                    float *dB, float *dC,
                                    int warmup, int iters)
{
    uint32_t *d_perm = to_gpu(perm);
    float ms = bench_kuiper(rows, shared, cols, d_perm, dA, dB, dC, warmup, iters);
    cudaFree(d_perm);
    return ms;
}

static float bench_sputnik_with_perm(int rows, int shared, int cols, int nnz,
                                     const std::vector<int> &perm,
                                     float *d_values, int *d_row_offsets,
                                     int *d_col_indices, float *d_dense,
                                     float *d_out,
                                     int warmup, int iters)
{
    int *d_perm = to_gpu(perm);
    float ms = bench_sputnik(rows, shared, cols, nnz,
                             d_perm, d_values, d_row_offsets,
                             d_col_indices, d_dense, d_out,
                             warmup, iters);
    cudaFree(d_perm);
    return ms;
}

/*
 * Run a single swizzle-effect test: generate a non-uniform matrix, then
 * benchmark both kernels with sorted (swizzled) and identity row indices.
 */
static void run_swizzle_test(int rows, int shared, int cols,
                             int avg_density_pct, SparsityShape shape,
                             int warmup, int iters)
{
    if (cols % CFG_BLOCK_ITEMS_X != 0) return;

    CSR csr;
    gen_sparse_nonuniform(rows, shared, avg_density_pct, shape, csr);

    /* Identity permutation (no load balancing) */
    std::vector<uint32_t> identity_u(rows);
    std::vector<int>      identity_i(rows);
    std::iota(identity_u.begin(), identity_u.end(), 0);
    std::iota(identity_i.begin(), identity_i.end(), 0);

    /* Sorted permutation (load-balanced swizzle) */
    std::vector<uint32_t> swizzled_u(rows);
    for (int i = 0; i < rows; i++)
        swizzled_u[i] = (uint32_t)csr.row_indices[i];

    /* Dense B */
    std::vector<float> B(shared * cols);
    for (int i = 0; i < shared * cols; i++)
        B[i] = rand_unit();

    /* Upload shared data */
    Kuiper_Sparse_Matrix_smatrix__float dA_k;
    dA_k.nnz     = (uint32_t)csr.nnz;
    dA_k.elems   = to_gpu(csr.values);
    dA_k.col_ind = to_gpu(csr.col_ind);
    dA_k.row_off = to_gpu(csr.row_off);
    float *dB_k = to_gpu(B);
    float *dC_k = gpu_zeros<float>(rows * cols);

    float *d_values      = to_gpu(csr.values);
    int   *d_row_offsets = to_gpu(csr.row_off_i);
    int   *d_col_indices = to_gpu(csr.col_ind_i);
    float *d_dense       = to_gpu(B);
    float *d_out         = gpu_zeros<float>(rows * cols);

    /* Bench Kuiper: identity vs swizzled */
    float k_id = bench_kuiper_with_perm(rows, shared, cols, identity_u, dA_k, dB_k, dC_k, warmup, iters);
    float k_sw = bench_kuiper_with_perm(rows, shared, cols, swizzled_u, dA_k, dB_k, dC_k, warmup, iters);

    /* Bench Sputnik: identity vs swizzled */
    float s_id = bench_sputnik_with_perm(rows, shared, cols, csr.nnz,
                                         identity_i, d_values, d_row_offsets,
                                         d_col_indices, d_dense, d_out, warmup, iters);
    float s_sw = bench_sputnik_with_perm(rows, shared, cols, csr.nnz,
                                         csr.row_indices, d_values, d_row_offsets,
                                         d_col_indices, d_dense, d_out, warmup, iters);

    char shape_label[32];
    snprintf(shape_label, sizeof(shape_label), "~%d%% %s", avg_density_pct, shape_name(shape));

    printf("%-6d %-6d %-6d %-16s %-8d  "
           "Kuiper: %7.3f → %7.3f ms (%5.2fx)  "
           "Sputnik: %7.3f → %7.3f ms (%5.2fx)\n",
           rows, shared, cols, shape_label, csr.nnz,
           k_id, k_sw, k_id / k_sw,
           s_id, s_sw, s_id / s_sw);

    cudaFree(dA_k.elems); cudaFree(dA_k.col_ind); cudaFree(dA_k.row_off);
    cudaFree(dB_k); cudaFree(dC_k);
    cudaFree(d_values); cudaFree(d_row_offsets); cudaFree(d_col_indices);
    cudaFree(d_dense); cudaFree(d_out);
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

/* Burn cycles to bring the GPU out of its idle clock state. */
__global__ static void warmup_kernel(float *sink, int iters)
{
    float a = threadIdx.x * 1e-3f, b = 1.000001f, c = 0.999999f;
    for (int i = 0; i < iters; i++) { a = fmaf(a, b, c); a = fmaf(a, c, b); }
    if (a == 12345.678f) sink[0] = a;   /* never taken; defeats DCE */
}

#define NVML_TRY(call)                                                       \
    do {                                                                     \
        nvmlReturn_t rc = (call);                                            \
        if (rc != NVML_SUCCESS) {                                            \
            fprintf(stderr, "NVML: %s failed: %s\n", #call,                  \
                    nvmlErrorString(rc));                                    \
            return -1;                                                       \
        }                                                                    \
    } while (0)

/* Throttle reasons that mean the clock is being pulled down under load. */
static const unsigned long long kBadThrottle =
    nvmlClocksThrottleReasonSwPowerCap | nvmlClocksThrottleReasonHwSlowdown |
    nvmlClocksThrottleReasonSwThermalSlowdown |
    nvmlClocksThrottleReasonHwThermalSlowdown |
    nvmlClocksThrottleReasonHwPowerBrakeSlowdown;

static void describe_throttle(unsigned long long r)
{
    if (r & nvmlClocksThrottleReasonSwPowerCap)
        printf("    - SW power cap (the board is at its power limit)\n");
    if (r & nvmlClocksThrottleReasonHwSlowdown)
        printf("    - HW slowdown\n");
    if (r & nvmlClocksThrottleReasonSwThermalSlowdown)
        printf("    - SW thermal slowdown\n");
    if (r & nvmlClocksThrottleReasonHwThermalSlowdown)
        printf("    - HW thermal slowdown\n");
    if (r & nvmlClocksThrottleReasonHwPowerBrakeSlowdown)
        printf("    - HW power brake\n");
}

/*
 * Spin the GPU for `ms` milliseconds while sampling the SM clock and the
 * throttle reasons through NVML.
 *
 * Two jobs. First, this GPU idles at 210 MHz and takes order a second to ramp,
 * so without a warm-up the earliest cases get measured on a slow clock.
 * Second, it tells us whether timings from this machine are trustworthy at
 * all: if the clock is being throttled under load, whichever kernel runs
 * second is systematically penalised, which can easily manufacture a bogus
 * multi-x "win".
 *
 * Returns 0 if the clock held steady, 1 if it did not, -1 if NVML was
 * unavailable (in which case we cannot tell, and say so).
 */
static int gpu_warmup_and_check(int ms, unsigned int *clk_min,
                                unsigned int *clk_max,
                                unsigned long long *reasons)
{
    float *sink = gpu_zeros<float>(1);
    *clk_min = ~0u; *clk_max = 0; *reasons = 0;

    nvmlDevice_t dev;
    bool have_nvml = (nvmlInit() == NVML_SUCCESS);
    if (have_nvml) {
        int cuda_dev = 0;
        cudaGetDevice(&cuda_dev);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, cuda_dev);
        char pci[32];
        snprintf(pci, sizeof(pci), "%08X:%02X:%02X.0", props.pciDomainID,
                 props.pciBusID, props.pciDeviceID);
        if (nvmlDeviceGetHandleByPciBusId(pci, &dev) != NVML_SUCCESS)
            have_nvml = false;
    }

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0));
    float elapsed = 0;
    do {
        for (int i = 0; i < 20; i++)
            warmup_kernel<<<1024, 256>>>(sink, 4096);
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed, t0, t1));
        if (have_nvml && elapsed > 300.0f) {  /* skip the ramp-up transient */
            unsigned int c = 0;
            unsigned long long r = 0;
            if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &c) == NVML_SUCCESS) {
                if (c < *clk_min) *clk_min = c;
                if (c > *clk_max) *clk_max = c;
            }
            if (nvmlDeviceGetCurrentClocksThrottleReasons(dev, &r) == NVML_SUCCESS)
                *reasons |= r;
        }
    } while (elapsed < (float)ms);
    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));
    cudaFree(sink);
    if (have_nvml) nvmlShutdown();

    if (!have_nvml || *clk_max == 0) return -1;
    bool throttled = (*reasons & kBadThrottle) != 0;
    bool unstable  = (*clk_max - *clk_min) * 100u > *clk_max * 2u;  /* >2% swing */
    return (throttled || unstable) ? 1 : 0;
}

/*
 * Refuse to produce numbers nobody should trust, unless explicitly forced.
 */
static void require_stable_clocks(bool force)
{
    unsigned int lo = 0, hi = 0;
    unsigned long long reasons = 0;
    int verdict = gpu_warmup_and_check(2000, &lo, &hi, &reasons);

    if (verdict < 0) {
        printf("GPU warm-up done. NVML unavailable: cannot verify clock "
               "stability.\n");
        return;
    }
    printf("GPU warm-up done. SM clock under load: %u-%u MHz.\n", lo, hi);
    if (verdict == 0)
        return;

    printf("\n");
    printf("*** WARNING: this GPU's clock is not pinned. ***\n\n");
    if (reasons & kBadThrottle) {
        printf("  Active throttle reasons under load:\n");
        describe_throttle(reasons);
    }
    if ((hi - lo) * 100u > hi * 2u)
        printf("  SM clock varied %u-%u MHz (%.1f%%) during a steady load.\n",
               lo, hi, 100.0 * (hi - lo) / hi);
    printf("\n"
           "  Timings will drift, and because the two kernels are measured\n"
           "  back to back, whichever runs second is systematically penalised.\n"
           "  That is more than enough to invent a several-x difference that\n"
           "  does not exist. Pin the clock first:\n\n"
           "      sudo nvidia-smi -pm 1\n"
           "      sudo nvidia-smi -lgc %u,%u\n\n"
           "  and restore it afterwards with:\n\n"
           "      sudo nvidia-smi -rgc\n\n"
           "  Re-run with -f to benchmark anyway and accept the noise.\n",
           lo, lo);
    if (!force) {
        exit(1);
    }
    printf("\n  -f given: continuing anyway. Treat the numbers below as "
           "indicative only.\n");
}

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

/*
 * Print the fixed tile configuration both kernels run with, so a result table
 * is self-describing.
 */
static void print_config()
{
    printf("Fixed configuration (no dispatcher on either side): %s\n",
           BENCH_CONFIG_NAME);
    printf("  blockItemsY = %d, blockItemsK = %d, blockItemsX = %d, "
           "blockWidth = %d\n",
           BenchConfig::kBlockItemsY, BenchConfig::kBlockItemsK,
           BenchConfig::kBlockItemsX, BenchConfig::kBlockWidth);
    printf("  Kuiper:  %s\n",
           "Klas_SPMM_g_spmm_f32_" BENCH_CONFIG_NAME);
    printf("  Sputnik: CudaSpmmEx<SpmmConfig<float, float4, float4, %d, %d, "
           "%d, %d, %d, %s>>\n",
           BenchConfig::kBlockItemsY, BenchConfig::kBlockItemsK,
           BenchConfig::kBlockItemsX, BenchConfig::kBlockWidth,
           BenchConfig::kResidueUnroll,
           BenchConfig::kPredicateLoads ? "true" : "false");
    printf("  %d threads/block, %d float4 per thread in the n-dimension\n",
           BenchConfig::kThreadsPerBlock, BenchConfig::kThreadItemsX);
    printf("  Problems whose cols is not a multiple of %d are skipped.\n\n",
           CFG_BLOCK_ITEMS_X);
}

int main(int argc, char **argv)
{
    int warmup = 5;
    int iters  = 20;
    int rounds = 4;   /* must be even: see the round-up below */
    bool force = false;

    /* Parse optional --warmup and --iters */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
            iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--rounds") == 0 && i + 1 < argc)
            rounds = atoi(argv[++i]);
        else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--force") == 0)
            force = true;
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--warmup N] [--iters N] [--rounds N] [-f]\n"
                   "  -f, --force   benchmark even if the GPU clock is not "
                   "pinned\n", argv[0]);
            return 0;
        }
    }

    /*
     * The round loop alternates which kernel is timed first. An odd count
     * would give one of them an extra turn in the favourable first slot and
     * thus an extra shot at a low minimum, so round up to even.
     */
    if (rounds % 2 != 0) {
        printf("note: --rounds %d is odd, using %d so each kernel is timed "
               "first equally often\n", rounds, rounds + 1);
        rounds++;
    }

    srand(42);
    print_gpu_info();
    print_config();
    printf("Timing: min over %d rounds; kernel order alternated, so each is "
           "timed first %d time%s.\n", rounds, rounds / 2,
           rounds / 2 == 1 ? "" : "s");
    require_stable_clocks(force);
    printf("\n");

    printf("%-6s %-6s %-6s %-16s %-8s  %-28s  %-28s  %-7s  %s\n",
           "rows", "K", "cols", "density", "nnz",
           "Kuiper (fixed cfg)", "Sputnik (fixed cfg)", "K/S", "check");
    printf("%s\n", std::string(138, '-').c_str());

    /* n-dimension sweep. cols must stay a multiple of blockItemsX. */
    printf("\n--- n-dimension sweep ---\n");
    for (int mult : {1, 2, 4, 8})
        run_bench(128, 1024, CFG_BLOCK_ITEMS_X * mult, 10, warmup, iters, rounds);

    /* Square matrices at various sizes and densities */
    int sizes[]     = { 256, 512, 1024, 2048 };
    int densities[] = { 1, 5, 10, 30, 50 };

    for (int s : sizes)
        for (int d : densities)
            run_bench(s, s, s < 128 ? 128 : s, d, warmup, iters, rounds);

    printf("\n--- Non-square matrices ---\n");
    run_bench(512,  1024, 256,  10, warmup, iters, rounds);
    run_bench(1024, 512,  256,  10, warmup, iters, rounds);
    run_bench(2048, 256,  128,  10, warmup, iters, rounds);
    run_bench(256,  256,  1024, 10, warmup, iters, rounds);
    run_bench(1024, 1024, 128,  5,  warmup, iters, rounds);

    /* ---- Non-uniform sparsity (exercises load-balancing swizzle) ---- */
    printf("\n--- Non-uniform sparsity (powerlaw: few dense rows, long sparse tail) ---\n");
    for (int s : {512, 1024, 2048})
        for (int d : {5, 10, 30})
            run_bench_nonuniform(s, s, s, d, SparsityShape::powerlaw, warmup, iters, rounds);

    printf("\n--- Non-uniform sparsity (bimodal: half dense, half sparse) ---\n");
    for (int s : {512, 1024, 2048})
        for (int d : {5, 10, 30})
            run_bench_nonuniform(s, s, s, d, SparsityShape::bimodal, warmup, iters, rounds);

    printf("\n--- Non-uniform sparsity (linear gradient) ---\n");
    for (int s : {512, 1024, 2048})
        for (int d : {5, 10, 30})
            run_bench_nonuniform(s, s, s, d, SparsityShape::linear, warmup, iters, rounds);

    /* ---- Swizzle effect: identity vs load-balanced row ordering ---- */
    printf("\n--- Swizzle effect: identity → swizzled (speedup from load balancing) ---\n");

    for (SparsityShape sh : {SparsityShape::powerlaw, SparsityShape::bimodal, SparsityShape::linear})
        for (int s : {512, 1024, 2048})
            for (int d : {5, 10, 30})
                run_swizzle_test(s, s, s, d, sh, warmup, iters);

    return 0;
}
