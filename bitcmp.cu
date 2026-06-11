// Unified bit-equivalence driver for the matmul kernels:
//
//   imp1 : imp1.cu              forward  K-accumulation (i = 0 .. k-1)
//   imp2 : imp2.cu              reverse  K-accumulation (i = k-1 .. 0)
//   imp3 : imp3.cu              tiled    K-accumulation (tiles of 16, forward)
//   imp4 : imp4.cu              Kahan-compensated forward K-accumulation
//   imp5 : imp5.cu              tiled (16) forward, Kahan-compensated across tiles
//   kw1  : KWitness1_matmul_f32 verified Kuiper witness, forward  (bit-eq to imp1)
//   kw2  : KWitness2_matmul_f32 verified Kuiper witness, reverse  (bit-eq to imp2)
//   kw3  : KWitness3_matmul_f32 verified Kuiper witness, tiled    (bit-eq to imp3)
//   kw4  : KWitness4_matmul_f32 verified Kuiper witness, Kahan    (bit-eq to imp4)
//   kw5  : KWitness5_matmul_f32 verified Kuiper witness, tiled+Kahan (bit-eq to imp5)
//
// imp1..imp5 each define `ker`/`matmul`, so each is pulled into its own
// namespace. The Kuiper witnesses are extracted with distinct names.
//
// Build:
//   make obj/KWitness1.cu obj/KWitness2.cu obj/KWitness3.cu obj/KWitness4.cu obj/KWitness5.cu
//   nvcc -O2 -arch=native -I include -I obj -o bitcmp bitcmp.cu \
//        obj/KWitness1.cu obj/KWitness2.cu obj/KWitness3.cu obj/KWitness4.cu obj/KWitness5.cu
//
// Usage:
//   ./bitcmp                          # default suite over several shapes
//   ./bitcmp A B [m] [n] [k] [seed]   # compare kernel A vs B
//                                     # (A,B in {imp1..imp5,kw1..kw5})
//
// Exit code: 0 if the checked equivalences hold, 1 otherwise, 2 on usage/CUDA error.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <ctime>

#include "KWitness1.h"
#include "KWitness2.h"
#include "KWitness3.h"
#include "KWitness4.h"
#include "KWitness5.h"

namespace imp1 {
#include "imp1.cu"
}
namespace imp2 {
#include "imp2.cu"
}
namespace imp3 {
#include "imp3.cu"
}
namespace imp4 {
#include "imp4.cu"
}
namespace imp5 {
#include "imp5.cu"
}

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,   \
                    __LINE__, cudaGetErrorString(err__));                      \
            exit(2);                                                           \
        }                                                                      \
    } while (0)

// Uniform launcher signature: c = a (m x k) * b (k x n).
typedef void (*kernel_fn)(float *a, float *b, float *c, int m, int n, int k);

static void run_imp1(float *a, float *b, float *c, int m, int n, int k)
{ imp1::matmul(a, b, c, m, n, k); }
static void run_imp2(float *a, float *b, float *c, int m, int n, int k)
{ imp2::matmul(a, b, c, m, n, k); }
static void run_imp3(float *a, float *b, float *c, int m, int n, int k)
{ imp3::matmul(a, b, c, m, n, k); }
static void run_imp4(float *a, float *b, float *c, int m, int n, int k)
{ imp4::matmul(a, b, c, m, n, k); }
static void run_imp5(float *a, float *b, float *c, int m, int n, int k)
{ imp5::matmul(a, b, c, m, n, k); }
static void run_kw1(float *a, float *b, float *c, int m, int n, int k)
{ KWitness1_matmul_f32((uint32_t) m, (uint32_t) n, (uint32_t) k, a, b, c); }
static void run_kw2(float *a, float *b, float *c, int m, int n, int k)
{ KWitness2_matmul_f32((uint32_t) m, (uint32_t) n, (uint32_t) k, a, b, c); }
static void run_kw3(float *a, float *b, float *c, int m, int n, int k)
{ KWitness3_matmul_f32((uint32_t) m, (uint32_t) n, (uint32_t) k, a, b, c); }
static void run_kw4(float *a, float *b, float *c, int m, int n, int k)
{ KWitness4_matmul_f32((uint32_t) m, (uint32_t) n, (uint32_t) k, a, b, c); }
static void run_kw5(float *a, float *b, float *c, int m, int n, int k)
{ KWitness5_matmul_f32((uint32_t) m, (uint32_t) n, (uint32_t) k, a, b, c); }

struct Kernel { const char *name; kernel_fn fn; };
static const Kernel KERNELS[] = {
    { "imp1", run_imp1 },
    { "imp2", run_imp2 },
    { "imp3", run_imp3 },
    { "imp4", run_imp4 },
    { "imp5", run_imp5 },
    { "kw1",  run_kw1  },
    { "kw2",  run_kw2  },
    { "kw3",  run_kw3  },
    { "kw4",  run_kw4  },
    { "kw5",  run_kw5  },
};
static const int NKERNELS = (int) (sizeof(KERNELS) / sizeof(KERNELS[0]));

static const Kernel *find_kernel(const char *name)
{
    for (int i = 0; i < NKERNELS; i++)
        if (strcmp(name, KERNELS[i].name) == 0)
            return &KERNELS[i];
    return NULL;
}

static float rand_float(void)
{
    return (float) (rand() / (double) RAND_MAX) * 2.0f - 1.0f;
}

// Run a kernel on device inputs d_a, d_b; return a host copy of the m*n result.
static float *run_to_host(kernel_fn fn, float *d_a, float *d_b, int m, int n,
                          int k)
{
    size_t c_elems = (size_t) m * n;
    float *d_c;
    CUDA_CHECK(cudaMalloc(&d_c, c_elems * sizeof(float)));
    fn(d_a, d_b, d_c, m, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float *h_c = (float *) malloc(c_elems * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_c, d_c, c_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_c));
    return h_c;
}

// Bit-compare two host result matrices; print a verdict; return # differing.
static size_t bitcmp(const char *a_name, const char *b_name, const float *ca,
                     const float *cb, int m, int n)
{
    size_t c_elems = (size_t) m * n;
    size_t ndiff = 0, first = c_elems;
    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < c_elems; i++) {
        uint32_t x, y;
        memcpy(&x, &ca[i], sizeof(x));
        memcpy(&y, &cb[i], sizeof(y));
        if (x != y) {
            if (first == c_elems) first = i;
            ndiff++;
            double va = ca[i], vb = cb[i], ad = fabs(va - vb);
            if (ad > max_abs) max_abs = ad;
            if (vb != 0.0) {
                double rd = fabs((va - vb) / vb);
                if (rd > max_rel) max_rel = rd;
            }
        }
    }
    if (ndiff == 0) {
        printf("  %-4s vs %-4s : BIT-EQUIVALENT (%zu elements)\n", a_name,
               b_name, c_elems);
    } else {
        size_t r = first / (size_t) n, c = first % (size_t) n;
        printf("  %-4s vs %-4s : DIFFER %zu/%zu; first (%zu,%zu) %.9g vs %.9g; "
               "max|d|=%.3g max rel=%.3g\n",
               a_name, b_name, ndiff, c_elems, r, c, (double) ca[first],
               (double) cb[first], max_abs, max_rel);
    }
    return ndiff;
}

// Allocate + fill A,B, copy to device. Caller frees via free_inputs.
static void make_inputs(int m, int n, int k, unsigned seed, float **d_a,
                        float **d_b)
{
    size_t a_elems = (size_t) m * k, b_elems = (size_t) k * n;
    float *h_a = (float *) malloc(a_elems * sizeof(float));
    float *h_b = (float *) malloc(b_elems * sizeof(float));
    srand(seed);
    for (size_t i = 0; i < a_elems; i++) h_a[i] = rand_float();
    for (size_t i = 0; i < b_elems; i++) h_b[i] = rand_float();
    CUDA_CHECK(cudaMalloc(d_a, a_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_b, b_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(*d_a, h_a, a_elems * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_b, h_b, b_elems * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(h_a);
    free(h_b);
}

// Compare a single pair on one shape. Returns # differing elements.
static size_t compare_pair(const Kernel *ka, const Kernel *kb, int m, int n,
                           int k, unsigned seed)
{
    float *d_a, *d_b;
    make_inputs(m, n, k, seed, &d_a, &d_b);
    float *ca = run_to_host(ka->fn, d_a, d_b, m, n, k);
    float *cb = run_to_host(kb->fn, d_a, d_b, m, n, k);
    size_t nd = bitcmp(ka->name, kb->name, ca, cb, m, n);
    free(ca);
    free(cb);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    return nd;
}

// Default suite: for each shape, verify the witness equivalences (imp1==kw1,
// imp2==kw2) and show imp1 vs imp2 as a contrast (expected to differ).
static int run_suite(void)
{
    struct { int m, n, k; } shapes[] = {
        { 256, 256, 256 }, { 128, 64, 200 }, { 300, 50, 17 },
        { 1, 1, 1024 },    { 64, 64, 333 },  { 512, 512, 64 },
    };
    int nshapes = (int) (sizeof(shapes) / sizeof(shapes[0]));
    const Kernel *imp1k = find_kernel("imp1"), *imp2k = find_kernel("imp2");
    const Kernel *imp3k = find_kernel("imp3"), *imp4k = find_kernel("imp4");
    const Kernel *imp5k = find_kernel("imp5");
    const Kernel *kw1k = find_kernel("kw1"), *kw2k = find_kernel("kw2");
    const Kernel *kw3k = find_kernel("kw3"), *kw4k = find_kernel("kw4");
    const Kernel *kw5k = find_kernel("kw5");
    int failures = 0;

    for (int s = 0; s < nshapes; s++) {
        int m = shapes[s].m, n = shapes[s].n, k = shapes[s].k;
        unsigned seed = (unsigned) (1000 + s);
        printf("shape m=%d n=%d k=%d seed=%u\n", m, n, k, seed);
        // Witness equivalences (must hold):
        if (compare_pair(imp1k, kw1k, m, n, k, seed) != 0) failures++;
        if (compare_pair(imp2k, kw2k, m, n, k, seed) != 0) failures++;
        if (compare_pair(imp3k, kw3k, m, n, k, seed) != 0) failures++;
        if (compare_pair(imp4k, kw4k, m, n, k, seed) != 0) failures++;
        if (compare_pair(imp5k, kw5k, m, n, k, seed) != 0) failures++;
        // Contrast (expected to differ for k>1 due to FP non-associativity):
        compare_pair(imp1k, imp2k, m, n, k, seed);
        compare_pair(imp1k, imp3k, m, n, k, seed);
        compare_pair(imp1k, imp4k, m, n, k, seed);
        compare_pair(imp1k, imp5k, m, n, k, seed);
    }

    printf("\n%s\n", failures == 0
        ? "PASS: every witness is bit-equivalent to its kernel."
        : "FAIL: a witness diverged from its kernel.");
    return failures == 0 ? 0 : 1;
}

int main(int argc, char **argv)
{
    // Pair mode: first arg names a kernel.
    if (argc > 1 && find_kernel(argv[1]) != NULL) {
        if (argc < 3 || find_kernel(argv[2]) == NULL) {
            fprintf(stderr, "Usage: %s A B [m] [n] [k] [seed]  "
                            "(A,B in imp1,imp2,imp3,imp4,imp5,kw1,kw2,kw3,kw4,kw5)\n", argv[0]);
            return 2;
        }
        const Kernel *ka = find_kernel(argv[1]);
        const Kernel *kb = find_kernel(argv[2]);
        int m = (argc > 3) ? atoi(argv[3]) : 256;
        int n = (argc > 4) ? atoi(argv[4]) : 256;
        int k = (argc > 5) ? atoi(argv[5]) : 256;
        unsigned seed = (argc > 6) ? (unsigned) strtoul(argv[6], NULL, 10)
                                   : (unsigned) time(NULL);
        if (m <= 0 || n <= 0 || k <= 0) {
            fprintf(stderr, "Dimensions must be positive\n");
            return 2;
        }
        printf("m=%d n=%d k=%d seed=%u\n", m, n, k, seed);
        return compare_pair(ka, kb, m, n, k, seed) == 0 ? 0 : 1;
    }

    if (argc > 1) {
        fprintf(stderr, "Unknown kernel '%s' (expected imp1,imp2,imp3,imp4,imp5,kw1,kw2,kw3,kw4,kw5)\n",
                argv[1]);
        return 2;
    }

    return run_suite();
}
