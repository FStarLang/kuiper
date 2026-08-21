/*
 * Fixed SpMM tile configuration, shared by the benchmark driver
 * (spmm_bench.cu) and the Sputnik instantiation TU (sputnik_spmm_ex.cu).
 *
 * The point of this file is to pin *one* kernel configuration on both sides so
 * that the benchmark measures the same algorithm with the same tile shape,
 * rather than whatever each library's dispatcher happens to pick.
 *
 * Mapping between the two parameterizations
 * -----------------------------------------
 *
 *   Sputnik SpmmConfig<ScalarValue, SparseValue, DenseValue,
 *                      kBlockItemsY, kBlockItemsK, kBlockItemsX, kBlockWidth,
 *                      kResidueUnroll, kPredicateLoads, ...>
 *
 *   Kuiper   inst f32 blockItemsK blockItemsX blockWidth
 *
 *   Kuiper                          Sputnik
 *   ----------------------------    ---------------------------------------
 *   blockItemsK                     kBlockItemsK
 *   blockItemsX                     kBlockItemsX
 *   blockWidth                      kBlockWidth
 *   (none: one row per block)       kBlockItemsY = 1
 *   chunk f32 = 4 (float4 loads)    SparseValue = DenseValue = float4
 *   no column-tail predication      kPredicateLoads = false
 *
 * Constraints that the pair must satisfy
 * --------------------------------------
 *
 *   Kuiper  (Kuiper.Sparse.SPMM.spmm):
 *     blockWidth * 4 divides blockItemsK    (4-wide loads of A's values/indices)
 *     blockWidth * 4 divides blockItemsX    (4-wide loads of B and C)
 *     blockItemsX divides cols              (no predicated column tail)
 *
 *   Sputnik (SpmmConfig static_asserts):
 *     kBlockItemsY * kBlockWidth is a multiple of 32   (whole warps)
 *
 * Kuiper's kernel launches one thread block per (row, column tile) with
 * blockWidth threads, i.e. kBlockItemsY is always 1.  Combined with Sputnik's
 * whole-warp requirement that forces blockWidth >= 32, and the 4-wide load
 * requirement then forces blockItemsK, blockItemsX >= 4 * blockWidth >= 128.
 * So the smallest configuration both sides can express is 128 x 128 x 32,
 * which is the default below.
 *
 * Other pairs that work (all have blockWidth in {32, 64, 128}):
 *   K x X x W = 128x128x32  128x256x32  128x512x32
 *               256x128x32  256x256x32  256x512x32  256x256x64  256x512x64
 *               512x128x32  512x256x32  512x512x32  512x256x64  512x512x64
 *               512x512x128
 * The Kuiper side must have the matching `g_spmm_f32_<K>x<X>x<W>` instance
 * generated in src/klas/Klas.SPMM.fst.sh; the Sputnik side is instantiated on
 * demand by sputnik_spmm_ex.cu, so any of these can be selected with e.g.
 *
 *   make NVCC_FLAGS="... -DCFG_BLOCK_ITEMS_K=256 -DCFG_BLOCK_ITEMS_X=256 \
 *                        -DCFG_BLOCK_WIDTH=64"
 *
 * Note that blockItemsX also bounds which problems can run: the benchmark
 * skips any configuration whose `cols` is not a multiple of blockItemsX.
 */

#ifndef SPMM_BENCH_CONFIG_H
#define SPMM_BENCH_CONFIG_H

#include "sputnik/spmm/spmm_config.h"

#ifndef CFG_BLOCK_ITEMS_K
#define CFG_BLOCK_ITEMS_K 128
#endif

#ifndef CFG_BLOCK_ITEMS_X
#define CFG_BLOCK_ITEMS_X 128
#endif

#ifndef CFG_BLOCK_WIDTH
#define CFG_BLOCK_WIDTH 32
#endif

/* Name of the Kuiper instance for a given tile shape. */
#define KLAS_SPMM_FN_(K, X, W) Klas_SPMM_g_spmm_f32_##K##x##X##x##W
#define KLAS_SPMM_FN(K, X, W)  KLAS_SPMM_FN_(K, X, W)

/*
 * The stream-parametric variant of the same instance. It launches and returns
 * without synchronizing, so the benchmark can issue every iteration into one
 * stream and wait once -- the same thing Sputnik's CudaSpmmEx does. The
 * synchronous entry point above instead creates a stream, launches,
 * synchronizes and destroys the stream on every single call, which is a large
 * fixed cost next to a kernel that can run in a few microseconds.
 */
#define KLAS_SPMM_FN_ON_(K, X, W) Klas_SPMM_g_spmm_f32_##K##x##X##x##W##_on
#define KLAS_SPMM_FN_ON(K, X, W)  KLAS_SPMM_FN_ON_(K, X, W)

/* The matching Sputnik configuration. */
#define BENCH_SPUTNIK_CONFIG                                              \
    sputnik::SpmmConfig<float,  /* ScalarValue                          */ \
                        float4, /* SparseValue: 4-wide loads of A       */ \
                        float4, /* DenseValue:  4-wide loads of B and C */ \
                        1,      /* kBlockItemsY: Kuiper has no m-tiling */ \
                        CFG_BLOCK_ITEMS_K,                                 \
                        CFG_BLOCK_ITEMS_X,                                 \
                        CFG_BLOCK_WIDTH,                                   \
                        4,      /* kResidueUnroll: Sputnik default      */ \
                        false>  /* kPredicateLoads: X divides cols      */

#define BENCH_STR_(x) #x
#define BENCH_STR(x)  BENCH_STR_(x)

#define BENCH_CONFIG_NAME                                                 \
    BENCH_STR(CFG_BLOCK_ITEMS_K) "x" BENCH_STR(CFG_BLOCK_ITEMS_X) "x"     \
    BENCH_STR(CFG_BLOCK_WIDTH)

#endif /* SPMM_BENCH_CONFIG_H */
