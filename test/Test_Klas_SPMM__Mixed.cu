#include "Klas_SPMM.h"
#include "test-common.h"
#include <cmath>
#include <numeric>
#include <type_traits>
#include <vector>

const char *progname = __FILE__;

template <typename T> static T *upload(const std::vector<T> &values)
{
    T *device;
    MUST(cudaMalloc(&device, values.size() * sizeof(T)));
    MUST(cudaMemcpy(device, values.data(), values.size() * sizeof(T),
        cudaMemcpyHostToDevice));
    return device;
}

template <typename CSR, typename TB, typename TC>
static bool run_case(const char *label, uint32_t cols, bool asynchronous,
    void (*sync)(uint32_t, uint32_t, uint32_t, CSR, uint32_t *, TB *, TC *),
    void (*async)(uint32_t, uint32_t, uint32_t, CSR, uint32_t *, TB *, TC *,
        cudaStream_t),
    bool mapped = false, bool wide_accumulator = false)
{
    using TA = typename std::remove_pointer<decltype(CSR{}.elems)>::type;
    const uint32_t rows = 20, shared = 1100, guard = 32;
    // Empty rows, misaligned starts, complete tiles, and partial tiles.
    const uint32_t lengths[rows] = {0, 1, 2, 3, 4, 7, 8, 31, 32, 33, 63, 64, 65,
        255, 256, 257, 511, 512, 513, 1024};
    std::vector<TA> values;
    std::vector<uint32_t> indices, offsets(1, 0), order(rows);
    std::iota(order.rbegin(), order.rend(), 0);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t k = 0; k < lengths[r]; ++k) {
            // Narrow accumulation loses the ones between these endpoints.
            float value = k == 0                ? 2048.0f
                          : k + 1 == lengths[r] ? -2048.0f
                                                : 1.0f;
            values.push_back(TA(value));
            indices.push_back(k);
        }
        offsets.push_back(values.size());
    }
    std::vector<TB> b(shared * cols);
    for (uint32_t k = 0; k < shared; ++k)
        for (uint32_t c = 0; c < cols; ++c)
            // In the f64 cases, a float accumulator loses the small terms
            // between the two 2^25 products in the 65-entry row.
            b[k * cols + c] = TB(
                float(c % 4 + 1) * 0.25f *
                (wide_accumulator && (k == 0 || k == 64) ? 16384.0f : 1.0f));
    std::vector<TC> output(rows * cols + guard, TC(-1000.0f));
    for (uint32_t i = 0; i < rows * cols; ++i)
        output[i] = TC(float(int(i % 7) - 3) * 4.0f);
    const auto initial = output;

    CSR a;
    a.nnz = values.size();
    a.elems = upload(values);
    a.col_ind = upload(indices);
    a.row_off = upload(offsets);
    auto dorder = upload(order);
    auto db = upload(b);
    auto dc = upload(output);
    if (asynchronous) {
        cudaStream_t stream;
        MUST(cudaStreamCreate(&stream));
        async(rows, shared, cols, a, dorder, db, dc, stream);
        async(rows, shared, cols, a, dorder, db, dc, stream);
        MUST(cudaStreamSynchronize(stream));
        MUST(cudaStreamDestroy(stream));
    } else {
        sync(rows, shared, cols, a, dorder, db, dc);
    }
    MUST(cudaMemcpy(
        output.data(), dc, output.size() * sizeof(TC), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            double expected = 0;
            for (uint32_t k = offsets[r]; k < offsets[r + 1]; ++k) {
                double a = float(values[k]);
                double bvalue = float(b[indices[k] * cols + c]);
                expected += mapped ? (a + a) * (bvalue + 1.0) : a * bvalue;
            }
            if (mapped) {
                const double product = expected;
                expected = float(initial[r * cols + c]);
                // Every launch combines with the current C, rounding to TC.
                for (int launch = 0; launch < (asynchronous ? 2 : 1); ++launch)
                    expected = float(TC(0.25 * expected + 0.5 * product));
            }
            float actual = float(output[r * cols + c]);
            if (!std::isfinite(actual) || actual != float(TC(expected))) {
                fprintf(stderr, "%s (%s, cols=%u) at (%u,%u): %g != %g\n",
                    label, asynchronous ? "async" : "sync", cols, r, c, actual,
                    float(TC(expected)));
                ok = false;
            }
        }
    }
    for (uint32_t i = rows * cols; i < output.size(); ++i)
        if (float(output[i]) != -1000.0f)
            ok = false;

    MUST(cudaFree(a.elems));
    MUST(cudaFree(a.col_ind));
    MUST(cudaFree(a.row_off));
    MUST(cudaFree(dorder));
    MUST(cudaFree(db));
    MUST(cudaFree(dc));
    return ok;
}

static void mapped_sync(uint32_t rows, uint32_t shared, uint32_t cols,
    Kuiper_Sparse_Matrix_csr_matrix__half a, uint32_t *order, float *b, half *c)
{
    Klas_SPMM_g_spmm_mapped_f16_f32_f16_f32_32x32x4(
        0.5f, 0.25f, rows, shared, cols, a, order, b, c);
}

static void mapped_async(uint32_t rows, uint32_t shared, uint32_t cols,
    Kuiper_Sparse_Matrix_csr_matrix__half a, uint32_t *order, float *b, half *c,
    cudaStream_t stream)
{
    Klas_SPMM_g_spmm_mapped_f16_f32_f16_f32_32x32x4_on(
        0.5f, 0.25f, rows, shared, cols, a, order, b, c, stream);
}

int main()
{
    bool ok = true;
    int count = 0;
#define CHECK_VARIANT(types, width, wide)                                      \
    for (bool asynchronous : {false, true}) {                                  \
        ok = run_case(#types, width, asynchronous,                             \
                 Klas_SPMM_g_spmm_##types##_32x32x4,                           \
                 Klas_SPMM_g_spmm_##types##_32x32x4_on, false, wide) &&        \
             ok;                                                               \
        ++count;                                                               \
    }
    for (uint32_t cols : {8u, 16u, 32u, 40u}) {
        CHECK_VARIANT(f16_f16_f32_f32, cols, false);
        CHECK_VARIANT(bf16_bf16_f32_f32, cols, false);
        CHECK_VARIANT(f16_f32_f16_f32, cols, false);
        CHECK_VARIANT(f16_bf16_f32_f64, cols, true);
        CHECK_VARIANT(f16_f32_f32_f64, cols, true);
        for (bool asynchronous : {false, true}) {
            ok = run_case("mapped GEMM", cols, asynchronous, mapped_sync,
                     mapped_async, true) &&
                 ok;
            ++count;
        }
    }
    // Partial final tiles containing complete eight-element working groups.
    for (uint32_t cols : {24u, 48u, 56u}) {
        CHECK_VARIANT(f16_f16_f32_f32, cols, false);
        CHECK_VARIANT(bf16_bf16_f32_f32, cols, false);
        CHECK_VARIANT(f16_bf16_f32_f64, cols, true);
        CHECK_VARIANT(f16_f32_f32_f64, cols, true);
    }
    // The float-B/float-C control retains its four-element N alignment.
    for (uint32_t cols : {4u, 12u, 36u}) {
        CHECK_VARIANT(f16_f32_f32_f64, cols, true);
    }
#undef CHECK_VARIANT
    if (!ok)
        return 1;
    printf("All %d mixed-precision SPMM tests passed\n", count);
    return 0;
}
