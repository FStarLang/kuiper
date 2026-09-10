#include "test-common.h"

// As in the WMMA example test, these inline device entry points are extracted
// from Pulse. The driver supplies the unmodelled collective execution setup.
#include "Kuiper_Example_TensorCore_WGMMA.cu"

__global__ void wgmma_test(const __nv_bfloat16 *a, const __nv_bfloat16 *b,
                           float *c, bool accumulate)
{
    // Two independent warpgroups, with a multidimensional block, exercise
    // both the accumulator lane mapping and operand isolation.
    __shared__ __align__(16) __nv_bfloat16 sa[2][1024];
    __shared__ __align__(16) __nv_bfloat16 sb[2][128];
    unsigned t = threadIdx.x + blockDim.x * threadIdx.y;
    unsigned wg = t / 128;
    unsigned lane = t % 128;
    for (unsigned i = lane; i < 1024; i += 128) {
        unsigned row = i / 16, k = i % 16;
        unsigned packed = row / 8 * 128 + k / 8 * 64 + row % 8 * 8 + k % 8;
        sa[wg][packed] = a[wg * 1024 + i];
    }
    unsigned k = lane / 8, col = lane % 8;
    sb[wg][k / 8 * 64 + col * 8 + k % 8] = b[wg * 128 + lane];
    // Publish generic-proxy writes to WGMMA's async proxy, then wait for
    // every producer. These are runtime obligations, outside the Pulse API.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (accumulate)
        Kuiper_Example_TensorCore_WGMMA_accumulate_twice(sa[wg], sb[wg],
                                                         c + wg * 512);
    else
        Kuiper_Example_TensorCore_WGMMA_multiply(sa[wg], sb[wg], c + wg * 512);
}

int main()
{
    __nv_bfloat16 a[2 * 1024], b[2 * 128];
    float initial[2 * 512], result[2 * 512];
    // Small, exactly representable integers give an exact oracle without
    // assuming an undocumented Tensor Core rounding/accumulation order.
    for (int i = 0; i < 2 * 1024; ++i)
        a[i] = __float2bfloat16(float((i * 7 + i / 16) % 11 - 5));
    for (int i = 0; i < 2 * 128; ++i)
        b[i] = __float2bfloat16(float((i * 3 + i / 8) % 7 - 3));
    for (int i = 0; i < 2 * 512; ++i)
        initial[i] = float(i % 13 - 6);

    __nv_bfloat16 *da, *db;
    float *dc;
    MUST(cudaMalloc(&da, sizeof(a)));
    MUST(cudaMalloc(&db, sizeof(b)));
    MUST(cudaMalloc(&dc, sizeof(initial)));
    MUST(cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(db, b, sizeof(b), cudaMemcpyHostToDevice));
    for (int accumulate = 0; accumulate < 2; ++accumulate) {
        MUST(cudaMemcpy(dc, initial, sizeof(initial), cudaMemcpyHostToDevice));
        wgmma_test<<<1, dim3(32, 8)>>>(da, db, dc, accumulate != 0);
        MUST(cudaGetLastError());
        MUST(cudaMemcpy(result, dc, sizeof(result), cudaMemcpyDeviceToHost));
        for (int wg = 0; wg < 2; ++wg) {
            for (int row = 0; row < 64; ++row) {
                for (int col = 0; col < 8; ++col) {
                    float product = 0;
                    for (int k = 0; k < 16; ++k)
                        product +=
                            __bfloat162float(a[wg * 1024 + row * 16 + k]) *
                            __bfloat162float(b[wg * 128 + k * 8 + col]);
                    int i = wg * 512 + row * 8 + col;
                    float expected =
                        accumulate ? initial[i] + 2 * product : product;
                    if (result[i] != expected) {
                        fprintf(stderr,
                                "WGMMA mismatch: accumulate=%d wg=%d (%d,%d): "
                                "%g != %g\n",
                                accumulate, wg, row, col, result[i], expected);
                        return 1;
                    }
                }
            }
        }
    }
    MUST(cudaFree(da));
    MUST(cudaFree(db));
    MUST(cudaFree(dc));
    puts("WGMMA BF16/F32 64x8x16: OK");
    return 0;
}
