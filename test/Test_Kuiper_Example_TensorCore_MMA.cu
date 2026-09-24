#include "test-common.h"
#include <cstring>

#include "Kuiper_Example_TensorCore_MMA.cu"

static constexpr int A_SIZE = 1024;
static constexpr int B_SIZE = 512;
static constexpr int C_SIZE = 512;

__global__ void mma_test(
    __nv_bfloat16 *a, __nv_bfloat16 *b, float *c, bool strided)
{
    unsigned t =
        threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    unsigned warp = t / 32;
    if (strided)
        Kuiper_Example_TensorCore_MMA_accumulate_twice(
            a + warp * A_SIZE, b + warp * B_SIZE, c + warp * C_SIZE);
    else
        Kuiper_Example_TensorCore_MMA_multiply(
            a + warp * A_SIZE, b + warp * B_SIZE, c + warp * C_SIZE);
}

// Independent element-index formulation of the documented PTX fragment map.
// Do not use the implementation's packing, lane, or load/store helpers here.
__global__ void mma_reference(
    const __nv_bfloat16 *a, const __nv_bfloat16 *b, float *c, bool strided)
{
    unsigned t =
        threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    unsigned lane = t % 32, group = lane / 4, thread = lane % 4;
    a += (t / 32) * A_SIZE + (strided ? 528 : 0);
    b += (t / 32) * B_SIZE + (strided ? 272 : 0);
    c += (t / 32) * C_SIZE + (strided ? 264 : 0);
    unsigned lda = strided ? 32 : 16;
    unsigned ldb = strided ? 32 : 16;
    unsigned ldc = strided ? 16 : 8;
    uint32_t ar[4] = {}, br[2] = {};
    float d[4];
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        unsigned row = group + ((i % 4 >= 2) ? 8 : 0);
        unsigned col = 2 * thread + i % 2 + ((i >= 4) ? 8 : 0);
        ar[i / 2] |= uint32_t(__bfloat16_as_ushort(a[row * lda + col]))
                     << (16 * (i % 2));
    }
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        unsigned row = 2 * thread + i % 2 + ((i >= 2) ? 8 : 0);
        br[i / 2] |= uint32_t(__bfloat16_as_ushort(b[group * ldb + row]))
                     << (16 * (i % 2));
        unsigned cr = group + ((i >= 2) ? 8 : 0);
        unsigned cc = 2 * thread + i % 2;
        d[i] = strided ? c[cr * ldc + cc] : 0.0f;
    }
    for (int step = 0; step < (strided ? 2 : 1); ++step) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(ar[0]), "r"(ar[1]), "r"(ar[2]), "r"(ar[3]), "r"(br[0]),
            "r"(br[1]));
    }
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        unsigned row = group + ((i >= 2) ? 8 : 0);
        unsigned col = 2 * thread + i % 2;
        c[row * ldc + col] = d[i];
    }
}

static uint32_t bits(float value)
{
    uint32_t result;
    memcpy(&result, &value, sizeof result);
    return result;
}

static __nv_bfloat16 bf16_bits(uint16_t value)
{
    __nv_bfloat16 result;
    static_assert(sizeof result == sizeof value);
    memcpy(&result, &value, sizeof result);
    return result;
}

int main()
{
    int device;
    cudaDeviceProp properties;
    MUST(cudaGetDevice(&device));
    MUST(cudaGetDeviceProperties(&properties, device));
    if (properties.major < 8) {
        fprintf(stderr, "MMA BF16/F32 requires sm_80 or newer.\n");
        return 1;
    }

    __nv_bfloat16 a[2 * A_SIZE], b[2 * B_SIZE];
    float initial[2 * C_SIZE], result[2 * C_SIZE], reference[2 * C_SIZE];
    __nv_bfloat16 *da, *db;
    float *dc, *dr;
    MUST(cudaMalloc(&da, sizeof a));
    MUST(cudaMalloc(&db, sizeof b));
    MUST(cudaMalloc(&dc, sizeof result));
    MUST(cudaMalloc(&dr, sizeof reference));
    const uint16_t finite[] = {0x3f80, 0xbf00, 0x3f81, 0x3eab, 0xc120, 0x0080,
        0x0001, 0x8001, 0x0000, 0x8000, 0x4001, 0xc001};
    const uint16_t special[] = {
        0x3f80, 0xbf80, 0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc1, 0xffc2};
    for (int mode = 0; mode < 3; ++mode) {
        for (int i = 0; i < 2 * A_SIZE; ++i) {
            a[i] = mode == 0
                       ? __float2bfloat16(float((i * 7 + i / 16) % 11 - 5))
                       : bf16_bits(mode == 1 ? finite[(i * 7 + i / 16) % 12]
                                             : special[(i * 3 + i / 16) % 8]);
        }
        for (int i = 0; i < 2 * B_SIZE; ++i) {
            b[i] = mode == 0
                       ? __float2bfloat16(float((i * 3 + i / 8) % 7 - 3))
                       : bf16_bits(mode == 1 ? finite[(i * 5 + i / 8) % 12]
                                             : special[(i * 5 + i / 8) % 8]);
        }
        for (int i = 0; i < 2 * C_SIZE; ++i)
            initial[i] = float(i % 13 - 6);
        MUST(cudaMemcpy(da, a, sizeof a, cudaMemcpyHostToDevice));
        MUST(cudaMemcpy(db, b, sizeof b, cudaMemcpyHostToDevice));
        for (int strided = 0; strided < 2; ++strided) {
            MUST(cudaMemcpy(
                dc, initial, sizeof initial, cudaMemcpyHostToDevice));
            MUST(cudaMemcpy(
                dr, initial, sizeof initial, cudaMemcpyHostToDevice));
            // Two independent warps in a three-dimensional block.
            mma_test<<<1, dim3(16, 2, 2)>>>(da, db, dc, strided != 0);
            MUST(cudaGetLastError());
            mma_reference<<<1, dim3(16, 2, 2)>>>(da, db, dr, strided != 0);
            MUST(cudaGetLastError());
            MUST(cudaMemcpy(result, dc, sizeof result, cudaMemcpyDeviceToHost));
            MUST(cudaMemcpy(
                reference, dr, sizeof reference, cudaMemcpyDeviceToHost));
            for (int i = 0; i < 2 * C_SIZE; ++i) {
                if (bits(result[i]) != bits(reference[i])) {
                    fprintf(stderr,
                        "MMA mismatch: mode=%d strided=%d index=%d "
                        "got=0x%08x expected=0x%08x\n",
                        mode, strided, i, bits(result[i]), bits(reference[i]));
                    return 1;
                }
            }
            // Small integers also admit an exact oracle independent of PTX.
            if (mode == 0) {
                for (int warp = 0; warp < 2; ++warp) {
                    for (int row = 0; row < 16; ++row) {
                        for (int col = 0; col < 8; ++col) {
                            float product = 0;
                            for (int k = 0; k < 16; ++k) {
                                int ai = warp * A_SIZE + (strided ? 528 : 0) +
                                         row * (strided ? 32 : 16) + k;
                                int bi = warp * B_SIZE + (strided ? 272 : 0) +
                                         col * (strided ? 32 : 16) + k;
                                product += __bfloat162float(a[ai]) *
                                           __bfloat162float(b[bi]);
                            }
                            int i = warp * C_SIZE + (strided ? 264 : 0) +
                                    row * (strided ? 16 : 8) + col;
                            float expected =
                                strided ? initial[i] + 2 * product : product;
                            if (result[i] != expected) {
                                fprintf(stderr,
                                    "MMA integer oracle mismatch at %d\n", i);
                                return 1;
                            }
                        }
                    }
                }
            }
        }
    }
    MUST(cudaFree(da));
    MUST(cudaFree(db));
    MUST(cudaFree(dc));
    MUST(cudaFree(dr));
    puts("MMA BF16/F32 16x8x16: OK");
    return 0;
}
