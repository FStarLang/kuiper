#include "Kuiper_Example_Float32GPU.h"
#include <cmath>
#include <cstring>

#define GPU_API(name) Kuiper_Example_Float32GPU_##name

static float from_bits(uint32_t bits)
{
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static uint32_t to_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    return bits;
}

static void check(const char *name, float got, float expected)
{
    if (to_bits(got) != to_bits(expected) &&
        !(std::isnan(got) && std::isnan(expected))) {
        fprintf(stderr, "%s: got 0x%08x, expected 0x%08x\n", name, to_bits(got),
            to_bits(expected));
        exit(1);
    }
}

__global__ void reference(float x, float y, float *out)
{
    float product, exponent;
    asm volatile("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(product) : "f"(x), "f"(y));
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(exponent) : "f"(x));
    out[0] = product;
    out[1] = exponent;
}

int main()
{
    const uint32_t pairs[][2] = {
        {0, 0x3f800000},
        {0x80000000, 0x3f800000},
        {0x3f800000, 0xbf800000},
        {0x3fc00000, 0xc0000000},
        {1, 0x7f000000},
        {0x80000001, 0x7f000000},
        {0x3f800000, 1},
        {0x3f800000, 0x80000001},
        {0x007fffff, 0x40000000},
        {0x807fffff, 0x40000000},
        {0x00800000, 0x3f000000},
        {0x80800000, 0x3f000000},
        {0x00800000, 0x40000000},
        {0x7f7fffff, 0x40000000},
        {0xff7fffff, 0x40000000},
        {0x3eaaaaab, 0x3faaaaab},
        {0x43000000, 0x3f800000},
        {0xc2fe0000, 0x3f800000},
        {0x7f800000, 0x3f800000},
        {0xff800000, 0x3f800000},
        {0, 0x7f800000},
        {0x7fc00001, 0x3f800000},
        {0x7f800001, 0x3f800000},
        {0x3f800000, 0x7fc00001},
    };
    float *device;
    float expected[2];
    MUST(cudaMalloc(&device, sizeof expected));
    for (const auto &pair : pairs) {
        float x = from_bits(pair[0]), y = from_bits(pair[1]);
        reference<<<1, 1>>>(x, y, device);
        MUST(cudaGetLastError());
        MUST(cudaMemcpy(
            expected, device, sizeof expected, cudaMemcpyDeviceToHost));
        check("mul.rn.ftz", GPU_API(multiply)(x, y), expected[0]);
        check("ex2.approx.ftz", GPU_API(exponentiate)(x), expected[1]);
    }
    MUST(cudaFree(device));

    check("normal multiply", GPU_API(multiply)(1.5f, -2.0f), -3.0f);
    check(
        "positive input FTZ", GPU_API(multiply)(from_bits(1), 0x1p127f), 0.0f);
    check("negative input FTZ",
        GPU_API(multiply)(from_bits(0x80000001), 0x1p127f), -0.0f);
    check("positive output FTZ", GPU_API(multiply)(0x1p-126f, 0.5f), 0.0f);
    check("negative output FTZ", GPU_API(multiply)(-0x1p-126f, 0.5f), -0.0f);
    check("base-two exponent", GPU_API(exponentiate)(1.0f), 2.0f);
    check("exponent output FTZ", GPU_API(exponentiate)(-127.0f), 0.0f);
    puts("Float32 GPU checks passed.");
}
