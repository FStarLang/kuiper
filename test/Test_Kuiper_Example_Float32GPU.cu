#include "Kuiper_Example_Float32GPU.h"
#include <cmath>
#include <cstring>
#include <vector>

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

static void check_bits(const char *name, float got, float expected)
{
    if (to_bits(got) != to_bits(expected)) {
        fprintf(stderr, "%s: got 0x%08x, expected 0x%08x\n", name, to_bits(got),
            to_bits(expected));
        exit(1);
    }
}

static void check(const char *name, float got, float expected)
{
    if (!(std::isnan(got) && std::isnan(expected))) {
        check_bits(name, got, expected);
    }
}

__global__ void reference(float x, float y, float *out)
{
    float product, exponent, reciprocal, inverse_root;
    asm volatile("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(product) : "f"(x), "f"(y));
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(exponent) : "f"(x));
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(reciprocal) : "f"(x));
    asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inverse_root) : "f"(x));
    out[0] = product;
    out[1] = exponent;
    out[2] = reciprocal;
    out[3] = inverse_root;
}

__global__ void arithmetic_reference(
    const float *inputs, float *outputs, uint32_t n)
{
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
        i += blockDim.x * gridDim.x) {
        float x = inputs[3 * i], y = inputs[3 * i + 1], z = inputs[3 * i + 2];
        float sum, fused, fx, fy, fz, composed;
        asm volatile("add.rn.ftz.f32 %0, %1, %2;" : "=f"(sum) : "f"(x), "f"(y));
        asm volatile("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(fused)
            : "f"(x), "f"(y), "f"(z));
        asm volatile("mul.rn.ftz.f32 %0, %1, 0f3f800000;" : "=f"(fx) : "f"(x));
        asm volatile("mul.rn.ftz.f32 %0, %1, 0f3f800000;" : "=f"(fy) : "f"(y));
        asm volatile("mul.rn.ftz.f32 %0, %1, 0f3f800000;" : "=f"(fz) : "f"(z));
        float rounded = __fmaf_rn(fx, fy, fz);
        asm volatile("mul.rn.ftz.f32 %0, %1, 0f3f800000;"
            : "=f"(composed)
            : "f"(rounded));
        outputs[5 * i] = sum;
        outputs[5 * i + 1] = fused;
        outputs[5 * i + 2] = __fadd_rn(x, y);
        outputs[5 * i + 3] = __fmaf_rn(x, y, z);
        outputs[5 * i + 4] = composed;
    }
}

static void check_arithmetic()
{
    const uint32_t special[] = {
        0,
        0x80000000,
        1,
        0x80000001,
        0x007fffff,
        0x807fffff,
        0x00800000,
        0x80800000,
        0x00800001,
        0x80800001,
        0x3f7fffff,
        0xbf7fffff,
        0x3f800000,
        0xbf800000,
        0x3f800001,
        0xbf800001,
        0x3eaaaaab,
        0x7f7fffff,
        0xff7fffff,
        0x7f800000,
        0xff800000,
        0x7fc00001,
        0xffc12345,
        0x7f800001,
        0xff800001,
    };
    // A non-FTZ FMA rounded before flushing produces the smallest normal.
    std::vector<float> inputs = {
        from_bits(0x3f7fffff), from_bits(0x00800000), 0.0f};
    for (uint32_t x : special)
        for (uint32_t y : special)
            for (uint32_t z : special) {
                inputs.push_back(from_bits(x));
                inputs.push_back(from_bits(y));
                inputs.push_back(from_bits(z));
            }
    uint32_t state = 0x2468ace1;
    for (uint32_t i = 0; i < 131072 * 3; ++i) {
        state = state * 1664525u + 1013904223u;
        inputs.push_back(from_bits(state));
    }
    uint32_t n = static_cast<uint32_t>(inputs.size() / 3);
    std::vector<float> expected(5 * n),
        actual(2 * n + 2, from_bits(0x7fc12345));
    std::vector<float> preserved(inputs.size());
    float *device_inputs, *device_expected, *device_actual;
    MUST(cudaMalloc(&device_inputs, inputs.size() * sizeof(float)));
    MUST(cudaMalloc(&device_expected, expected.size() * sizeof(float)));
    MUST(cudaMalloc(&device_actual, actual.size() * sizeof(float)));
    MUST(cudaMemcpy(device_inputs, inputs.data(), inputs.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(device_actual, actual.data(), actual.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    arithmetic_reference<<<128, 128>>>(device_inputs, device_expected, n);
    MUST(cudaGetLastError());
    GPU_API(arithmetic)(n, device_inputs, device_actual + 1);
    MUST(cudaMemcpy(expected.data(), device_expected,
        expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(actual.data(), device_actual, actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(preserved.data(), device_inputs,
        inputs.size() * sizeof(float), cudaMemcpyDeviceToHost));
    unsigned different[3] = {};
    for (uint32_t i = 0; i < n; ++i) {
        check_bits("add.rn.ftz", actual[2 * i + 1], expected[5 * i]);
        check_bits("fma.rn.ftz", actual[2 * i + 2], expected[5 * i + 1]);
        different[0] +=
            to_bits(expected[5 * i]) != to_bits(expected[5 * i + 2]);
        different[1] +=
            to_bits(expected[5 * i + 1]) != to_bits(expected[5 * i + 3]);
        different[2] +=
            to_bits(expected[5 * i + 1]) != to_bits(expected[5 * i + 4]);
    }
    check_bits("FTZ FMA rounds at the underflow boundary", actual[2], 0.0f);
    check_bits("rounded-before-flushing negative control", expected[4],
        from_bits(0x00800000));
    check_bits("leading output guard", actual.front(), from_bits(0x7fc12345));
    check_bits("trailing output guard", actual.back(), from_bits(0x7fc12345));
    if (!different[0] || !different[1] || !different[2] ||
        memcmp(
            inputs.data(), preserved.data(), inputs.size() * sizeof(float))) {
        fprintf(stderr, "missing FTZ-sensitive coverage or modified inputs\n");
        exit(1);
    }
    MUST(cudaFree(device_actual));
    MUST(cudaFree(device_expected));
    MUST(cudaFree(device_inputs));
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
        {0x40800000, 0x3f800000},
        {0x3f7fffff, 0x3f800000},
        {0x3f800001, 0x3f800000},
        {0x00800001, 0x3f800000},
        {0x7e7fffff, 0x3f800000},
        {0x7e800000, 0x3f800000},
        {0x7e800001, 0x3f800000},
        {0x7f000000, 0x3f800000},
        {0xffc12345, 0x3f800000},
        {0xff800001, 0x3f800000},
    };
    float *device;
    float expected[4];
    MUST(cudaMalloc(&device, sizeof expected));
    for (const auto &pair : pairs) {
        float x = from_bits(pair[0]), y = from_bits(pair[1]);
        reference<<<1, 1>>>(x, y, device);
        MUST(cudaGetLastError());
        MUST(cudaMemcpy(
            expected, device, sizeof expected, cudaMemcpyDeviceToHost));
        check("mul.rn.ftz", GPU_API(multiply)(x, y), expected[0]);
        check("ex2.approx.ftz", GPU_API(exponentiate)(x), expected[1]);
        check_bits("rcp.approx.ftz", GPU_API(reciprocal)(x), expected[2]);
        check_bits("rsqrt.approx.ftz", GPU_API(inverse_root)(x), expected[3]);
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
    check_bits("reciprocal", GPU_API(reciprocal)(4.0f), 0.25f);
    check_bits("reciprocal +zero", GPU_API(reciprocal)(0.0f), INFINITY);
    check_bits("reciprocal -zero", GPU_API(reciprocal)(-0.0f), -INFINITY);
    check_bits("reciprocal positive input FTZ",
        GPU_API(reciprocal)(from_bits(1)), INFINITY);
    check_bits("reciprocal negative input FTZ",
        GPU_API(reciprocal)(from_bits(0x80000001)), -INFINITY);
    check_bits(
        "reciprocal positive output FTZ", GPU_API(reciprocal)(0x1p127f), 0.0f);
    check_bits("reciprocal negative output FTZ", GPU_API(reciprocal)(-0x1p127f),
        -0.0f);
    check_bits("reciprocal +infinity", GPU_API(reciprocal)(INFINITY), 0.0f);
    check_bits("reciprocal -infinity", GPU_API(reciprocal)(-INFINITY), -0.0f);
    check_bits("inverse square root", GPU_API(inverse_root)(4.0f), 0.5f);
    check_bits("inverse root +zero", GPU_API(inverse_root)(0.0f), INFINITY);
    check_bits("inverse root -zero", GPU_API(inverse_root)(-0.0f), -INFINITY);
    check_bits("inverse root positive input FTZ",
        GPU_API(inverse_root)(from_bits(1)), INFINITY);
    check_bits("inverse root negative input FTZ",
        GPU_API(inverse_root)(from_bits(0x80000001)), -INFINITY);
    check_bits("inverse root +infinity", GPU_API(inverse_root)(INFINITY), 0.0f);
    check("inverse root negative input", GPU_API(inverse_root)(-1.0f), NAN);
    check("inverse root -infinity", GPU_API(inverse_root)(-INFINITY), NAN);
    check_arithmetic();
    puts("Float32 GPU checks passed.");
}
