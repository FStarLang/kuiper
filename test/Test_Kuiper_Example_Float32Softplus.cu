#include "float_test_common.c.inc"
#include <cmath>
#include <vector>
#include "Kuiper_Example_Float32Softplus.h"

using float_test::from_bits;
using float_test::to_bits;

__global__ void reference(
    const float *inputs, const __nv_bfloat16 *pairs, float *outputs, uint32_t n)
{
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
        i += blockDim.x * gridDim.x) {
        float x = inputs[i];
        float a = __fadd_rn(
            __bfloat162float(pairs[2 * i]), __bfloat162float(pairs[2 * i + 1]));
        outputs[4 * i] = x > 20.0f ? x : log1pf(expf(x));
        outputs[4 * i + 1] = a;
        outputs[4 * i + 2] = a > 20.0f ? 1.0f : 0.0f;
        outputs[4 * i + 3] = a > 20.0f ? a : log1pf(expf(a));
    }
}

int main()
{
    constexpr uint32_t n = 4096;
    std::mt19937 random(0);
    std::vector<float> inputs(n);
    std::vector<__nv_bfloat16> pairs(2 * n);
    for (float &value : inputs)
        value = float_test::random_float32(random);
    for (__nv_bfloat16 &value : pairs) {
        __nv_bfloat16_raw raw;
        raw.x = static_cast<uint16_t>(random());
        value = raw;
    }
    const float controls[] = {
        std::nextafter(20.0f, 0.0f),
        20.0f,
        std::nextafter(20.0f, 21.0f),
        -87.33984375f,
        -100.0f,
        100.0f,
        -0.0f,
        0.0f,
        from_bits(0x7f800000),
        from_bits(0xff800000),
        from_bits(0x7fc12345),
        from_bits(0x7f812345),
        from_bits(1),
        from_bits(0x80000001),
        from_bits(0x7f7fffff),
        from_bits(0xff7fffff),
    };
    for (size_t i = 0; i < sizeof(controls) / sizeof(controls[0]); ++i) {
        inputs[i] = controls[i];
        pairs[2 * i] = __float2bfloat16(controls[i]);
        pairs[2 * i + 1] = __float2bfloat16(0.0f);
    }
    const float biases[] = {
        -0x1p-24f,
        0.0f,
        0x1p-24f,
        0x1p-20f,
        0x1p-19f,
        -0x1p-19f,
    };
    for (size_t i = 0; i < sizeof(biases) / sizeof(biases[0]); ++i) {
        pairs[2 * i] = __float2bfloat16(20.0f);
        pairs[2 * i + 1] = __float2bfloat16(biases[i]);
    }
    constexpr uint32_t guard = 0x4abcd123;
    std::vector<float> expected(4 * n), actual(4 * n + 2, from_bits(guard));
    std::vector<float> preserved_inputs(n);
    std::vector<__nv_bfloat16> preserved_pairs(2 * n);
    float *device_inputs, *device_expected, *device_actual;
    __nv_bfloat16 *device_pairs;
    MUST(cudaMalloc(&device_inputs, inputs.size() * sizeof(float)));
    MUST(cudaMalloc(&device_pairs, pairs.size() * sizeof(__nv_bfloat16)));
    MUST(cudaMalloc(&device_expected, expected.size() * sizeof(float)));
    MUST(cudaMalloc(&device_actual, actual.size() * sizeof(float)));
    MUST(cudaMemcpy(device_inputs, inputs.data(), inputs.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(device_pairs, pairs.data(),
        pairs.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(device_actual, actual.data(), actual.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    reference<<<32, 128>>>(device_inputs, device_pairs, device_expected, n);
    MUST(cudaGetLastError());
    MUST(cudaDeviceSynchronize());
    Kuiper_Example_Float32Softplus_run(
        n, device_inputs, device_pairs, device_actual + 1);
    MUST(cudaGetLastError());
    MUST(cudaMemcpy(expected.data(), device_expected,
        expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(actual.data(), device_actual, actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(preserved_inputs.data(), device_inputs,
        inputs.size() * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(preserved_pairs.data(), device_pairs,
        pairs.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < 4 * n; ++i) {
        if (to_bits(actual[i + 1]) != to_bits(expected[i])) {
            fprintf(stderr, "case %u op %u: got %08x, expected %08x\n", i / 4,
                i % 4, to_bits(actual[i + 1]), to_bits(expected[i]));
            return 1;
        }
    }
    if (to_bits(actual.front()) != guard || to_bits(actual.back()) != guard ||
        memcmp(inputs.data(), preserved_inputs.data(),
            inputs.size() * sizeof(float)) ||
        memcmp(pairs.data(), preserved_pairs.data(),
            pairs.size() * sizeof(__nv_bfloat16))) {
        fprintf(stderr, "input preservation or output guards failed\n");
        return 1;
    }
    // The exact sums exceed 20, but round-to-even selects the nonlinear arm.
    for (uint32_t i : {2u, 3u}) {
        double exact = 20.0 + double(biases[i]);
        if (!(exact > 20.0) || actual[4 * i + 2] != 20.0f ||
            actual[4 * i + 3] != 0.0f || actual[4 * i + 4] != 20.0f) {
            fprintf(stderr, "missing rounded-threshold witness %u\n", i);
            return 1;
        }
    }
    if (std::fpclassify(actual[4 * 3 + 1]) != FP_SUBNORMAL) {
        fprintf(stderr, "ordinary exponential subnormal was lost\n");
        return 1;
    }
    MUST(cudaFree(device_actual));
    MUST(cudaFree(device_expected));
    MUST(cudaFree(device_pairs));
    MUST(cudaFree(device_inputs));
    puts("Float32 softplus checks passed.");
}
