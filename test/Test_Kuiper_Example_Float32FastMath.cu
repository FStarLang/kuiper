#include "float_test_common.c.inc"
#include <cmath>
#include <vector>
#include "Kuiper_Example_Float32FastMath.h"

using float_test::to_bits;

__global__ void reference(const float *inputs, float *outputs, uint32_t n)
{
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
        i += blockDim.x * gridDim.x) {
        float x = inputs[3 * i], y = inputs[3 * i + 1], z = inputs[3 * i + 2];
        outputs[7 * i] = __expf(x);
        outputs[7 * i + 1] = __fdividef(x, y);
        outputs[7 * i + 2] = __fmaf_rn(x, y, z);
        outputs[7 * i + 3] = __fsub_rn(x, y);
        outputs[7 * i + 4] = expf(x);
        outputs[7 * i + 5] = x / y;
        outputs[7 * i + 6] = __fadd_rn(__fmul_rn(x, y), z);
    }
}

int main()
{
    constexpr uint32_t n = 50000;
    std::mt19937 random(0);
    std::vector<float> inputs(3 * n);
    for (float &value : inputs)
        value = float_test::random_float32(random);
    // Keep a subnormal exponential and a fused-vs-separate rounding witness.
    inputs[0] = -87.33984375f;
    inputs[1] = 1.0f;
    inputs[2] = 0.0f;
    inputs[3] = std::nextafter(1.0f, 2.0f);
    inputs[4] = std::nextafter(1.0f, 0.0f);
    inputs[5] = -1.0f;
    std::vector<float> expected(7 * n), actual(4 * n);
    float *device_inputs, *device_expected, *device_actual;
    MUST(cudaMalloc(&device_inputs, inputs.size() * sizeof(float)));
    MUST(cudaMalloc(&device_expected, expected.size() * sizeof(float)));
    MUST(cudaMalloc(&device_actual, actual.size() * sizeof(float)));
    MUST(cudaMemcpy(device_inputs, inputs.data(), inputs.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    reference<<<128, 128>>>(device_inputs, device_expected, n);
    MUST(cudaGetLastError());
    Kuiper_Example_Float32FastMath_run(n, device_inputs, device_actual);
    MUST(cudaMemcpy(expected.data(), device_expected,
        expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(actual.data(), device_actual, actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    unsigned different[3] = {};
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t op = 0; op < 4; ++op) {
            if (to_bits(actual[4 * i + op]) != to_bits(expected[7 * i + op])) {
                fprintf(stderr, "case %u op %u: got %08x, expected %08x\n", i,
                    op, to_bits(actual[4 * i + op]),
                    to_bits(expected[7 * i + op]));
                return 1;
            }
        }
        for (uint32_t op = 0; op < 3; ++op)
            different[op] += to_bits(expected[7 * i + op]) !=
                             to_bits(expected[7 * i + 4 + op]);
    }
    if (std::fpclassify(actual[0]) != FP_SUBNORMAL || !different[0] ||
        !different[1] || !different[2]) {
        fprintf(
            stderr, "missing subnormal or replacement-sensitive coverage\n");
        return 1;
    }
    MUST(cudaFree(device_actual));
    MUST(cudaFree(device_expected));
    MUST(cudaFree(device_inputs));
    puts("Float32 fast-math checks passed.");
}
