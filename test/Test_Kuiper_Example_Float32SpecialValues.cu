#include "float_test_common.c.inc"
#include <cmath>
#include <limits>
#include <math_constants.h>
#include <vector>
#include "Kuiper_Example_Float32SpecialValues.h"

using float_test::from_bits;
using float_test::to_bits;

__global__ void reference(const float *inputs, float *outputs, uint32_t n)
{
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
        i += blockDim.x * gridDim.x) {
        float x = inputs[i];
        float shifted = __fsub_rn(-CUDART_INF_F, x);
        outputs[5 * i] = expf(x);
        outputs[5 * i + 1] = shifted;
        outputs[5 * i + 2] = expf(shifted);
        outputs[5 * i + 3] = __fsub_rn(x, 1.0f);
        outputs[5 * i + 4] = __expf(x);
    }
}

int main()
{
    std::vector<float> inputs = {std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(), -87.33984375f};
    for (uint32_t bits = 0; bits < 65536; ++bits)
        inputs.push_back(from_bits(bits << 16));
    std::mt19937 random(0);
    for (unsigned i = 0; i < 50000; ++i)
        inputs.push_back(float_test::random_float32(random));
    const uint32_t n = static_cast<uint32_t>(inputs.size());
    constexpr uint32_t guard = 0x4b123456;
    constexpr float infinity = std::numeric_limits<float>::infinity();
    std::vector<float> expected(5 * n), actual(4 * n + 2, from_bits(guard));
    std::vector<float> preserved(n);
    float *device_inputs, *device_expected, *device_actual;
    MUST(cudaMalloc(&device_inputs, inputs.size() * sizeof(float)));
    MUST(cudaMalloc(&device_expected, expected.size() * sizeof(float)));
    MUST(cudaMalloc(&device_actual, actual.size() * sizeof(float)));
    MUST(cudaMemcpy(device_inputs, inputs.data(), inputs.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(device_actual, actual.data(), actual.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    reference<<<128, 128>>>(device_inputs, device_expected, n);
    MUST(cudaGetLastError());
    Kuiper_Example_Float32SpecialValues_run(
        n, device_inputs, device_actual + 1);
    MUST(cudaMemcpy(expected.data(), device_expected,
        expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(actual.data(), device_actual, actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(preserved.data(), device_inputs,
        preserved.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (to_bits(actual.front()) != guard || to_bits(actual.back()) != guard ||
        memcmp(
            inputs.data(), preserved.data(), inputs.size() * sizeof(float))) {
        fprintf(stderr, "output guard or input preservation failure\n");
        return 1;
    }
    unsigned finite = 0, positive_infinite = 0, negative_infinite = 0, nan = 0;
    unsigned different_from_fast = 0;
    for (uint32_t i = 0; i < n; ++i) {
        float x = inputs[i], shifted = actual[4 * i + 2];
        float exponential = actual[4 * i + 1], padded = actual[4 * i + 3];
        // Compare extraction with the native oracle, without specifying NaN
        // payloads in the library contracts.
        for (uint32_t op = 0; op < 4; ++op) {
            if (to_bits(actual[4 * i + op + 1]) !=
                to_bits(expected[5 * i + op])) {
                fprintf(stderr, "case %u op %u: got %08x, expected %08x\n", i,
                    op, to_bits(actual[4 * i + op + 1]),
                    to_bits(expected[5 * i + op]));
                return 1;
            }
        }
        if (std::isfinite(x) || x == infinity) {
            finite += std::isfinite(x);
            positive_infinite += x == infinity;
            if (to_bits(shifted) != 0xff800000 || to_bits(padded) != 0) {
                fprintf(
                    stderr, "nonzero padding for maximum %08x\n", to_bits(x));
                return 1;
            }
        } else {
            negative_infinite += x == -infinity;
            nan += std::isnan(x);
            if (!std::isnan(shifted) || !std::isnan(padded)) {
                fprintf(stderr, "missing NaN padding for maximum %08x\n",
                    to_bits(x));
                return 1;
            }
        }
        if ((x == -infinity && to_bits(exponential) != 0) ||
            (x == infinity && to_bits(exponential) != 0x7f800000) ||
            (std::isnan(x) &&
                (!std::isnan(exponential) || !std::isnan(actual[4 * i + 4])))) {
            fprintf(
                stderr, "expf special-value failure for %08x\n", to_bits(x));
            return 1;
        }
        different_from_fast +=
            std::isfinite(x) &&
            to_bits(exponential) != to_bits(expected[5 * i + 4]);
    }
    if (!finite || !positive_infinite || !negative_infinite || !nan ||
        !different_from_fast) {
        fprintf(stderr, "missing exceptional-value or fast-math coverage\n");
        return 1;
    }
    MUST(cudaFree(device_actual));
    MUST(cudaFree(device_expected));
    MUST(cudaFree(device_inputs));
    puts("Float32 special-value and conditional-padding checks passed.");
}
