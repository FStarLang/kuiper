#include "float_test_common.c.inc"
#include <vector>
#include "Kuiper_Example_Float32Trigonometry.h"

using float_test::to_bits;

__global__ void reference(const float *inputs, float *outputs, uint32_t n)
{
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
        i += blockDim.x * gridDim.x) {
        float x = inputs[i];
        outputs[4 * i] = __sinf(x);
        outputs[4 * i + 1] = __cosf(x);
        outputs[4 * i + 2] = sinf(x);
        outputs[4 * i + 3] = cosf(x);
    }
}

int main()
{
    constexpr uint32_t n = 50000;
    std::mt19937 random(0);
    std::vector<float> inputs(n);
    for (float &value : inputs)
        value = float_test::random_float32(random);
    inputs.front() = 1.0f;
    std::vector<float> expected(4 * n), actual(2 * n);
    float *device_inputs, *device_expected, *device_actual;
    MUST(cudaMalloc(&device_inputs, inputs.size() * sizeof(float)));
    MUST(cudaMalloc(&device_expected, expected.size() * sizeof(float)));
    MUST(cudaMalloc(&device_actual, actual.size() * sizeof(float)));
    MUST(cudaMemcpy(device_inputs, inputs.data(), inputs.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    reference<<<128, 128>>>(device_inputs, device_expected, n);
    MUST(cudaGetLastError());
    Kuiper_Example_Float32Trigonometry_run(n, device_inputs, device_actual);
    MUST(cudaMemcpy(expected.data(), device_expected,
        expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(actual.data(), device_actual, actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    unsigned different[2] = {};
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t op = 0; op < 2; ++op) {
            if (to_bits(actual[2 * i + op]) != to_bits(expected[4 * i + op])) {
                fprintf(stderr, "case %u op %u: got %08x, expected %08x\n", i,
                    op, to_bits(actual[2 * i + op]),
                    to_bits(expected[4 * i + op]));
                return 1;
            }
            different[op] += to_bits(expected[4 * i + op]) !=
                             to_bits(expected[4 * i + 2 + op]);
        }
    }
    if (!different[0] || !different[1]) {
        fprintf(stderr, "missing generic-libm negative-control coverage\n");
        return 1;
    }
    MUST(cudaFree(device_actual));
    MUST(cudaFree(device_expected));
    MUST(cudaFree(device_inputs));
    puts("Float32 trigonometry checks passed.");
}
