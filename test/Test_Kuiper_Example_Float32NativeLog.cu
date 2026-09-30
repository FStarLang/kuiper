#include "test-common.h"
#include "float_test_common.c.inc"
#include <cmath>
#include <vector>
#include "Kuiper_Example_Float32NativeLog.h"

using float_test::to_bits;

__global__ void reference(const float *inputs, float *outputs, uint32_t n)
{
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
        i += blockDim.x * gridDim.x) {
        outputs[2 * i] = __logf(inputs[i]);
        outputs[2 * i + 1] = logf(inputs[i]);
    }
}

int main()
{
    constexpr uint32_t n = 50000;
    std::mt19937 random(0);
    std::vector<float> inputs(n);
    for (float &value : inputs)
        value = float_test::random_float32(random);
    inputs.front() = std::nextafter(2.0f, 3.0f);
    std::vector<float> expected(2 * n), actual(n);
    float *device_inputs, *device_expected, *device_actual;
    MUST(cudaMalloc(&device_inputs, inputs.size() * sizeof(float)));
    MUST(cudaMalloc(&device_expected, expected.size() * sizeof(float)));
    MUST(cudaMalloc(&device_actual, actual.size() * sizeof(float)));
    MUST(cudaMemcpy(device_inputs, inputs.data(), inputs.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    reference<<<128, 128>>>(device_inputs, device_expected, n);
    MUST(cudaGetLastError());
    Kuiper_Example_Float32NativeLog_run(n, device_inputs, device_actual);
    MUST(cudaMemcpy(expected.data(), device_expected,
        expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(actual.data(), device_actual, actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    unsigned different = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (to_bits(actual[i]) != to_bits(expected[2 * i])) {
            fprintf(stderr, "case %u: got %08x, expected %08x\n", i,
                to_bits(actual[i]), to_bits(expected[2 * i]));
            return 1;
        }
        different += to_bits(expected[2 * i]) != to_bits(expected[2 * i + 1]);
    }
    if (!different) {
        fprintf(stderr, "missing generic-logf negative-control coverage\n");
        return 1;
    }
    MUST(cudaFree(device_actual));
    MUST(cudaFree(device_expected));
    MUST(cudaFree(device_inputs));
    puts("Float32 native-log checks passed.");
}
