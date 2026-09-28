#include <cstring>
#include <vector>
#include "Kuiper_Example_Float32Trigonometry.h"

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
    std::vector<float> inputs = {1.0f, from_bits(1), from_bits(0x80000001),
        from_bits(0x007fffff), from_bits(0x807fffff), from_bits(0x7f800001),
        from_bits(0xff800001), from_bits(0x7fc12345), from_bits(0xffc12345)};
    for (uint32_t bits = 0; bits < 65536; ++bits)
        inputs.push_back(from_bits(bits << 16));
    uint32_t state = 0x13579bdf;
    for (uint32_t i = 0; i < 131072; ++i) {
        state = state * 1664525u + 1013904223u;
        inputs.push_back(from_bits(state));
    }
    uint32_t n = static_cast<uint32_t>(inputs.size());
    std::vector<float> expected(4 * n),
        actual(2 * n + 2, from_bits(0x7fc12345)), preserved(n);
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
    Kuiper_Example_Float32Trigonometry_run(n, device_inputs, device_actual + 1);
    MUST(cudaMemcpy(expected.data(), device_expected,
        expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(actual.data(), device_actual, actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(preserved.data(), device_inputs,
        inputs.size() * sizeof(float), cudaMemcpyDeviceToHost));
    unsigned different[2] = {};
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t op = 0; op < 2; ++op) {
            if (to_bits(actual[2 * i + op + 1]) !=
                to_bits(expected[4 * i + op])) {
                fprintf(stderr, "case %u op %u: got %08x, expected %08x\n", i,
                    op, to_bits(actual[2 * i + op + 1]),
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
    if (to_bits(actual.front()) != 0x7fc12345 ||
        to_bits(actual.back()) != 0x7fc12345 ||
        memcmp(
            inputs.data(), preserved.data(), inputs.size() * sizeof(float))) {
        fprintf(stderr, "modified inputs or output guards\n");
        return 1;
    }
    MUST(cudaFree(device_actual));
    MUST(cudaFree(device_expected));
    MUST(cudaFree(device_inputs));
    puts("Float32 trigonometry checks passed.");
}
