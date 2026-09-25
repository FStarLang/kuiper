#include <cstring>
#include <vector>
#include "Kuiper_Example_Float32FastMath.h"

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
    const uint32_t special[] = {
        0,
        0x80000000,
        1,
        0x80000001,
        0x007fffff,
        0x807fffff,
        0x00800000,
        0x80800000,
        0x3f800000,
        0xbf800000,
        0x3f800001,
        0x3eaaaaab,
        0x7e800001,
        0x7f7fffff,
        0xff7fffff,
        0x7f800000,
        0xff800000,
        0x7fc00001,
        0xffc12345,
        0x7f800001,
        0xff800001,
    };
    std::vector<float> inputs = {-87.33984375f, 1.0f, 0.0f};
    for (uint32_t x : special)
        for (uint32_t y : special)
            for (uint32_t z : special) {
                inputs.push_back(from_bits(x));
                inputs.push_back(from_bits(y));
                inputs.push_back(from_bits(z));
            }
    uint32_t state = 0x13579bdf;
    for (uint32_t i = 0; i < 131072; ++i) {
        for (uint32_t j = 0; j < 3; ++j) {
            state = state * 1664525u + 1013904223u;
            inputs.push_back(
                i % 2 ? from_bits(state)
                      : (static_cast<int32_t>(state % 53761) - 28160) / 256.0f);
        }
    }
    uint32_t n = static_cast<uint32_t>(inputs.size() / 3);
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
    if (to_bits(actual[0]) == 0 || to_bits(actual[0]) >= 0x00800000 ||
        !different[0] || !different[1] || !different[2]) {
        fprintf(
            stderr, "missing subnormal or replacement-sensitive coverage\n");
        return 1;
    }
    MUST(cudaFree(device_actual));
    MUST(cudaFree(device_expected));
    MUST(cudaFree(device_inputs));
    puts("Float32 fast-math checks passed.");
}
