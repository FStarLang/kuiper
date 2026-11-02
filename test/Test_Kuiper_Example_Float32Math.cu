#include <cstring>
#include <vector>
#include "Kuiper_Example_Float32Math.h"

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

__global__ void reference(const float *inputs, float *outputs, uint32_t n)
{
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
        i += blockDim.x * gridDim.x) {
        float base = inputs[3 * i], coordinate = inputs[3 * i + 1],
              position = inputs[3 * i + 2];
        float power = powf(base, coordinate / 64.0f);
        float angle = position * (1.0f / power);
        float sn = sinf(angle), cs = cosf(angle);
        outputs[10 * i] = power;
        outputs[10 * i + 1] = sinf(position);
        outputs[10 * i + 2] = cosf(position);
        outputs[10 * i + 3] = angle;
        outputs[10 * i + 4] = sn;
        outputs[10 * i + 5] = cs;
        outputs[10 * i + 6] = __bfloat162float(__float2bfloat16_rn(sn));
        outputs[10 * i + 7] = __bfloat162float(__float2bfloat16_rn(cs));
        outputs[10 * i + 8] = __sinf(position);
        outputs[10 * i + 9] = __cosf(position);
    }
}

int main()
{
    std::vector<float> inputs;
    for (uint32_t position = 0; position < 7; ++position)
        for (uint32_t coordinate = 0; coordinate < 64; ++coordinate) {
            inputs.push_back(1000000.0f);
            inputs.push_back(static_cast<float>(coordinate));
            inputs.push_back(static_cast<float>(position));
        }
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        inputs.push_back(1000000.0f);
        inputs.push_back(1.0f);
        inputs.push_back(from_bits(bits << 16));
    }
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
        0x7f7fffff,
        0xff7fffff,
        0x7f800000,
        0xff800000,
        0x7fc00001,
        0xffc12345,
        0x7f800001,
        0xff800001,
    };
    for (uint32_t x : special)
        for (uint32_t y : special)
            for (uint32_t z : special) {
                inputs.push_back(from_bits(x));
                inputs.push_back(from_bits(y));
                inputs.push_back(from_bits(z));
            }
    uint32_t state = 0x2468ace1;
    for (uint32_t i = 0; i < 32768 * 3; ++i) {
        state = state * 1664525u + 1013904223u;
        inputs.push_back(from_bits(state));
    }
    uint32_t n = static_cast<uint32_t>(inputs.size() / 3);
    std::vector<float> expected(10 * n),
        actual(8 * n + 2, from_bits(0x7fc12345)), preserved(inputs.size());
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
    Kuiper_Example_Float32Math_run(n, device_inputs, device_actual + 1);
    MUST(cudaMemcpy(expected.data(), device_expected,
        expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(actual.data(), device_actual, actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(preserved.data(), device_inputs,
        inputs.size() * sizeof(float), cudaMemcpyDeviceToHost));
    const char *names[] = {"powf", "sinf", "cosf", "rotary angle",
        "rotary sine", "rotary cosine", "BF16 sine", "BF16 cosine"};
    unsigned different[2] = {};
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
            if (to_bits(actual[8 * i + j + 1]) !=
                to_bits(expected[10 * i + j])) {
                fprintf(stderr,
                    "case %u, base/coordinate/position "
                    "0x%08x/0x%08x/0x%08x\n",
                    i, to_bits(inputs[3 * i]), to_bits(inputs[3 * i + 1]),
                    to_bits(inputs[3 * i + 2]));
                check_bits(
                    names[j], actual[8 * i + j + 1], expected[10 * i + j]);
            }
        }
        if (i < 7 * 64) {
            different[0] +=
                to_bits(expected[10 * i + 1]) != to_bits(expected[10 * i + 8]);
            different[1] +=
                to_bits(expected[10 * i + 2]) != to_bits(expected[10 * i + 9]);
        }
    }
    check_bits("leading output guard", actual.front(), from_bits(0x7fc12345));
    check_bits("trailing output guard", actual.back(), from_bits(0x7fc12345));
    if (!different[0] || !different[1] ||
        memcmp(
            inputs.data(), preserved.data(), inputs.size() * sizeof(float))) {
        fprintf(
            stderr, "missing ordinary/fast distinction or modified inputs\n");
        exit(1);
    }
    MUST(cudaFree(device_actual));
    MUST(cudaFree(device_expected));
    MUST(cudaFree(device_inputs));
    puts("Float32 ordinary math checks passed.");
}
