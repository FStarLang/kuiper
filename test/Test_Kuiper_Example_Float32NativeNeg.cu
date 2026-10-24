#include "test-common.h"
#include "float_test_common.c.inc"
#include <vector>
#include "Kuiper_Example_Float32NativeNeg.h"

using float_test::from_bits;
using float_test::to_bits;

__global__ void reference(const float *inputs, float *outputs, uint32_t n)
{
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
        i += blockDim.x * gridDim.x) {
        float value;
        asm("neg.f32 %0, %1;" : "=f"(value) : "f"(inputs[i]));
        outputs[2 * i] = value;
        outputs[2 * i + 1] = __fsub_rn(0.0f, inputs[i]);
    }
}

int main()
{
    std::vector<float> inputs;
    for (uint32_t bits = 0; bits < 65536; ++bits)
        inputs.push_back(from_bits(bits << 16));
    std::mt19937 random(0);
    for (unsigned i = 0; i < 50000; ++i)
        inputs.push_back(float_test::random_float32(random));
    for (uint32_t bits :
        {0u, 0x80000000u, 1u, 0x80000001u, 0x007fffffu, 0x807fffffu,
            0x00800000u, 0x80800000u, 0x7f7fffffu, 0xff7fffffu, 0x7f800000u,
            0xff800000u, 0x7fc00001u, 0xffc00001u, 0x7f800001u, 0xff800001u})
        inputs.push_back(from_bits(bits));
    const uint32_t n = inputs.size();
    constexpr uint32_t guard = 0x4b123456u;
    std::vector<float> expected(2 * n), actual(n + 2, from_bits(guard));
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
    Kuiper_Example_Float32NativeNeg_run(n, device_inputs, device_actual + 1);
    MUST(cudaMemcpy(expected.data(), device_expected,
        expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(actual.data(), device_actual, actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(preserved.data(), device_inputs,
        preserved.size() * sizeof(float), cudaMemcpyDeviceToHost));
    const auto is_nan = [](uint32_t bits) {
        return (bits & float_test::exponent_mask) ==
                   float_test::exponent_mask &&
               (bits & float_test::fraction_mask) != 0;
    };
    unsigned different = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t want = to_bits(inputs[i]) ^ float_test::sign_mask;
        const uint32_t got = to_bits(actual[i + 1]);
        const uint32_t ptx = to_bits(expected[2 * i]);
        const uint32_t host =
            to_bits(Kuiper_Example_Float32NativeNeg_host_negate(inputs[i]));
        // NaN signs and payloads are outside the public native contract.
        const bool matches = is_nan(to_bits(inputs[i]))
                                 ? is_nan(got) && is_nan(ptx) && is_nan(host)
                                 : got == want && ptx == want && host == want;
        if (!matches || to_bits(preserved[i]) != to_bits(inputs[i])) {
            fprintf(stderr,
                "case %u: input %08x, got %08x, PTX %08x, host %08x\n", i,
                to_bits(inputs[i]), got, ptx, host);
            return 1;
        }
        different +=
            !is_nan(to_bits(inputs[i])) && ptx != to_bits(expected[2 * i + 1]);
    }
    if (to_bits(actual.front()) != guard || to_bits(actual.back()) != guard ||
        to_bits(expected[0]) != 0x80000000u || to_bits(expected[1]) != 0u ||
        !different) {
        fprintf(stderr, "guard or positive-zero subtraction control failed\n");
        return 1;
    }
    MUST(cudaFree(device_actual));
    MUST(cudaFree(device_expected));
    MUST(cudaFree(device_inputs));
    puts("Float32 native-negation checks passed.");
}
