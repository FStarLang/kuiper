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

__device__ __half native_exp(__half x) { return hexp(x); }
__device__ __nv_bfloat16 native_exp(__nv_bfloat16 x) { return hexp(x); }
__device__ double native_exp(double x) { return exp(x); }

template <typename T>
__global__ void exp_reference(const T *inputs, T *outputs, uint32_t n)
{
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
        i += blockDim.x * gridDim.x)
        outputs[i] = native_exp(inputs[i]);
}

template <typename T, typename Bits>
static void check_exponential(
    const char *name, unsigned fraction_bits, void (*run)(uint32_t, T *, T *))
{
    static_assert(sizeof(T) == sizeof(Bits), "float representation");
    const Bits sign = Bits{1} << (sizeof(Bits) * 8 - 1);
    const Bits fraction = (Bits{1} << fraction_bits) - 1;
    const Bits infinity = (sign - 1) & ~fraction;
    const Bits one = (sign >> 1) - (Bits{1} << fraction_bits);
    const Bits quiet = Bits{1} << (fraction_bits - 1);
    std::vector<Bits> inputs = {0, sign, infinity, Bits(sign | infinity),
        Bits(infinity | 1), Bits(sign | infinity | 1), Bits(infinity | quiet),
        Bits(sign | infinity | quiet), one, Bits(sign | one), 1, Bits(sign | 1),
        Bits(infinity - 1), Bits(sign | (infinity - 1))};
    if (sizeof(Bits) == 2)
        for (uint32_t bits = 0; bits < 65536; ++bits)
            inputs.push_back(static_cast<Bits>(bits));
    const uint32_t n = static_cast<uint32_t>(inputs.size());
    const Bits guard = one | 123;
    std::vector<Bits> actual(n + 2, guard), expected(n), preserved(n);
    T *device_inputs, *device_actual, *device_expected;
    MUST(cudaMalloc(&device_inputs, n * sizeof(T)));
    MUST(cudaMalloc(&device_actual, actual.size() * sizeof(T)));
    MUST(cudaMalloc(&device_expected, n * sizeof(T)));
    MUST(cudaMemcpy(
        device_inputs, inputs.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(device_actual, actual.data(), actual.size() * sizeof(T),
        cudaMemcpyHostToDevice));
    exp_reference<<<128, 128>>>(device_inputs, device_expected, n);
    MUST(cudaGetLastError());
    run(n, device_inputs, device_actual + 1);
    MUST(cudaMemcpy(actual.data(), device_actual, actual.size() * sizeof(T),
        cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(expected.data(), device_expected, n * sizeof(T),
        cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(preserved.data(), device_inputs, n * sizeof(T),
        cudaMemcpyDeviceToHost));
    if (actual.front() != guard || actual.back() != guard ||
        inputs != preserved) {
        fprintf(stderr, "%s exp guard or input preservation failure\n", name);
        exit(1);
    }
    for (uint32_t i = 0; i < n; ++i) {
        Bits x = inputs[i], result = actual[i + 1];
        bool is_nan = (x & infinity) == infinity && (x & fraction);
        bool result_is_nan =
            (result & infinity) == infinity && (result & fraction);
        if (result != expected[i] || ((x == 0 || x == sign) && result != one) ||
            (x == infinity && result != infinity) ||
            (x == Bits(sign | infinity) && result != 0) ||
            (is_nan && !result_is_nan)) {
            fprintf(stderr, "%s exp case 0x%016llx: got 0x%016llx\n", name,
                static_cast<unsigned long long>(x),
                static_cast<unsigned long long>(result));
            exit(1);
        }
    }
    MUST(cudaFree(device_expected));
    MUST(cudaFree(device_actual));
    MUST(cudaFree(device_inputs));
}

__global__ void subtraction_reference(
    const float *left, const float *right, float *outputs, uint32_t n)
{
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
        i += blockDim.x * gridDim.x)
        outputs[i] = __fsub_rn(left[i], right[i]);
}

static void check_subtraction()
{
    const uint32_t special[] = {0, 0x80000000, 0x7f800000, 0xff800000,
        0x7fc12345, 0xffc12345, 0x7f800001, 0xff800001, 0x3f800000, 0xbf800000,
        0x3f800001, 0x33800000, 0xb3800000, 1, 0x80000001, 0x007fffff,
        0x00800000, 0x7f7fffff, 0xff7fffff};
    std::vector<float> left, right;
    for (uint32_t x : special)
        for (uint32_t y : special) {
            left.push_back(from_bits(x));
            right.push_back(from_bits(y));
        }
    std::mt19937 random(1);
    for (unsigned i = 0; i < 50000; ++i) {
        left.push_back(float_test::random_float32(random));
        right.push_back(float_test::random_float32(random));
    }
    const uint32_t n = static_cast<uint32_t>(left.size());
    constexpr uint32_t guard = 0x4b123456;
    std::vector<float> actual(2 * n + 2, from_bits(guard)), expected(n),
        preserved_left(n), preserved_right(n);
    float *device_left, *device_right, *device_actual, *device_expected;
    MUST(cudaMalloc(&device_left, n * sizeof(float)));
    MUST(cudaMalloc(&device_right, n * sizeof(float)));
    MUST(cudaMalloc(&device_actual, actual.size() * sizeof(float)));
    MUST(cudaMalloc(&device_expected, n * sizeof(float)));
    MUST(cudaMemcpy(
        device_left, left.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(
        device_right, right.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(device_actual, actual.data(), actual.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    subtraction_reference<<<128, 128>>>(
        device_left, device_right, device_expected, n);
    MUST(cudaGetLastError());
    Kuiper_Example_Float32SpecialValues_run_subtraction(
        n, device_left, device_right, device_actual + 1);
    MUST(cudaMemcpy(actual.data(), device_actual, actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(expected.data(), device_expected, n * sizeof(float),
        cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(preserved_left.data(), device_left, n * sizeof(float),
        cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(preserved_right.data(), device_right, n * sizeof(float),
        cudaMemcpyDeviceToHost));
    if (to_bits(actual.front()) != guard || to_bits(actual.back()) != guard ||
        memcmp(left.data(), preserved_left.data(), n * sizeof(float)) ||
        memcmp(right.data(), preserved_right.data(), n * sizeof(float))) {
        fprintf(stderr, "subtraction guard or input preservation failure\n");
        exit(1);
    }
    for (uint32_t i = 0; i < n; ++i) {
        float x = left[i], y = right[i], result = actual[2 * i + 1];
        bool valid = true;
        if (std::isnan(x) || std::isnan(y) ||
            (std::isinf(x) && std::isinf(y) &&
                std::signbit(x) == std::signbit(y)))
            valid = std::isnan(result);
        else if (std::isinf(x))
            valid = to_bits(result) == to_bits(x);
        else if (std::isinf(y))
            valid = to_bits(result) == (to_bits(y) ^ float_test::sign_mask);
        else if (to_bits(x) == to_bits(y))
            valid = to_bits(result) == 0;
        else if (x == 0 && y == 0)
            valid = to_bits(result) == to_bits(x);
        if (!valid || to_bits(result) != to_bits(expected[i]) ||
            to_bits(result) != to_bits(actual[2 * i + 2])) {
            fprintf(stderr, "sub_rn(%08x, %08x): got %08x, expected %08x\n",
                to_bits(x), to_bits(y), to_bits(result), to_bits(expected[i]));
            exit(1);
        }
    }
    MUST(cudaFree(device_expected));
    MUST(cudaFree(device_actual));
    MUST(cudaFree(device_right));
    MUST(cudaFree(device_left));
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
        if ((x == 0 && to_bits(exponential) != 0x3f800000) ||
            (x == -infinity && to_bits(exponential) != 0) ||
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
    check_exponential<__half, uint16_t>(
        "Float16", 10, Kuiper_Example_Float32SpecialValues_run_exp_f16);
    check_exponential<__nv_bfloat16, uint16_t>(
        "BFloat16", 7, Kuiper_Example_Float32SpecialValues_run_exp_bf16);
    check_exponential<double, uint64_t>(
        "Float64", 52, Kuiper_Example_Float32SpecialValues_run_exp_f64);
    check_subtraction();
    puts("All-format exponential, deterministic Float32 subtraction, and "
         "conditional-padding checks passed.");
}
