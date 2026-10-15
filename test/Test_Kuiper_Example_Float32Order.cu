#include "test-common.h"
#include "float_test_common.c.inc"
#include <vector>

#include "Kuiper_Example_Float32Order.cu"

#define ORDER_API(name) Kuiper_Example_Float32Order_##name

using float_test::from_bits;
using float_test::sign_mask;

static bool is_nan_bits(uint32_t bits)
{
    return (bits & ~sign_mask) > float_test::exponent_mask;
}

// Integer ordering is independent of the extracted floating comparison.
static bool less_bits(uint32_t x, uint32_t y)
{
    if (is_nan_bits(x) || is_nan_bits(y))
        return false;
    if (((x | y) & ~sign_mask) == 0)
        return false;
    uint32_t kx = (x & sign_mask) ? ~x : (x ^ sign_mask);
    uint32_t ky = (y & sign_mask) ? ~y : (y ^ sign_mask);
    return kx < ky;
}

__global__ void compare_triples(
    const uint32_t *values, unsigned char *out, unsigned count)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        float x = __uint_as_float(values[3 * index]);
        float y = __uint_as_float(values[3 * index + 1]);
        float z = __uint_as_float(values[3 * index + 2]);
        out[3 * index] = ORDER_API(compare)(x, y);
        out[3 * index + 1] = ORDER_API(compare)(y, z);
        out[3 * index + 2] = ORDER_API(compare)(x, z);
    }
}

static void fail(const char *message, uint32_t x, uint32_t y, uint32_t z = 0)
{
    fprintf(stderr, "%s: %08x %08x %08x\n", message, x, y, z);
    exit(1);
}

int main()
{
    constexpr unsigned count = 50000;
    std::mt19937 random(0);
    std::vector<uint32_t> values(3 * count);
    for (uint32_t &value : values)
        value = float_test::to_bits(float_test::random_float32(random));
    std::vector<unsigned char> results(3 * count);
    uint32_t *device_values;
    unsigned char *device_results;
    MUST(cudaMalloc(&device_values, values.size() * sizeof(uint32_t)));
    MUST(cudaMalloc(&device_results, results.size()));
    MUST(cudaMemcpy(device_values, values.data(),
        values.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    compare_triples<<<(count + 127) / 128, 128>>>(
        device_values, device_results, count);
    MUST(cudaGetLastError());
    MUST(cudaMemcpy(results.data(), device_results, results.size(),
        cudaMemcpyDeviceToHost));
    MUST(cudaFree(device_results));
    MUST(cudaFree(device_values));

    for (unsigned i = 0; i < count; ++i) {
        const unsigned pairs[][2] = {{0, 1}, {1, 2}, {0, 2}};
        for (unsigned edge = 0; edge < 3; ++edge) {
            uint32_t x = values[3 * i + pairs[edge][0]];
            uint32_t y = values[3 * i + pairs[edge][1]];
            bool expected = less_bits(x, y);
            if (ORDER_API(compare)(from_bits(x), from_bits(y)) != expected)
                fail("Host comparison differs from integer oracle", x, y);
            if (results[3 * i + edge] != expected)
                fail("Device comparison differs from integer oracle", x, y);
            if (results[3 * i + edge] && (is_nan_bits(x) || is_nan_bits(y)))
                fail("Successful comparison has a NaN operand", x, y);
        }
        if (results[3 * i] && results[3 * i + 1] && !results[3 * i + 2])
            fail("Strict comparison is not transitive", values[3 * i],
                values[3 * i + 1], values[3 * i + 2]);
    }
    puts("Float32 ordering checks passed.");
}
