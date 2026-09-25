#include "test-common.h"
#include <cstring>
#include <vector>

#include "Kuiper_Example_Float32Order.cu"

#define ORDER_API(name) Kuiper_Example_Float32Order_##name

static float from_bits(uint32_t bits)
{
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static bool is_nan_bits(uint32_t bits)
{
    return (bits & 0x7fffffffU) > 0x7f800000U;
}

// Integer ordering is independent of the extracted floating comparison.
static bool less_bits(uint32_t x, uint32_t y)
{
    if (is_nan_bits(x) || is_nan_bits(y))
        return false;
    if (((x | y) & 0x7fffffffU) == 0)
        return false;
    uint32_t kx = (x & 0x80000000U) ? ~x : (x ^ 0x80000000U);
    uint32_t ky = (y & 0x80000000U) ? ~y : (y ^ 0x80000000U);
    return kx < ky;
}

__global__ void compare_pairs(
    const uint32_t *values, unsigned char *out, unsigned count)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count * count)
        out[index] = ORDER_API(compare)(__uint_as_float(values[index / count]),
            __uint_as_float(values[index % count]));
}

static void fail(const char *message, uint32_t x, uint32_t y, uint32_t z = 0)
{
    fprintf(stderr, "%s: %08x %08x %08x\n", message, x, y, z);
    exit(1);
}

int main()
{
    std::vector<uint32_t> values = {0, 0x80000000U, 1, 0x80000001U, 0x007fffffU,
        0x807fffffU, 0x00800000U, 0x80800000U, 0x3f7fffffU, 0x3f800000U,
        0x3f800001U, 0xbf7fffffU, 0xbf800000U, 0xbf800001U, 0x7f7fffffU,
        0xff7fffffU, 0x7f800000U, 0xff800000U, 0x7fc00000U, 0x7fc00001U,
        0xffc00001U, 0x7f800001U, 0xff800001U, 0x7fffffffU};
    uint32_t random = 0x31415926U;
    for (unsigned i = 0; i < 64; ++i) {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        values.push_back(random);
    }
    unsigned count = static_cast<unsigned>(values.size());
    std::vector<unsigned char> results(count * count);
    uint32_t *device_values;
    unsigned char *device_results;
    MUST(cudaMalloc(&device_values, values.size() * sizeof(uint32_t)));
    MUST(cudaMalloc(&device_results, results.size()));
    MUST(cudaMemcpy(device_values, values.data(),
        values.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    compare_pairs<<<(count * count + 127) / 128, 128>>>(
        device_values, device_results, count);
    MUST(cudaGetLastError());
    MUST(cudaMemcpy(results.data(), device_results, results.size(),
        cudaMemcpyDeviceToHost));
    MUST(cudaFree(device_results));
    MUST(cudaFree(device_values));

    for (unsigned i = 0; i < count; ++i) {
        for (unsigned j = 0; j < count; ++j) {
            uint32_t x = values[i], y = values[j];
            bool expected = less_bits(x, y);
            if (ORDER_API(compare)(from_bits(x), from_bits(y)) != expected)
                fail("Host comparison differs from integer oracle", x, y);
            if (results[i * count + j] != expected)
                fail("Device comparison differs from integer oracle", x, y);
            if (!results[i * count + j])
                continue;
            if (is_nan_bits(x) || is_nan_bits(y))
                fail("Successful comparison has a NaN operand", x, y);
            for (unsigned k = 0; k < count; ++k) {
                if (results[j * count + k] && !results[i * count + k])
                    fail(
                        "Strict comparison is not transitive", x, y, values[k]);
            }
        }
    }
    puts("Float32 ordering checks passed.");
}
