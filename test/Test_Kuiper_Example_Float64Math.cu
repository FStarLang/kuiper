#include <cstring>
#include <vector>
#include "Kuiper_Example_Float64Math.h"

struct math_ops {
    using value_type = double;
    using bits_type = uint64_t;
    static constexpr bits_type guard_bits = 0x7ff8123456789abcULL;

    static __device__ value_type native_pow(value_type x, value_type y)
    {
        return pow(x, y);
    }
    static __device__ value_type native_sin(value_type x) { return sin(x); }
    static __device__ value_type native_cos(value_type x) { return cos(x); }
    static __device__ value_type roundtrip(value_type x)
    {
        return static_cast<double>(
            __bfloat162float(__float2bfloat16_rn(static_cast<float>(x))));
    }
    static void run(uint32_t n, value_type *inputs, value_type *outputs)
    {
        Kuiper_Example_Float64Math_run(n, inputs, outputs);
    }
    static std::vector<bits_type> special_values()
    {
        return {0, 0x8000000000000000ULL, 1, 0x8000000000000001ULL,
            0x000fffffffffffffULL, 0x800fffffffffffffULL, 0x0010000000000000ULL,
            0x8010000000000000ULL, 0x3ff0000000000000ULL, 0xbff0000000000000ULL,
            0x3ff0000000000001ULL, 0x3fd5555555555555ULL, 0x7fefffffffffffffULL,
            0xffefffffffffffffULL, 0x7ff0000000000000ULL, 0xfff0000000000000ULL,
            0x7ff8000000000001ULL, 0xfff8123456789abcULL, 0x7ff0000000000001ULL,
            0xfff0000000000001ULL};
    }
    static const char *name() { return "Float64"; }
};

#include "float_math_test.c.inc"
