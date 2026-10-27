#include <cstring>
#include <vector>
#include "Kuiper_Example_Float16Math.h"

struct math_ops {
    using value_type = __half;
    using bits_type = uint16_t;
    static constexpr bits_type guard_bits = 0x7e55;

    static __device__ value_type native_pow(value_type x, value_type y)
    {
        return __float2half_rn(powf(__half2float(x), __half2float(y)));
    }
    static __device__ value_type native_sin(value_type x) { return hsin(x); }
    static __device__ value_type native_cos(value_type x) { return hcos(x); }
    static __device__ value_type roundtrip(value_type x)
    {
        return __float2half_rn(
            __bfloat162float(__float2bfloat16_rn(__half2float(x))));
    }
    static void run(uint32_t n, value_type *inputs, value_type *outputs)
    {
        Kuiper_Example_Float16Math_run(n, inputs, outputs);
    }
    static std::vector<bits_type> special_values()
    {
        return {0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x3c00,
            0xbc00, 0x3c01, 0x3555, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e01,
            0xfe55, 0x7c01, 0xfc01};
    }
    static const char *name() { return "Float16"; }
};

#include "float_math_test.c.inc"
