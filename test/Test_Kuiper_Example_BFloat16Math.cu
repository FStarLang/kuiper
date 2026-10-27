#include <cstring>
#include <vector>
#include "Kuiper_Example_BFloat16Math.h"

struct math_ops {
    using value_type = __nv_bfloat16;
    using bits_type = uint16_t;
    static constexpr bits_type guard_bits = 0x7fc5;

    static __device__ value_type native_pow(value_type x, value_type y)
    {
        return __float2bfloat16_rn(
            powf(__bfloat162float(x), __bfloat162float(y)));
    }
    static __device__ value_type native_sin(value_type x) { return hsin(x); }
    static __device__ value_type native_cos(value_type x) { return hcos(x); }
    static __device__ value_type roundtrip(value_type x)
    {
        return __float2bfloat16_rn(
            __half2float(__float2half_rn(__bfloat162float(x))));
    }
    static void run(uint32_t n, value_type *inputs, value_type *outputs)
    {
        Kuiper_Example_BFloat16Math_run(n, inputs, outputs);
    }
    static std::vector<bits_type> special_values()
    {
        return {0, 0x8000, 1, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080, 0x3f80,
            0xbf80, 0x3f81, 0x3eab, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1,
            0xffc5, 0x7f81, 0xff81};
    }
    static const char *name() { return "BFloat16"; }
};

#include "float_math_test.c.inc"
