#include "Kuiper_Example_FloatEquality.h"
#include <cmath>
#include <cstring>
#include <type_traits>

#define EQ_API(name) Kuiper_Example_FloatEquality_##name

static void check(bool condition)
{
    if (!condition) {
        fprintf(stderr, "Floating equality check failed\n");
        exit(1);
    }
}

template <typename T, typename U> static T from_bits(U bits)
{
    static_assert(sizeof(T) == sizeof(U), "float representation size");
    static_assert(
        std::is_trivially_copyable<T>::value, "float representation copy");
    T value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

template <typename T, typename U, size_t N>
static void check_pairs(const U (&values)[N], U magnitude_mask, U infinity_bits,
    bool (*ieee)(T, T), bool (*bits)(T, T), bool (*compat_eq)(T, T))
{
    for (U x : values) {
        for (U y : values) {
            T fx = from_bits<T>(x), fy = from_bits<T>(y);
            U mx = x & magnitude_mask, my = y & magnitude_mask;
            bool numeric = mx <= infinity_bits && my <= infinity_bits &&
                           (x == y || (mx == 0 && my == 0));
            check(ieee(fx, fy) == numeric);
            check(compat_eq(fx, fy) == numeric);
            check(bits(fx, fy) == (x == y));
        }
    }
}

// Exercise all four device bit-cast paths with zero signs and NaN payloads.
__global__ void device_equalities(unsigned char *out)
{
    float fp = __uint_as_float(0), fn = __uint_as_float(0x80000000U);
    double dp = __longlong_as_double(0),
           dn = __longlong_as_double(0x8000000000000000ULL);
    __half hp = __ushort_as_half(0), hn = __ushort_as_half(0x8000);
    __nv_bfloat16 bp = __ushort_as_bfloat16(0),
                  bn = __ushort_as_bfloat16(0x8000);
    out[0] = fp == fn && !kpr_f32_bit_eq(fp, fn);
    out[1] = dp == dn && !kpr_f64_bit_eq(dp, dn);
    out[2] = hp == hn && !kpr_f16_bit_eq(hp, hn);
    out[3] = bp == bn && !kpr_bf16_bit_eq(bp, bn);
    float fq = __uint_as_float(0x7fc00001U);
    double dq = __longlong_as_double(0x7ff8000000000001ULL);
    __half hq = __ushort_as_half(0x7e01);
    __nv_bfloat16 bq = __ushort_as_bfloat16(0x7fc1);
    out[4] = !(fq == fq) && kpr_f32_bit_eq(fq, fq) &&
             !kpr_f32_bit_eq(fq, __uint_as_float(0x7fc00002U));
    out[5] = !(dq == dq) && kpr_f64_bit_eq(dq, dq) &&
             !kpr_f64_bit_eq(dq, __longlong_as_double(0x7ff8000000000002ULL));
    out[6] = !(hq == hq) && kpr_f16_bit_eq(hq, hq) &&
             !kpr_f16_bit_eq(hq, __ushort_as_half(0x7e02));
    out[7] = !(bq == bq) && kpr_bf16_bit_eq(bq, bq) &&
             !kpr_bf16_bit_eq(bq, __ushort_as_bfloat16(0x7fc2));
}

int main(int argc, char **argv)
{
    const uint16_t half_bits[] = {0, 0x8000, 0x3c00, 0xbc00, 0x7c00, 0xfc00,
        0x7e01, 0x7e02, 0xfe01, 1, 0x8001, 0x7bff, 0x7f80, 0xff80, 0x7fc1,
        0x7fc2, 0xffc1, 0x7f7f, 0x7c01, 0xfc01, 0x7f81, 0xff81};
    const uint32_t float_bits[] = {0, 0x80000000, 0x3f800000, 0xbf800000,
        0x7f800000, 0xff800000, 0x7fc00001, 0x7fc00002, 0xffc00001, 1,
        0x80000001, 0x7f7fffff, 0x7f800001, 0xff800001};
    const uint64_t double_bits[] = {0, 0x8000000000000000ULL,
        0x3ff0000000000000ULL, 0xbff0000000000000ULL, 0x7ff0000000000000ULL,
        0xfff0000000000000ULL, 0x7ff8000000000001ULL, 0x7ff8000000000002ULL,
        0xfff8000000000001ULL, 1, 0x8000000000000001ULL, 0x7fefffffffffffffULL,
        0x7ff0000000000001ULL, 0xfff0000000000001ULL};
    check_pairs<__half>(half_bits, uint16_t(0x7fff), uint16_t(0x7c00),
        EQ_API(ieee_f16), EQ_API(bits_f16), EQ_API(eq_f16));
    check_pairs<__nv_bfloat16>(half_bits, uint16_t(0x7fff), uint16_t(0x7f80),
        EQ_API(ieee_bf16), EQ_API(bits_bf16), EQ_API(eq_bf16));
    check_pairs<float>(float_bits, uint32_t(0x7fffffff), uint32_t(0x7f800000),
        EQ_API(ieee_f32), EQ_API(bits_f32), EQ_API(eq_f32));
    check_pairs<double>(double_bits, uint64_t(0x7fffffffffffffffULL),
        uint64_t(0x7ff0000000000000ULL), EQ_API(ieee_f64), EQ_API(bits_f64),
        EQ_API(eq_f64));

    float fz = EQ_API(mul_zero_f32)(-1.0f);
    double dz = EQ_API(mul_zero_f64)(-1.0);
    check(EQ_API(ieee_f32)(fz, 0.0f) && !EQ_API(bits_f32)(fz, 0.0f));
    check(EQ_API(ieee_f64)(dz, 0.0) && !EQ_API(bits_f64)(dz, 0.0));
    check(!EQ_API(bits_f32)(EQ_API(add_zero_f32)(fz), fz));
    check(!EQ_API(bits_f64)(EQ_API(add_zero_f64)(dz), dz));
    float fi = EQ_API(reciprocal_f32)(fz), fp = EQ_API(reciprocal_f32)(0.0f);
    double di = EQ_API(reciprocal_f64)(dz), dp = EQ_API(reciprocal_f64)(0.0);
    check(std::isinf(fi) && std::signbit(fi) && std::isinf(fp) &&
          !std::signbit(fp));
    check(std::isinf(di) && std::signbit(di) && std::isinf(dp) &&
          !std::signbit(dp));
    check(!EQ_API(ieee_f32)(fi, fp) && !EQ_API(ieee_f64)(di, dp));

    if (argc == 2 && strcmp(argv[1], "--host-only") == 0) {
        fprintf(stderr, "GPU checks skipped (--host-only)\n");
    } else {
        check(argc == 1);
        unsigned char *device;
        unsigned char results[8];
        MUST(cudaMalloc(&device, sizeof results));
        device_equalities<<<1, 1>>>(device);
        MUST(cudaGetLastError());
        MUST(cudaMemcpy(
            results, device, sizeof results, cudaMemcpyDeviceToHost));
        MUST(cudaFree(device));
        for (unsigned char result : results)
            check(result == 1);
    }
    puts("Floating equality checks passed.");
}
