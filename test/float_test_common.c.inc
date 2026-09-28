#include <cstdint>
#include <cstring>
#include <random>

namespace float_test
{

constexpr uint32_t sign_mask = uint32_t{1} << 31;
constexpr uint32_t fraction_mask = (uint32_t{1} << 23) - 1;
constexpr uint32_t exponent_mask = uint32_t{255} << 23;
constexpr uint32_t quiet_nan_mask = uint32_t{1} << 22;

inline float from_bits(uint32_t bits)
{
    float value;
    static_assert(sizeof value == sizeof bits, "binary32 representation");
    memcpy(&value, &bits, sizeof value);
    return value;
}

inline uint32_t to_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    return bits;
}

// Sample exceptional classes explicitly; raw bits alone rarely produce
// infinity.
inline float random_float32(std::mt19937 &random)
{
    uint32_t bits = random();
    switch (random() % 7) {
    case 0:
        bits &= sign_mask;
        break;
    case 1:
        bits = (bits & sign_mask) | exponent_mask;
        break;
    case 2:
        bits |= exponent_mask | quiet_nan_mask;
        break;
    case 3:
        bits = (bits & ~quiet_nan_mask) | exponent_mask | 1;
        break;
    case 4:
        bits = (bits & (sign_mask | fraction_mask)) | 1;
        break;
    case 5:
        return (static_cast<int32_t>(bits % 65536) - 32768) / 256.0f;
    default:
        break;
    }
    return from_bits(bits);
}

} // namespace float_test
