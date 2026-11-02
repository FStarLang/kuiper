// Shared regression for the ordinary math API in each floating format.
// The driver supplies native CUDA operations independently of Kuiper's
// wrappers.
using value_type = math_ops::value_type;
using bits_type = math_ops::bits_type;

static value_type from_bits(bits_type bits)
{
    static_assert(
        sizeof(value_type) == sizeof(bits_type), "float representation");
    value_type value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static bits_type to_bits(value_type value)
{
    bits_type bits;
    memcpy(&bits, &value, sizeof bits);
    return bits;
}

static void check_bits(const char *name, value_type got, value_type expected)
{
    if (to_bits(got) != to_bits(expected)) {
        fprintf(stderr, "%s: got 0x%016llx, expected 0x%016llx\n", name,
            static_cast<unsigned long long>(to_bits(got)),
            static_cast<unsigned long long>(to_bits(expected)));
        exit(1);
    }
}

__global__ void reference(
    const value_type *inputs, value_type *outputs, uint32_t n)
{
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
        i += blockDim.x * gridDim.x) {
        value_type base = inputs[3 * i], coordinate = inputs[3 * i + 1],
                   position = inputs[3 * i + 2];
        value_type power = math_ops::native_pow(
            base, coordinate / static_cast<value_type>(64));
        value_type angle = position * (static_cast<value_type>(1) / power);
        value_type sn = math_ops::native_sin(angle),
                   cs = math_ops::native_cos(angle);
        outputs[8 * i] = power;
        outputs[8 * i + 1] = math_ops::native_sin(position);
        outputs[8 * i + 2] = math_ops::native_cos(position);
        outputs[8 * i + 3] = angle;
        outputs[8 * i + 4] = sn;
        outputs[8 * i + 5] = cs;
        outputs[8 * i + 6] = math_ops::roundtrip(sn);
        outputs[8 * i + 7] = math_ops::roundtrip(cs);
    }
}

int main()
{
    std::vector<value_type> inputs;
    // Keep the rotary base finite in Float16 as well as BFloat16 and Float64.
    for (uint32_t position = 0; position < 7; ++position)
        for (uint32_t coordinate = 0; coordinate < 64; ++coordinate) {
            inputs.push_back(static_cast<value_type>(10000));
            inputs.push_back(static_cast<value_type>(coordinate));
            inputs.push_back(static_cast<value_type>(position));
        }
    // Exercise every 16-bit representation in all three argument positions.
    if (sizeof(bits_type) == 2) {
        for (uint32_t bits = 0; bits < 65536; ++bits) {
            value_type value = from_bits(static_cast<bits_type>(bits));
            inputs.insert(inputs.end(), {static_cast<value_type>(10000),
                                            static_cast<value_type>(1), value});
            inputs.insert(inputs.end(), {value, static_cast<value_type>(32),
                                            static_cast<value_type>(1)});
            inputs.insert(inputs.end(), {static_cast<value_type>(2), value,
                                            static_cast<value_type>(1)});
        }
    }
    // Include signed zeros, subnormals, infinities, and both kinds of NaNs.
    auto special = math_ops::special_values();
    for (bits_type x : special)
        for (bits_type y : special)
            for (bits_type z : special)
                inputs.insert(
                    inputs.end(), {from_bits(x), from_bits(y), from_bits(z)});
    uint64_t state = 0x2468ace113579bdfULL;
    for (uint32_t i = 0; i < 32768 * 3; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        inputs.push_back(from_bits(static_cast<bits_type>(state)));
    }
    uint32_t n = static_cast<uint32_t>(inputs.size() / 3);
    value_type guard = from_bits(math_ops::guard_bits);
    std::vector<value_type> expected(8 * n), actual(8 * n + 2, guard),
        preserved(inputs.size());
    value_type *device_inputs, *device_expected, *device_actual;
    MUST(cudaMalloc(&device_inputs, inputs.size() * sizeof(value_type)));
    MUST(cudaMalloc(&device_expected, expected.size() * sizeof(value_type)));
    MUST(cudaMalloc(&device_actual, actual.size() * sizeof(value_type)));
    MUST(cudaMemcpy(device_inputs, inputs.data(),
        inputs.size() * sizeof(value_type), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(device_actual, actual.data(),
        actual.size() * sizeof(value_type), cudaMemcpyHostToDevice));
    reference<<<128, 128>>>(device_inputs, device_expected, n);
    MUST(cudaGetLastError());
    math_ops::run(n, device_inputs, device_actual + 1);
    MUST(cudaMemcpy(expected.data(), device_expected,
        expected.size() * sizeof(value_type), cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(actual.data(), device_actual,
        actual.size() * sizeof(value_type), cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(preserved.data(), device_inputs,
        inputs.size() * sizeof(value_type), cudaMemcpyDeviceToHost));
    const char *names[] = {"power", "sine", "cosine", "rotary angle",
        "rotary sine", "rotary cosine", "cast sine", "cast cosine"};
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
            if (to_bits(actual[8 * i + j + 1]) !=
                to_bits(expected[8 * i + j])) {
                fprintf(stderr,
                    "case %u, base/coordinate/position "
                    "0x%016llx/0x%016llx/0x%016llx\n",
                    i, static_cast<unsigned long long>(to_bits(inputs[3 * i])),
                    static_cast<unsigned long long>(to_bits(inputs[3 * i + 1])),
                    static_cast<unsigned long long>(
                        to_bits(inputs[3 * i + 2])));
                check_bits(
                    names[j], actual[8 * i + j + 1], expected[8 * i + j]);
            }
        }
    }
    check_bits("leading output guard", actual.front(), guard);
    check_bits("trailing output guard", actual.back(), guard);
    if (memcmp(inputs.data(), preserved.data(),
            inputs.size() * sizeof(value_type))) {
        fprintf(stderr, "modified inputs\n");
        exit(1);
    }
    MUST(cudaFree(device_actual));
    MUST(cudaFree(device_expected));
    MUST(cudaFree(device_inputs));
    printf("%s ordinary math checks passed.\n", math_ops::name());
}
