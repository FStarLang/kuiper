#include "Klas_UGS_Packed.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t status = (expr);                                           \
        if (status != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error: %s at %s:%d\n",                 \
                         cudaGetErrorString(status), __FILE__, __LINE__);       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

__global__ void f2h_kernel(const float *input, half *output, size_t count)
{
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        output[i] = __float2half_rn(input[i]);
}

static std::vector<float> gen_floats(size_t count, uint64_t seed)
{
    std::vector<float> values(count);
    uint64_t state = seed ^ 0x9e3779b97f4a7c15ULL;
    for (size_t i = 0; i < count; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t bits = static_cast<uint32_t>(state >> 32);
        float unit = static_cast<float>(bits) * (1.0f / 4294967296.0f);
        values[i] = unit * 0.04f - 0.02f;
    }
    return values;
}

static void fill_half(half *device, size_t count, uint64_t seed)
{
    std::vector<float> host = gen_floats(count, seed);
    float *device_float = nullptr;
    CUDA_CHECK(cudaMalloc(&device_float, count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_float, host.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice));
    constexpr int threads = 256;
    int blocks = static_cast<int>((count + threads - 1) / threads);
    f2h_kernel<<<blocks, threads>>>(device_float, device, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(device_float));
}

static std::vector<half> read_reference(const std::string &path, size_t count)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        std::fprintf(stderr, "failed to open reference: %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }
    std::streamsize bytes = input.tellg();
    if (bytes != static_cast<std::streamsize>(count * sizeof(half))) {
        std::fprintf(stderr, "reference has %lld bytes; expected %zu\n",
                     static_cast<long long>(bytes), count * sizeof(half));
        std::exit(EXIT_FAILURE);
    }
    input.seekg(0);
    std::vector<half> result(count);
    input.read(reinterpret_cast<char *>(result.data()), bytes);
    if (!input) {
        std::fprintf(stderr, "failed to read reference: %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }
    return result;
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s M N K reference.bin\n", argv[0]);
        return EXIT_FAILURE;
    }

    uint32_t m = static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10));
    uint32_t n = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
    uint32_t k = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
    uint32_t n2 = 2 * n;

    size_t a_count = static_cast<size_t>(m) * k;
    size_t w_count = static_cast<size_t>(k) * n2;
    size_t out_count = static_cast<size_t>(m) * n;

    half *a = nullptr;
    half *w = nullptr;
    half *output = nullptr;
    CUDA_CHECK(cudaMalloc(&a, a_count * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&w, w_count * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&output, out_count * sizeof(half)));

    fill_half(a, a_count, 2023);
    fill_half(w, w_count, 2022);

    Klas_UGS_Packed_swiglu(m, k, n, a, w, w, output);

    std::vector<half> actual(out_count);
    CUDA_CHECK(cudaMemcpy(actual.data(), output, out_count * sizeof(half),
                          cudaMemcpyDeviceToHost));
    std::vector<half> expected = read_reference(argv[4], out_count);

    size_t exact = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < out_count; ++i) {
        uint16_t actual_bits;
        uint16_t expected_bits;
        std::memcpy(&actual_bits, &actual[i], sizeof(actual_bits));
        std::memcpy(&expected_bits, &expected[i], sizeof(expected_bits));
        exact += actual_bits == expected_bits;
        max_abs = std::fmax(max_abs,
                            std::fabs(__half2float(actual[i]) -
                                      __half2float(expected[i])));
    }

    std::printf("bit-exact elements: %zu / %zu (%.4f%%), max_abs=%g\n",
                exact, out_count, 100.0 * static_cast<double>(exact) / out_count,
                max_abs);

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(w));
    CUDA_CHECK(cudaFree(output));
    return exact == out_count ? EXIT_SUCCESS : EXIT_FAILURE;
}
