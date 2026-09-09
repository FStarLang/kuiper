// Regression test for the extracted exact-operation sigmoid. Compare raw
// bits, including NaN payloads and signed zero, with the CUDA reference.
#define Kuiper_Example_Sigmoid_sigmoid test_sigmoid
#include "Kuiper_Example_Sigmoid.cu"
#undef Kuiper_Example_Sigmoid_sigmoid
#include <cstdio>
#include <cstdlib>
#include <vector>

template <bool Reference> __global__ void evaluate(uint32_t *output) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  // This odd affine permutation uniformly spreads 2^20 samples over the
  // 32-bit space and includes 0. Additional values cover both signs of
  // zero, min/max subnormal, min normal, max finite, infinity, and NaNs.
  const uint32_t edge[] = {
      0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x807fffff,
      0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
      0x7f800001, 0xff800001, 0x7fc00000, 0xffc00000, 0x7fffffff, 0xffffffff};
  uint32_t u = i < sizeof(edge) / sizeof(edge[0]) ? edge[i] : i * 2654435761u;
  float x = __uint_as_float(u);
  float y;
  if (Reference) {
    float exponential = expf(-x);
    float denominator = __fadd_rn(1.0f, exponential);
    y = __fdiv_rn(1.0f, denominator);
  } else {
    y = test_sigmoid(x);
  }
  output[i] = __float_as_uint(y);
}

int main() {
  const size_t n = 1048576;
  uint32_t *device_ref, *device_got;
  MUST(cudaMalloc(&device_ref, n * sizeof(uint32_t)));
  MUST(cudaMalloc(&device_got, n * sizeof(uint32_t)));
  evaluate<true><<<4096, 256>>>(device_ref);
  MUST(cudaGetLastError());
  evaluate<false><<<4096, 256>>>(device_got);
  MUST(cudaGetLastError());
  MUST(cudaDeviceSynchronize());
  std::vector<uint32_t> ref(n), got(n);
  MUST(cudaMemcpy(ref.data(), device_ref, n * sizeof(uint32_t),
                  cudaMemcpyDeviceToHost));
  MUST(cudaMemcpy(got.data(), device_got, n * sizeof(uint32_t),
                  cudaMemcpyDeviceToHost));
  MUST(cudaFree(device_ref));
  MUST(cudaFree(device_got));
  uint32_t mismatches = 0;
  for (size_t i = 0; i < n; ++i)
    mismatches += ref[i] != got[i];
  if (mismatches) {
    std::printf("%u bit mismatches\n", mismatches);
    return 1;
  }
  std::printf("1048576 bitwise sigmoid comparisons, OK\n");
  return 0;
}
