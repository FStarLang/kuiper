#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// Simple error checking macros
#define CHECK_CUDA(call) {                                          \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess) {                                       \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)      \
                  << " in " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

#define CHECK_CUBLAS(call) {                                        \
    cublasStatus_t err = call;                                      \
    if (err != CUBLAS_STATUS_SUCCESS) {                             \
        std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

void run_benchmark(int N, cublasHandle_t handle) {
    //std::cout << "Benchmarking GEMM for half precision for matrix size " << N << "x" << N << std::endl;
    int numElements = N * N;
    size_t bytes = numElements * sizeof(__half);

    // Allocate and initialize host matrices
    __half *h_A = new __half[numElements];
    __half *h_B = new __half[numElements];
    __half *h_C = new __half[numElements];

    for (int i = 0; i < numElements; i++) {
        // Generate a random float and convert to half
        float randValA = static_cast<float>(rand()) / RAND_MAX;
        float randValB = static_cast<float>(rand()) / RAND_MAX;
        h_A[i] = __float2half_rn(randValA);
        h_B[i] = __float2half_rn(randValB);
        h_C[i] = __float2half_rn(0.0f);
    }

    // Allocate device memory
    __half *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // Copy matrices from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice));

    // Set alpha and beta as half-precision values.
    __half h_alpha = __float2half_rn(1.0f);
    __half h_beta = __float2half_rn(0.0f);

    // Create CUDA events for timing


    // Warm-up call
    int iterations = 1000;

    for (int i = 0; i < iterations/10; i++) {
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, N, N,
                              &h_alpha,
                              d_A, CUDA_R_16F, N,
                              d_B, CUDA_R_16F, N,
                              &h_beta,
                              d_C, CUDA_R_16F, N,
                              CUBLAS_COMPUTE_16F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing GEMM over multiple iterations (e.g., 50 iterations)

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, N, N,
                                  &h_alpha,
                                  d_A, CUDA_R_16F, N,
                                  d_B, CUDA_R_16F, N,
                                  &h_beta,
                                  d_C, CUDA_R_16F, N,
                                  CUBLAS_COMPUTE_16F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    float avgTime = milliseconds / iterations;
    // Calculate GFLOPs: (2*N^3 operations) / (average time in seconds * 1e9)
    double gflops = (2.0 * N * N * N) / ((avgTime / 1000.0) * 1e9);
    std::cout << N << ", " << avgTime << ", " << gflops << std::endl;

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // List of matrix sizes to benchmark
    //int sizes[] = {1024, 2048, 4096, 8192, 8192*2};
    //int sizes[] = {4096};
    int sizes[] = {2048, 4096, 8192};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    std::cout << "size, time, GFLOPS" << std::endl;

    for (int i = 0; i < numSizes; i++) {
        run_benchmark(sizes[i], handle);
    }

    // Destroy cuBLAS handle and exit
    CHECK_CUBLAS(cublasDestroy(handle));
    return 0;
}