#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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
        std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__ << ":" << err << std::endl; \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

void run_benchmark(int N, cublasHandle_t handle) {
    //std::cout << "Benchmarking SGEMM for matrix size " << N << "x" << N << std::endl;
    int numElements = N * N;
    size_t bytes = numElements * sizeof(float);

    // Allocate and initialize host matrices
    float *h_A = new float[numElements];
    float *h_B = new float[numElements];
    float *h_C = new float[numElements];

    for (int i = 0; i < numElements; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        h_C[i] = 0.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // Copy matrices from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Create CUDA events for timing


    // Warm-up call
    int iterations = 1000;
    int warmups = iterations/10;
    for (int i = 0; i < warmups; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, N, N,
                                &alpha,
                                d_A, N,
                                d_B, N,
                                &beta,
                                d_C, N));
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    // Timing SGEMM over multiple iterations (e.g., 10 iterations)
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        /*CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, CUDA_R_32F,
            N, d_B, CUDA_R_32F, N, &beta, d_C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));*/
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N, N,
                                 &alpha,
                                 d_B, N,
                                 d_A, N,
                                 &beta,
                                 d_C, N));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    float avgTime = milliseconds / iterations;
    //std::cout << "Average time per SGEMM call: " << avgTime << " ms" << std::endl;

    // Calculate GFLOPs: (2*N^3 operations) / (average time in seconds * 1e9)
    // Since avgTime is in milliseconds, convert to seconds by dividing by 1000.
    double gflops = (2.0 * N * N * N) / ((avgTime / 1000.0) * 1e9);
    //std::cout << "GFLOPs: " << gflops << std::endl;
    //std::cout << "---------------------------------------" << std::endl;
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
    int sizes[] = {1024, 2048, 4096, 8192};
    //int sizes[] = {8192, 8192*2};
    //int sizes[] = {2048};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    std::cout << "size, time, GFLOPS" << std::endl;

    for (int i = 0; i < numSizes; i++) {
        run_benchmark(sizes[i], handle);
    }

    // Destroy cuBLAS handle and exit
    CHECK_CUBLAS(cublasDestroy(handle));
    return 0;
}
