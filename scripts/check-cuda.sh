#!/bin/bash

set -uex

cleanup () {
    rm -f tmp.cu check_cuda.exe
}
trap cleanup EXIT

cat >tmp.cu <<EOF
#include <stdio.h>
#include <assert.h>

__global__ void helloCUDA()
{
    printf("Hello, CUDA!\n");
}

int main()
{
    helloCUDA<<<1, 1>>>();
    if (cudaGetLastError() != cudaSuccess)
      assert(!"kcall");
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess)
      assert(!"sync");
    return 0;
}
EOF

nvcc tmp.cu -o check_cuda.exe

./check_cuda.exe

rm -f tmp.cu check_cuda.exe

exit -0
