#!/bin/bash

set -uex

# What driver are we on? (Not fatal: NVML can be broken while compute works.)
nvidia-smi || echo "*** nvidia-smi failed"

# Is nvcc installed?
nvcc --version

cleanup () {
    rm -f tmp.cu check.exe
}
trap cleanup EXIT

# Basic test
cat >tmp.cu <<EOF
#include <stdio.h>

#define CHECK(what) do { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
      fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); \
      return 1; \
    } \
  } while (0)

__global__ void helloCUDA()
{
    printf("Hello, CUDA!\n");
}

int main()
{
    helloCUDA<<<1, 1>>>();
    CHECK("kcall");
    cudaDeviceSynchronize();
    CHECK("sync");
    return 0;
}
EOF

nvcc -arch=native tmp.cu -o check.exe
./check.exe

# Now try with kuiper.h
cat >tmp.cu <<EOF
#include "kuiper.h"

int main()
{
	INFO();
	return 0;
}
EOF

./configure _env
source _env
rm -f _env
nvcc -arch=native -DKUIPER_CFG_TENSORCORES=${KUIPER_CFG_TENSORCORES} -I include/ tmp.cu -o check.exe

./check.exe
