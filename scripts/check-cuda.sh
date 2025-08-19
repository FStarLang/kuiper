#!/bin/bash

set -uex

# Is nvcc installed?
nvcc --version

cleanup () {
    rm -f tmp.cu check.exe
}
trap cleanup EXIT

# Basic test
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

nvcc tmp.cu -o check.exe
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
nvcc -DKUIPER_CFG_TENSORCORES=${KUIPER_CFG_TENSORCORES} -I include/ tmp.cu -o check.exe

./check.exe
