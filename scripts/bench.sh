#!/bin/bash

set -eux

LAPS=5
DIM=4096
TILE=8
CHECK=0

# This is just a first stab, the tiled versions will use different
# tiles internally, so this is not a good test.

# Just one lap of the naivest-one, it takes like a minute.
./obj/Test_Kuiper_GEMM_Naive__F32.exe 1 $DIM $DIM $DIM 0
./obj/Test_Kuiper_GEMM_Naive2__F32.exe $LAPS $DIM $DIM $DIM 0
./obj/Test_Kuiper_GEMM_Tiled__F32.exe $LAPS $DIM $DIM $DIM $TILE 0
./obj/Test_Kuiper_GEMM_SHMem__F32.exe $LAPS $DIM $DIM $DIM $TILE 0
./obj/Test_Kuiper_GEMM_SHMem__F32_GEMM.exe $LAPS $DIM $DIM $DIM 0

./obj/Test_Kuiper_GEMM_BlockTiling1D__F32.exe $LAPS $DIM $DIM $DIM $TILE 0
./obj/Test_Kuiper_GEMM_BlockTiling1D__F32_GEMM.exe $LAPS $DIM $DIM $DIM 0

./obj/Test_Kuiper_GEMM_OrigBlockTiling1D__F32_GEMM.exe $LAPS $DIM $DIM $DIM 0

./obj/Test_Kuiper_GEMM_BlockTiling2D__F32_128x128x8_8x8_GEMM.exe $LAPS $DIM $DIM $DIM 0

obj/Test_Kuiper_GEMM_TensorCore2D__F16_F16_64x64x64_16x16x16_2x2.exe $LAPS $DIM $DIM $DIM 0

make -C bench

echo "CUBLAS:"
./bench/bench 0

# This takes like 15 minutes
# echo "bespoke naive"
# ./bench/bench 1

echo "bespoke gmem coallesce"
./bench/bench 2

echo "bespoke shmem"
./bench/bench 3

echo "bespoke 1d blocktiling"
./bench/bench 4

echo "bespoke 2d blocktiling"
./bench/bench 5

exit 0
