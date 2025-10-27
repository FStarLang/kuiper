#!/bin/bash

set -eux

LAPS=10
DIM=4096

./obj/Test_Kuiper_GEMM_TensorCorePiped__F16_F16_64x64x32_16x16x16_4x4.exe $LAPS $DIM $DIM $DIM 0
./obj/Test_Kuiper_GEMM_TensorCorePiped_change_sync__F16_F16_64x64x32_16x16x16_4x4.exe $LAPS $DIM $DIM $DIM 0
./obj/Test_Kuiper_GEMM_TensorCorePiped_align__F16_F16_64x64x32_16x16x16_4x4.exe $LAPS $DIM $DIM $DIM 0
./obj/Test_Kuiper_GEMM_TensorCorePiped_align__F16_F16_128x128x64_16x16x16_4x8.exe $LAPS $DIM $DIM $DIM 0

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
