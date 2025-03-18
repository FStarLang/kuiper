#!/bin/bash

set -eux

LAPS=5
DIM=4096
TILE=8

# This is just a first stab, the tiled versions will use different
# tiles internally, so this is not a good test.

# Just one lap of the naivest-one, it takes like a minute.
./obj/Test_Kuiper_GEMM_Naive__F32.exe 1 $DIM $DIM $DIM 0
./obj/Test_Kuiper_GEMM_Naive2__F32.exe $LAPS $DIM $DIM $DIM 0
./obj/Test_Kuiper_GEMM_Tiled__F32.exe $LAPS $DIM $DIM $DIM $TILE 0
./obj/Test_Kuiper_GEMM_SHMem__F32.exe $LAPS $DIM $DIM $DIM $TILE 0
./obj/Test_Kuiper_GEMM_SHMem__F32_GEMM.exe $LAPS $DIM $DIM $DIM

exit 0
