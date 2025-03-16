#!/bin/bash

set -eux

LAPS=5
DIM=4096
TILE=8

# This is just a first stab, the tiled versions will use different
# tiles internally, so this is not a good test.

./obj/Test_Kuiper_MatMul_Naive__F32.exe $LAPS $DIM $DIM $DIM 0
./obj/Test_Kuiper_MatMul_Naive2__F32.exe $LAPS $DIM $DIM $DIM 0
./obj/Test_Kuiper_MatMul_Tiled__F32.exe $LAPS $DIM $DIM $DIM $TILE 0
./obj/Test_Kuiper_MatMul_SHMem__F32.exe $LAPS $DIM $DIM $DIM $TILE 0

exit 0
