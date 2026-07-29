#!/bin/bash

set -eu

make -f verify.mk obj/Klas_GEMM_TensorCore2D_To.o

funcs=$(grep -Eo 'Klas_GEMM_TensorCore2D_To_g_gemm_[^() ]*' obj/Klas_GEMM_TensorCore2D_To.cu)

outf=$(mktemp -p . tune_XXXXXX.out)

echo "Saving output to $outf"

go () {
  date
  nvidia-smi

  for func in $funcs; do
    echo "About to test $func"
    nvcc -O3 -I include -I obj \
            -o bench.exe \
            -DKUIPER_CFG_TENSORCORES=1 \
            -Dstem="$func" \
            -Dtile_sizes= \
            -Dtc_tile_sizes= \
            -Dregch_sizes= \
            -Det_lbl= \
            obj/Klas_GEMM_TensorCore2D_To.o \
            test/Tune_Klas_GEMM_TensorCore2D_To.cu
    ./bench.exe 200 4096 4096 4096 0 || echo "RES ERROR"
  done
}

go |& tee "$outf"
