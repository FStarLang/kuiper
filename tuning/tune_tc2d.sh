#!/bin/bash

set -eu

make -f verify.mk obj/Klas_GEMM_TensorCore2D.o

funcs=$(grep -Eo 'Klas_GEMM_TensorCore2D_g_gemm_[^() ]*' obj/Klas_GEMM_TensorCore2D.cu)

outf=$(mktemp -p . tune_XXXXXX.out)

echo "Saving output to $outf"

go () {
  date
  nvidia-smi

  for func in $funcs; do
    # Skip the dynamic versions
    if [ "$func" = "Klas_GEMM_TensorCore2D_g_gemm_f16_f16_16x16x16_2x2" ] \
    || [ "$func" = "Klas_GEMM_TensorCore2D_g_gemm_f16_f16_16x16x16_4x4" ] \
    || [ "$func" = "Klas_GEMM_TensorCore2D_g_gemm_f16_f16_16x16x16_8x8" ]
    then
      echo "Skipping $func"
      continue
    fi
    echo "About to test $func"
    nvcc -O3 -I include -I obj \
            -o bench.exe \
            -DKUIPER_CFG_TENSORCORES=1 \
            -Dstem=$func \
            -Dtile_sizes= \
            -Dtc_tile_sizes= \
            -Dregch_sizes= \
            -Det_lbl= \
            obj/Klas_GEMM_TensorCore2D.o \
            test/Tune_Klas_GEMM_TensorCore2D.cu
    ./bench.exe 200 4096 4096 4096 0 || echo "RES ERROR"
  done
}

go |& tee $outf
