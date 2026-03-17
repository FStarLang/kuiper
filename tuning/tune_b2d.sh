#!/bin/bash

set -eu

make -f verify.mk obj/Kuiper_GEMM_BlockTiling2D.o

funcs=$(grep -Eo 'Kuiper_GEMM_BlockTiling2D_g_gemm_[^() ]*' obj/Kuiper_GEMM_BlockTiling2D.cu)

outf=$(mktemp -p . tune_XXXXXX.out)

echo "Saving output to $outf"

go () {
  date
  nvidia-smi

  for func in $funcs; do
    # Skip the dynamic versions
    if [ "$func" = "Kuiper_GEMM_BlockTiling2D_g_gemm_f32_8x8" ]
    then
      echo "Skipping $func"
      continue
    fi
    echo "About to test $func"
    nvcc -I include -I obj \
            -o bench.exe \
            -Dstem=$func \
            -Dtile_sizes= \
            -Dregch_sizes= \
            -Det_lbl= \
            -DKUIPER_CFG_TENSORCORES=0 \
            obj/Kuiper_GEMM_BlockTiling2D.o \
            test/Tune_Kuiper_GEMM_BlockTiling2D.cu
    ./bench.exe 50 4096 4096 4096 0 || echo "RES ERROR"
  done
}

go |& tee $outf
