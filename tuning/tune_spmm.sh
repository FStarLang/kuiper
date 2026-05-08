#!/bin/bash

set -eu

make -f verify.mk obj/Klas_SPMM.o

funcs=$(grep -Eo 'Klas_SPMM_g_spmm_[^() ]*' obj/Klas_SPMM.cu)

outf=$(mktemp -p . tune_XXXXXX.out)

echo "Saving output to $outf"

go () {
  date
  nvidia-smi

  for func in $funcs; do
    echo "About to test $func"
    nvcc -O3 -I include -I obj \
            -o bench.exe \
            -arch=native \
            -Dstem=$func \
            obj/Klas_SPMM.o \
            test/Tune_Klas_SPMM.cu
    ./bench.exe 10 2048 2048 2048 10 || echo "RES ERROR"
  done
}

go |& tee $outf
