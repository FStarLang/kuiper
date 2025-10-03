#!/bin/bash

set -eux

make -f verify.mk obj/Kuiper_GEMM_BlockTiling2D.o

# These loops must match those in Kuiper.GEMM.BlockTiling2D.fst.sh!!
for bm in 64 128; do
  for bn in 64 128; do
    if [ $((bn % 4)) -ne 0 ]; then continue; fi
    for bk in 8 16 32 64; do
      if [ $((bk % 4)) -ne 0 ]; then continue; fi
      if [ $(((4 * bm * bk) + (4 * bk * bn))) -gt 49152 ]; then continue; fi
      for tm in 8 16; do
        if [ $((bm % tm)) -ne 0 ]; then continue; fi
        for tn in 8 16 ; do
          if [ $((bn % tn)) -ne 0 ]; then continue; fi
          if [ $(((bm / tm) * (bn / tn))) -gt 1024 ]; then continue; fi
          for la in c; do
            for lb in r; do
              nvcc -O3 -I include -I obj \
                      -o bench.exe \
                      -DKUIPER_CFG_TENSORCORES=0 \
                      -Dtile_sizes=_${bm}x${bn}x${bk} \
                      -Dregch_sizes=_${tm}x${tn} \
                      obj/Kuiper_GEMM_BlockTiling2D.o \
                      test/Tune_Kuiper_GEMM_BlockTiling2D.cu
              ./bench.exe 50 4096 4096 4096 0 || echo "RES ERROR"
            done
          done
        done
      done
    done
  done
done
