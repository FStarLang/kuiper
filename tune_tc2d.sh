#!/bin/bash

set -eux

# These loops must match those in Kuiper.GEMM.TensorCore2D.fst.sh!!
for bm in 32 64 128; do
  if [ $((bm % 16)) -ne 0 ]; then continue; fi # tc tile
  for bn in 32 64 128; do
    if [ $((bn % 8)) -ne 0 ]; then continue; fi # chunk
    if [ $((bn % 16)) -ne 0 ]; then continue; fi # tc tile
    for bk in 8 16 32 64; do
      if [ $((bk % 8)) -ne 0 ]; then continue; fi # chunk
      if [ $((bk % 16)) -ne 0 ]; then continue; fi # tc tile
      if [ $(((2 * bm * bk) + (2 * bk * bn))) -gt 49152 ]; then continue; fi # shmem size constraint
      for wm in 2 4 8 16; do
        if [ $((bm % (16 * wm))) -ne 0 ]; then continue; fi
        for wn in 2 4 8 16; do
          if [ $((bn % (16 * wn))) -ne 0 ]; then continue; fi
          if [ $(((bm / (wm*16)) * (bn / (wn*16)) * 32)) -gt 1024 ]; then continue; fi
          nvcc -O3 -I include -I obj \
                  -o bench.exe \
                  -DKUIPER_CFG_TENSORCORES=1 \
                  -Dtile_sizes=_${bm}x${bn}x${bk} \
                  -Dregch_sizes=_${wm}x${wn} \
                  obj/Kuiper_GEMM_TensorCore2D.cu \
                  test/Tune_Kuiper_GEMM_TensorCore2D.cu
          ./bench.exe 200 4096 4096 4096 0 || echo "RES ERROR"
        done
      done
    done
  done
done
