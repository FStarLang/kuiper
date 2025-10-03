#!/bin/bash

cat << EOF
module Kuiper.GEMM.BlockTiling2D

#lang-pulse

open Kuiper
open Kuiper.Matrix.Reprs { row_major as rm, col_major as cm }
open Kuiper.GEMM.BlockTiling2D.Inst { spec }

// Dynamic parameter version. Only admitted since otherwise we
// need to repeat all requirements here. We only use this for some
// quick tuning, so it\'s not a big deal.
let g_gemm_f32_8x8_cr bm bn bk =
  admit();
  spec bm bn bk (cm _ _) (rm _ _)
    8sz 8sz f32

EOF

for bm in 64 128; do
  for bn in 64 128; do
    if [ $((bn % 4)) -ne 0 ]; then continue; fi
    for bk in 8 16 32 64; do
      if [ $((bk % 4)) -ne 0 ]; then continue; fi
      if [ $(((4 * bm * bk) + (4 * bk * bn))) -gt 49152 ]; then continue; fi
      for tm in 8 16; do
        if [ $((bm % tm)) -ne 0 ]; then continue; fi
        for tn in 8 16; do
          if [ $((bn % tn)) -ne 0 ]; then continue; fi
          if [ $(((bm / tm) * (bn / tn))) -gt 1024 ]; then continue; fi
          for la in c; do
            for lb in r; do
              echo "let g_gemm_f32_${bm}x${bn}x${bk}_${tm}x${tn}_${la}${lb} = spec ${bm}sz ${bn}sz ${bk}sz (${la}m _ _) (${lb}m _ _) ${tm}sz ${tn}sz f32"
            done
          done
        done
      done
    done
  done
done
