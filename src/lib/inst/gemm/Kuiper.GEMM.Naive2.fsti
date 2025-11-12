module Kuiper.GEMM.Naive2

#lang-pulse
open Kuiper
open Kuiper.Poly.GEMMCPU
open Kuiper.Matrix.Reprs { row_major as RM, col_major as CM }

inline_for_extraction noextract
let size_req : size_req_t =
  fun rows shared cols -> rows * cols <= max_blocks * max_threads
