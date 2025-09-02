module Kuiper.Poly.GEMM.SHMem

#lang-pulse

open Kuiper
open Kuiper.Poly.GEMMGPU.Type

inline_for_extraction noextract
let size_req : tiled_size_req_t =
  fun mrows mshared mcols tile ->
    mrows * mcols <= max_blocks /\
    tile * tile <= max_threads

inline_for_extraction noextract
val mmcomb_gpu : tiled_matmulcomb_gpu_ty size_req

