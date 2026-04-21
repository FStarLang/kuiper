module Kuiper.Poly.GEMM.BlockTiling1D

#lang-pulse

open Kuiper
open Kuiper.Poly.GEMMGPU.Type

let size_req : tiled_size_req_t =
  fun m n k tile -> m * n <= max_blocks

inline_for_extraction noextract
val mmcomb_gpu_approx : tiled_matmulcomb_gpu_approx_ty size_req
