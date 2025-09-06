module Kuiper.Poly.GEMM.TensorCore

#lang-pulse

open Kuiper
open Kuiper.Poly.GEMMGPU.Type

inline_for_extraction noextract
val mmcomb_gpu : block_tiled_tc_matmulcomb_gpu_ty
