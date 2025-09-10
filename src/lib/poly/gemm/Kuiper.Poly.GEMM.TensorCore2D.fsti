module Kuiper.Poly.GEMM.TensorCore2D

#lang-pulse

open Kuiper
open Kuiper.Poly.GEMMGPU.Type

inline_for_extraction noextract
val mmcomb_gpu : block_tiled2d_matmulcomb_gpu_ty
