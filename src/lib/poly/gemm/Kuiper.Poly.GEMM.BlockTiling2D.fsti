module Kuiper.Poly.GEMM.BlockTiling2D

#lang-pulse

open Kuiper
open Kuiper.Poly.GEMMGPU2D.Type

inline_for_extraction noextract
val matmul_gpu : tiled_matmulcomb_gpu_ty
