module Kuiper.Poly.GEMM.OrigBlockTiling1D

#lang-pulse

open Kuiper
open Kuiper.Poly.GEMMGPU.Type

inline_for_extraction noextract
val mmcomb_gpu : block_tiled1d_matmulcomb_gpu_ty
