module Old.Kuiper.Poly.GEMM.SHMem

#lang-pulse

open Kuiper
open Kuiper.Poly.GEMMGPU.Type

inline_for_extraction noextract
val mmcomb_gpu : tiled_matmulcomb_gpu_ty
