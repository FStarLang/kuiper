module Kuiper.Poly.GEMM.Tiled

#lang-pulse

open Kuiper
open Kuiper.Poly.GEMMGPU.Type

inline_for_extraction noextract
val matmul_gpu : tiled_matmul_gpu_ty
