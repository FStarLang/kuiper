module Kuiper.Poly.MatMul.Tiled.SHMem

#lang-pulse

open Kuiper
open Kuiper.Poly.MatMulGPU.Type

inline_for_extraction noextract
val matmul_gpu : tiled_matmul_gpu_ty
