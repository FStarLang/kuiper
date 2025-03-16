module Kuiper.Poly.MatMul.Tiled.SHMem

#lang-pulse

open Kuiper
open Kuiper.Poly.MatMulGPU.Type

type valid_tile = tile:szp{tile * tile <= max_threads}

inline_for_extraction noextract
val matmul_gpu (tile : valid_tile) : matmul_gpu_ty
