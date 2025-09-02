module Kuiper.GEMM.OrigBlockTiling1D

#lang-pulse
open Kuiper
open Kuiper.Poly.GEMMCPU
open Kuiper.Matrix.Reprs { row_major as RM, col_major as CM }

(* NOTE: no dynamic tile version as that would imply the use of a
   stack-allocated variable-length array, and NVCC complains
   (karamel warns too). *)

inline_for_extraction noextract
let size_req tile : size_req_t =
  fun rows shared cols ->
    (rows / tile) * (cols / tile) <= max_blocks

val matmul_f32_tiles64x8_8x64_rc8_rrr : fixed_repr_matmul_cpu_ty f32 (size_req 64) RM RM RM
// This combination of tile sizes is forbidden!
//  The generated code contains dynamic guards that immediately fail.
// val matmul_f32_tiles32x32_32x32_rc32_rrr : fixed_repr_matmul_cpu_ty f32 (size_req 32) RM RM RM
