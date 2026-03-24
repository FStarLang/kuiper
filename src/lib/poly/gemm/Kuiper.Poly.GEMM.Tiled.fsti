module Kuiper.Poly.GEMM.Tiled

#lang-pulse

open Kuiper
open Kuiper.Approximates
open Kuiper.Poly.GEMMGPU.Type

inline_for_extraction noextract
let size_req : tiled_size_req_t =
  fun mrows mshared mcols tile ->
    mrows * mcols <= max_blocks /\
    tile * tile <= max_threads

(* Approximate tiled GEMM: result matrix approximates MS.mmcomb over
   external real matrices rA, rB, rC related by %~ to eA, eB, eC. *)
inline_for_extraction noextract
val mmcomb_gpu_approx : tiled_matmulcomb_gpu_approx_ty size_req
