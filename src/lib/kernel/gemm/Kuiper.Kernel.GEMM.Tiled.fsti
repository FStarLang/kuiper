module Kuiper.Kernel.GEMM.Tiled

#lang-pulse

open Kuiper
open Kuiper.Kernel.GEMMGPU.Type
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Array2 { array2 }
open Kuiper.EMatrix
module MS = Kuiper.Spec.GEMM

inline_for_extraction noextract
let size_req : tiled_size_req_t =
  fun m n k tile ->
    m * n <= max_blocks

(* Approximate tiled GEMM: result matrix approximates MS.mmcomb over
   external real matrices rA, rB, rC related by %~ to eA, eB, eC. *)
inline_for_extraction noextract
val mmcomb_gpu_approx : tiled_matmulcomb_gpu_approx_ty size_req
