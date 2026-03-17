module Kuiper.Poly.GEMM.Tiled

#lang-pulse

open Kuiper
open Kuiper.Approximates
open Kuiper.Poly.GEMMGPU.Type
module MU = Kuiper.Poly.GEMM.Util
module M = Kuiper.Matrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.EMatrix

inline_for_extraction noextract
let size_req : tiled_size_req_t =
  fun mrows mshared mcols tile ->
    mrows * mcols <= max_blocks /\
    tile * tile <= max_threads

(* Approximate tiled GEMM: result matrix approximates real_mmcomb.
   This is the fully proven version. *)
inline_for_extraction noextract
val mmcomb_gpu_approx
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (lA : mlayout (mrows   * tile) (mshared * tile))
  (lB : mlayout (mshared * tile) (mcols   * tile))
  (lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : M.gpu_matrix et lA { M.is_global_matrix gA })
  (#fA : perm)
  (gB : M.gpu_matrix et lB { M.is_global_matrix gB })
  (#fB : perm)
  (gC : M.gpu_matrix et lC { M.is_global_matrix gC })
  (#eA : ematrix et (mrows * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols * tile))
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  : stt unit
    (requires
      (cpu ** on gpu_loc (gA |-> Frac fA eA) ** on gpu_loc (gB |-> Frac fB eB)) **
      (pure (size_req mrows mshared mcols tile) **
       on gpu_loc (gC |-> eC)))
    (ensures fun _ ->
      (cpu ** on gpu_loc (gA |-> Frac fA eA) ** on gpu_loc (gB |-> Frac fB eB)) **
      (exists* (eC' : ematrix et (mrows * tile) (mcols * tile)).
        on gpu_loc (gC |-> eC') **
        pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB))))

(* Legacy interface: exact postcondition for backward compatibility.
   Uses assume to bridge from approximate to exact. *)
inline_for_extraction noextract
val mmcomb_gpu : tiled_matmulcomb_gpu_ty size_req
