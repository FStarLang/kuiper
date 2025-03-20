module Kuiper.Poly.GEMMGPU2D.Type

#lang-pulse

open Kuiper
open Kuiper.EMatrix { ematrix, matrix_comb }
open Kuiper.EMatrix4 { ematrix4 }
open Kuiper.EMatrix6 { ematrix6 }
open Kuiper.Matrix.Reprs.Type
module MS = Kuiper.Spec.GEMM
module M  = Kuiper.Matrix
module M4 = Kuiper.Matrix4
module M6 = Kuiper.Matrix6

(* Clearly, this depends on the algorithm involved and the GPU we
   we're working with. For now, just use this definition. *)
type valid_tile = tile:szp{tile * tile <= max_threads}

unfold
inline_for_extraction
type matmulcomb_gpu_fixed_ty
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows : szp)
  (#shared : szp)
  (#cols : szp)
  (lA : mlayout rows shared)
  (lB : mlayout shared cols)
  (lC : mlayout rows cols)
=
  {| clayout lA |} ->
  {| clayout lB |} ->
  {| clayout lC |} ->
  (gA : M.gpu_matrix et lA) ->
  (gB : M.gpu_matrix et lB) ->
  (gC : M.gpu_matrix et lC) ->
  (#eA : ematrix et rows shared) ->
  (#eB : ematrix et shared cols) ->
  (#eC : ematrix et rows cols) ->
  (* This has a preserves. *)
  stt unit
    (requires
      (cpu ** (gA |-> eA) ** (gB |-> eB)) **
      (pure (rows * cols <= max_blocks) **
       (gC |-> eC)))
    (ensures fun _ ->
      (cpu ** (gA |-> eA) ** (gB |-> eB)) **
      (gC |-> MS.mmcomb comb eC eA eB))

unfold
inline_for_extraction
type matmulcomb_gpu_ty =
  (#et : Type0) -> {| scalar et |} ->
  (comb : binop et) ->
  (#rows : szp) ->
  (#shared : szp) ->
  (#cols : szp) ->
  (#lA : mlayout rows shared) ->
  (#lB : mlayout shared cols) ->
  (#lC : mlayout rows cols) ->
  matmulcomb_gpu_fixed_ty comb lA lB lC

(* The type of GPU-side matmuls that only work over already
tiled matrices. *)
unfold
inline_for_extraction
type tiled_matmulcomb_gpu_ty =
  (tile : valid_tile) ->
  (#et : Type0) -> {| scalar et |} ->
  (comb : binop et) ->
  (#mrows : szp) ->
  (#mshared : szp) ->
  (#mcols : szp) ->
  (#trows : szp) ->
  (#tcols : szp) ->
  (#browscols : szp { browscols <= 32 /\ browscols == trows /\ browscols == tcols }) ->
  (lA : M4.mlayout4 mrows   mshared (tile * trows) (tile * tcols)) ->
  (lB : M4.mlayout4 mshared mcols   (tile * trows) (tile * tcols)) ->
  (lC : M6.mlayout6 mrows   mcols   tile tile trows tcols) ->
  {| M4.clayout4 lA |} ->
  {| M4.clayout4 lB |} ->
  {| M6.clayout6 lC |} ->
  (gA : M4.gpu_matrix et lA) ->
  (gB : M4.gpu_matrix et lB) ->
  (gC : M6.gpu_matrix et lC) ->
  (#eA : ematrix4 et mrows   mshared (tile * trows) (tile * tcols)) ->
  (#eB : ematrix4 et mshared mcols   (tile * trows) (tile * tcols)) ->
  (#eC : ematrix6 et mrows   mcols   tile tile trows tcols) ->
  stt unit
    (requires
      (cpu ** (gA |-> eA) ** (gB |-> eB)) **
      (pure (mrows * mcols <= max_blocks) **
       pure (tile * tile <= max_threads) **
       (gC |-> eC)))
    (ensures fun _ ->
      (cpu ** (gA |-> eA) ** (gB |-> eB)) **
      (exists* eC'. (gC |-> eC')))
