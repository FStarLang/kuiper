module Kuiper.Poly.GEMMGPU.Type

#lang-pulse

open Kuiper
open Kuiper.Approximates { real_like, approx2, (%~) }
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix { ematrix, matrix_comb }
open Kuiper.Matrix.Reprs.Type
module MS = Kuiper.Spec.GEMM
module M  = Kuiper.Matrix

(* Clearly, this depends on the algorithm involved and the GPU we
   we're working with. For now, just use this definition. *)
type valid_tile = tile:szp{tile * tile <= max_threads}

(* Maybe make this szp -> szp -> szp -> bool? *)
inline_for_extraction noextract
type size_req_t = nat -> nat -> nat -> prop
inline_for_extraction noextract
type tiled_size_req_t = nat -> nat -> nat -> nat ->prop

unfold
inline_for_extraction
type matmulcomb_gpu_fixed_ty
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows : szp)
  (#shared : szp)
  (#cols : szp)
  (lA : full_mlayout rows shared)
  (lB : full_mlayout shared cols)
  (lC : full_mlayout rows cols)
  (size_req : prop)
=
  {| clayout lA |} ->
  {| clayout lB |} ->
  {| clayout lC |} ->
  (gA : M.gpu_matrix et lA { M.is_global_matrix gA }) ->
  (#fA : perm) ->
  (gB : M.gpu_matrix et lB { M.is_global_matrix gB }) ->
  (#fB : perm) ->
  (gC : M.gpu_matrix et lC { M.is_global_matrix gC }) ->
  (#eA : ematrix et rows shared) ->
  (#eB : ematrix et shared cols) ->
  (#eC : ematrix et rows cols) ->
  (* This has a preserves. *)
  stt unit
    (requires
      (cpu ** on gpu_loc (gA |-> Frac fA eA) ** on gpu_loc (gB |-> Frac fB eB)) **
      (pure size_req **
       on gpu_loc (gC |-> eC)))
    (ensures fun _ ->
      (cpu ** on gpu_loc (gA |-> Frac fA eA) ** on gpu_loc (gB |-> Frac fB eB)) **
      (on gpu_loc (gC |-> MS.mmcomb comb eC eA eB)))

unfold
inline_for_extraction
type matmulcomb_gpu_ty
  (size_req : size_req_t)
=
  (#et : Type0) -> {| scalar et |} ->
  (comb : binop et) ->
  (#rows : szp) ->
  (#shared : szp) ->
  (#cols : szp) ->
  (#lA : full_mlayout rows shared) ->
  (#lB : full_mlayout shared cols) ->
  (#lC : full_mlayout rows cols) ->
  matmulcomb_gpu_fixed_ty comb lA lB lC
    (size_req rows shared cols)

unfold
inline_for_extraction
type matmulcomb_gpu_approx_ty
  (size_req : size_req_t)
=
  (#et : Type0) -> {| scalar et |} -> {| real_like et |} ->
  (comb : binop et) ->
  (comb_r : binop real { approx2 comb comb_r }) ->
  (#rows : szp) ->
  (#shared : szp) ->
  (#cols : szp) ->
  (#lA : full_mlayout rows shared) ->
  (#lB : full_mlayout shared cols) ->
  (#lC : full_mlayout rows cols) ->
  {| clayout lA |} ->
  {| clayout lB |} ->
  {| clayout lC |} ->
  (gA : M.gpu_matrix et lA { M.is_global_matrix gA }) ->
  (#fA : perm) ->
  (gB : M.gpu_matrix et lB { M.is_global_matrix gB }) ->
  (#fB : perm) ->
  (gC : M.gpu_matrix et lC { M.is_global_matrix gC }) ->
  (#eA : ematrix et rows shared) ->
  (#eB : ematrix et shared cols) ->
  (#eC : ematrix et rows cols) ->
  (rA : ematrix real rows shared) ->
  (rB : ematrix real shared cols) ->
  (rC : ematrix real rows cols) ->
  stt unit
    (requires
      (cpu ** on gpu_loc (gA |-> Frac fA eA) ** on gpu_loc (gB |-> Frac fB eB)) **
      (pure (size_req rows shared cols) **
       pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
       on gpu_loc (gC |-> eC)))
    (ensures fun _ ->
      (cpu ** on gpu_loc (gA |-> Frac fA eA) ** on gpu_loc (gB |-> Frac fB eB)) **
      (exists* (eC' : ematrix et rows cols).
        on gpu_loc (gC |-> eC') **
        pure (eC' %~ MS.mmcomb comb_r rC rA rB)))

(* The type of GPU-side matmuls that only work over already
tiled matrices. *)
unfold
inline_for_extraction
type tiled_matmulcomb_gpu_ty
  (size_req : tiled_size_req_t)
=
  (tile : valid_tile) ->
  (#et : Type0) -> {| scalar et |} ->
  (comb : binop et) ->
  (#mrows : szp) ->
  (#mshared : szp) ->
  (#mcols : szp) ->
  (lA : mlayout (mrows   * tile) (mshared * tile)) ->
  (lB : mlayout (mshared * tile) (mcols   * tile)) ->
  (lC : mlayout (mrows   * tile) (mcols   * tile)) ->
  {| clayout lA |} ->
  {| clayout lB |} ->
  {| clayout lC |} ->
  (gA : M.gpu_matrix et lA { M.is_global_matrix gA }) ->
  (#fA : perm) ->
  (gB : M.gpu_matrix et lB { M.is_global_matrix gB }) ->
  (#fB : perm) ->
  (gC : M.gpu_matrix et lC { M.is_global_matrix gC }) ->
  (#eA : ematrix _ _ _) ->
  (#eB : ematrix _ _ _) ->
  (#eC : ematrix _ _ _) ->
  stt unit
    (requires
      (cpu ** on gpu_loc (gA |-> Frac fA eA) ** on gpu_loc (gB |-> Frac fB eB)) **
      (pure (size_req mrows mshared mcols tile) **
       on gpu_loc (gC |-> eC)))
    (ensures fun _ ->
      (cpu ** on gpu_loc (gA |-> Frac fA eA) ** on gpu_loc (gB |-> Frac fB eB)) **
      (on gpu_loc (gC |-> MS.mmcomb comb eC eA eB)))

(* The type of GPU-side approximate matmuls over tiled matrices.
   Takes external real matrices rA, rB, rC related by %~ to eA, eB, eC,
   and a real-valued combiner comb_r approximating comb. *)
unfold
inline_for_extraction
type tiled_matmulcomb_gpu_approx_ty
  (size_req : tiled_size_req_t)
=
  (tile : valid_tile) ->
  (#et : Type0) -> {| scalar et |} -> {| real_like et |} ->
  (comb : binop et) ->
  (comb_r : binop real { approx2 comb comb_r }) ->
  (#mrows : szp) ->
  (#mshared : szp) ->
  (#mcols : szp) ->
  (lA : mlayout (mrows   * tile) (mshared * tile)) ->
  (lB : mlayout (mshared * tile) (mcols   * tile)) ->
  (lC : mlayout (mrows   * tile) (mcols   * tile)) ->
  {| clayout lA |} ->
  {| clayout lB |} ->
  {| clayout lC |} ->
  (gA : M.gpu_matrix et lA { M.is_global_matrix gA }) ->
  (#fA : perm) ->
  (gB : M.gpu_matrix et lB { M.is_global_matrix gB }) ->
  (#fB : perm) ->
  (gC : M.gpu_matrix et lC { M.is_global_matrix gC }) ->
  (#eA : ematrix et (mrows * tile) (mshared * tile)) ->
  (#eB : ematrix et (mshared * tile) (mcols * tile)) ->
  (#eC : ematrix et (mrows * tile) (mcols * tile)) ->
  (rA : ematrix real (mrows * tile) (mshared * tile)) ->
  (rB : ematrix real (mshared * tile) (mcols * tile)) ->
  (rC : ematrix real (mrows * tile) (mcols * tile)) ->
  stt unit
    (requires
      (cpu ** on gpu_loc (gA |-> Frac fA eA) ** on gpu_loc (gB |-> Frac fB eB)) **
      (pure (size_req mrows mshared mcols tile) **
       pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
       on gpu_loc (gC |-> eC)))
    (ensures fun _ ->
      (cpu ** on gpu_loc (gA |-> Frac fA eA) ** on gpu_loc (gB |-> Frac fB eB)) **
      (exists* (eC' : ematrix et (mrows * tile) (mcols * tile)).
        on gpu_loc (gC |-> eC') **
        pure (eC' %~ MS.mmcomb comb_r rC rA rB)))
