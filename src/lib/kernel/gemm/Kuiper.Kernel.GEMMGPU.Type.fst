module Kuiper.Kernel.GEMMGPU.Type

#lang-pulse

open Kuiper
open Kuiper.EMatrix { chest2, matrix_comb }
module MS = Kuiper.Spec.GEMM
module T = Kuiper.Tensor

(* Clearly, this depends on the algorithm involved and the GPU we
   we're working with. For now, just use this definition. *)
type valid_tile = tile:szp{tile * tile <= max_threads}

(* Maybe make this szp -> szp -> szp -> bool? *)
inline_for_extraction noextract
type size_req_t = m:nat -> n:nat -> k:nat -> prop
inline_for_extraction noextract
type tiled_size_req_t = m:nat -> n:nat -> k:nat -> tile:nat -> prop

// TODO: check if both unfold + inline_for_extraction are needed here, or if one suffices

(* Type for a simple GEMM over GPU data. *)
unfold inline_for_extraction
type matmulcomb_gpu_ty
  (size_req : size_req_t)
=
  fn (#et : Type0) {| scalar et |}
     (comb : binop et)
     (#m #n #k : szp)
     (#lA : T.layout2 m k)
     (#lB : T.layout2 k n)
     (#lC : T.layout2 m n)
     {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
     (gA : T.array2 et lA { T.is_global gA })
     (gB : T.array2 et lB { T.is_global gB })
     (gC : T.array2 et lC { T.is_global gC })
     (#eA #eB #eC : chest2 et _ _)
     (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (size_req m n k) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.mmcomb comb eC eA eB)

(* Like above, with approximate spec. *)
unfold inline_for_extraction
type matmulcomb_gpu_approx_ty
  (size_req : size_req_t)
=
  fn (#et : Type0) {| scalar et, real_like et |}
     (comb : binop et)
     (comb_r : binop real { comb `approx2` comb_r })
     (#m #n #k : szp)
     (#lA : T.layout2 m k)
     (#lB : T.layout2 k n)
     (#lC : T.layout2 m n)
     {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
     (gA : T.array2 et lA { T.is_global gA })
     (gB : T.array2 et lB { T.is_global gB })
     (gC : T.array2 et lC { T.is_global gC })
     (#eA #eB #eC : chest2 et _ _)
     (rA rB rC : chest2 real _ _)
     (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (size_req m n k) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : chest2 et m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB))

(* A GEMM over tiled GPU data. *)
unfold inline_for_extraction
type tiled_matmulcomb_gpu_ty
  (size_req : tiled_size_req_t)
=
  fn (tile : valid_tile)
     (#et : Type0) {| scalar et |}
     (comb : binop et)
     (#m #n #k : szp)
     (#lA : T.layout2 (m * tile) (k * tile))
     (#lB : T.layout2 (k * tile) (n * tile))
     (#lC : T.layout2 (m * tile) (n * tile))
     {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
     (gA : T.array2 et lA { T.is_global gA })
     (gB : T.array2 et lB { T.is_global gB })
     (gC : T.array2 et lC { T.is_global gC })
     (#eA #eB #eC : chest2 _ _ _)
     (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (size_req m n k tile) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.mmcomb comb eC eA eB)

(* As above, with approximate spec. *)
unfold inline_for_extraction
type tiled_matmulcomb_gpu_approx_ty
  (size_req : tiled_size_req_t)
=
  fn (tile : valid_tile)
     (#et : Type0) {| scalar et, real_like et |}
     (comb : binop et)
     (comb_r : binop real { comb `approx2` comb_r })
     (#m #n #k : szp)
     (#lA : T.layout2 (m * tile) (k * tile))
     (#lB : T.layout2 (k * tile) (n * tile))
     (#lC : T.layout2 (m * tile) (n * tile))
     {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
     (gA : T.array2 et lA { T.is_global gA })
     (gB : T.array2 et lB { T.is_global gB })
     (gC : T.array2 et lC { T.is_global gC })
     (rA rB rC : chest2 real _ _)
     (#eA #eB #eC : chest2 _ _ _)
     (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (size_req m n k tile) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures (
    (exists* (eC' : chest2 et (m * tile) (n * tile)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB)))
