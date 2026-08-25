module Kuiper.Kernel.GEMM.BlockTiling1D

#lang-pulse

open Kuiper
open Kuiper.Kernel.GEMMGPU.Type
open Kuiper.Chest { chest2, chest3 }

module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Kernel.GEMM.Util
module T = Kuiper.Tensor

inline_for_extraction noextract
let size_req : tiled_size_req_t =
  fun m n k tile -> m * n <= max_blocks

(* Batched size requirement: all [batch * m * n] blocks must fit. *)
inline_for_extraction noextract
let bsize_req (batch m n k tile : nat) : prop = batch * (m * n) <= max_blocks

(* General (fused-map, multi-type) natively batched approximate 1D block-tiled
   GEMM: a single launch computes [batch] independent GEMMs; the result rank-3
   tensor approximates [MS.gbmmcomb] over external real chests. *)
inline_for_extraction noextract
fn gbmmcomb_gpu_approx
  (tile : valid_tile)
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #m #n #k : szp)
  (#lA : T.layout3 batch (m * tile) (k * tile))
  (#lB : T.layout3 batch (k * tile) (n * tile))
  (#lC : T.layout3 batch (m * tile) (n * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : T.array3 ta lA { T.is_global gA })
  (gB : T.array3 tb lB { T.is_global gB })
  (gC : T.array3 tc lC { T.is_global gC })
  (rA  : chest3 real batch (m * tile) (k * tile))
  (rB  : chest3 real batch (k * tile) (n * tile))
  (rC  : chest3 real batch (m * tile) (n * tile))
  (#eA : chest3 ta batch (m * tile) (k * tile))
  (#eB : chest3 tb batch (k * tile) (n * tile))
  (#eC : chest3 tc batch (m * tile) (n * tile))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r) **
    pure (bsize_req batch m n k tile) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : chest3 tc batch (m * tile) (n * tile)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.gbmmcomb mapA_r mapB_r comb_r rC rA rB))

(* Natively batched approximate 1D block-tiled GEMM: a single launch computes
   [batch] independent GEMMs; the result rank-3 tensor approximates
   MS.bmmcomb over external real chests rA, rB, rC. *)
inline_for_extraction noextract
fn bmmcomb_gpu_approx
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #m #n #k : szp)
  (#lA : T.layout3 batch (m * tile) (k * tile))
  (#lB : T.layout3 batch (k * tile) (n * tile))
  (#lC : T.layout3 batch (m * tile) (n * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : T.array3 et lA { T.is_global gA })
  (gB : T.array3 et lB { T.is_global gB })
  (gC : T.array3 et lC { T.is_global gC })
  (rA  : chest3 real batch (m * tile) (k * tile))
  (rB  : chest3 real batch (k * tile) (n * tile))
  (rC  : chest3 real batch (m * tile) (n * tile))
  (#eA : chest3 et batch (m * tile) (k * tile))
  (#eB : chest3 et batch (k * tile) (n * tile))
  (#eC : chest3 et batch (m * tile) (n * tile))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (bsize_req batch m n k tile) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : chest3 et batch (m * tile) (n * tile)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.bmmcomb comb_r rC rA rB))

(* General (fused-map, multi-type) approximate 1D block-tiled GEMM.  Derived
   from [gbmmcomb_gpu_approx] at [batch = 1] via a single-page (rank-3) relayout
   of the rank-2 matrices, so there is a single kernel description in this
   module (the batched one). *)
inline_for_extraction noextract
fn gmmcomb_gpu_approx
  (tile : valid_tile)
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (#lA : T.layout2 (m * tile) (k * tile))
  (#lB : T.layout2 (k * tile) (n * tile))
  (#lC : T.layout2 (m * tile) (n * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : T.array2 ta lA { T.is_global gA })
  (gB : T.array2 tb lB { T.is_global gB })
  (gC : T.array2 tc lC { T.is_global gC })
  (rA  : chest2 real (m * tile) (k * tile))
  (rB  : chest2 real (k * tile) (n * tile))
  (rC  : chest2 real (m * tile) (n * tile))
  (#eA : chest2 ta (m * tile) (k * tile))
  (#eB : chest2 tb (k * tile) (n * tile))
  (#eC : chest2 tc (m * tile) (n * tile))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r) **
    pure (m * n <= max_blocks) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : chest2 tc (m * tile) (n * tile)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.gmmcomb mapA_r mapB_r comb_r rC rA rB))

inline_for_extraction noextract
val mmcomb_gpu_approx : tiled_matmulcomb_gpu_approx_ty size_req
