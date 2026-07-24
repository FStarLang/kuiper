module Kuiper.Kernel.GEMM.Tiled

#lang-pulse

open Kuiper
open Kuiper.Kernel.GEMMGPU.Type
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor
open Kuiper.Chest { chest3, chest2 }
open Kuiper.EMatrix
module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Kernel.GEMM.Util
module SZ = Kuiper.SizeT

inline_for_extraction noextract
let size_req : tiled_size_req_t =
  fun m n k tile ->
    m * n <= max_blocks

(* Batched size requirement: all [batch * m * n] blocks must fit. *)
inline_for_extraction noextract
let bsize_req (batch m n k tile : nat) : prop =
  SZ.fits (batch * (m * n)) /\
  batch * (m * n) <= max_blocks

(* General (fused-map, multi-type) natively batched approximate tiled GEMM:
   a single launch computes [batch] independent tiled GEMMs; the result rank-3
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
  (#lA : layout3 batch (m * tile) (k * tile))
  (#lB : layout3 batch (k * tile) (n * tile))
  (#lC : layout3 batch (m * tile) (n * tile))
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : array3 ta lA { is_global gA })
  (gB : array3 tb lB { is_global gB })
  (gC : array3 tc lC { is_global gC })
  (rA : chest3 real batch (m * tile) (k * tile))
  (rB : chest3 real batch (k * tile) (n * tile))
  (rC : chest3 real batch (m * tile) (n * tile))
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

(* Natively batched approximate tiled GEMM: a single launch computes
   [batch] independent tiled GEMMs; the result rank-3 tensor approximates
   MS.bmmcomb over external real chests rA, rB, rC. *)
inline_for_extraction noextract
fn bmmcomb_gpu_approx
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #m #n #k : szp)
  (#lA : layout3 batch (m * tile) (k * tile))
  (#lB : layout3 batch (k * tile) (n * tile))
  (#lC : layout3 batch (m * tile) (n * tile))
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : array3 et lA { is_global gA })
  (gB : array3 et lB { is_global gB })
  (gC : array3 et lC { is_global gC })
  (rA : chest3 real batch (m * tile) (k * tile))
  (rB : chest3 real batch (k * tile) (n * tile))
  (rC : chest3 real batch (m * tile) (n * tile))
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

(* General (fused-map, multi-type) approximate tiled GEMM: reads A-cells of type
   [ta] and B-cells of type [tb], maps each into the accumulation type [tacc] via
   [mapA]/[mapB], accumulates, then combines with the old C-cell (type [tc]) via
   [comb].  The result matrix approximates [MS.gmmcomb] over the external real
   matrices, with the real maps [mapA_r]/[mapB_r]/[comb_r] related to the element
   operations by [MU.approx1]/[approx2].

   It is derived from [gbmmcomb_gpu_approx] at [batch = 1] via a single-page
   (rank-3) relayout of the rank-2 matrices, so there is a single kernel
   description in this module (the batched one). *)
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
  (#lA : layout2 (m * tile) (k * tile))
  (#lB : layout2 (k * tile) (n * tile))
  (#lC : layout2 (m * tile) (n * tile))
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : array2 ta lA { is_global gA })
  (gB : array2 tb lB { is_global gB })
  (gC : array2 tc lC { is_global gC })
  (rA : chest2 real (m * tile) (k * tile))
  (rB : chest2 real (k * tile) (n * tile))
  (rC : chest2 real (m * tile) (n * tile))
  (#eA : chest2 ta (m * tile) (k * tile))
  (#eB : chest2 tb (k * tile) (n * tile))
  (#eC : chest2 tc (m * tile) (n * tile))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r) **
    pure (m * n <= max_blocks) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : chest2 tc (m * tile) (n * tile)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.gmmcomb mapA_r mapB_r comb_r rC rA rB))

(* Approximate tiled GEMM: result matrix approximates MS.mmcomb over
   external real matrices rA, rB, rC related by %~ to eA, eB, eC. *)
inline_for_extraction noextract
val mmcomb_gpu_approx : tiled_matmulcomb_gpu_approx_ty size_req
