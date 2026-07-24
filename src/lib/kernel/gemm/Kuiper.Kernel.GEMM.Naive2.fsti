module Kuiper.Kernel.GEMM.Naive2

(* Like Naive, but spawns full blocks of threads going in row-major order
through the output matrix, with each thread computing a full dot product.
Tensor analog of Naive2. *)

#lang-pulse

open Kuiper

module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
open Kuiper.Tensor
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Kernel.GEMMGPU.Type

inline_for_extraction noextract
let size_req : size_req_t =
  fun m n k -> m * n <= max_blocks * max_threads

(* Batched size requirement: all [batch * m * n] threads (one per output
   cell of every page) must fit in the available blocks*threads. *)
inline_for_extraction noextract
let bsize_req (batch m n k: nat) : prop =
  SZ.fits (batch * (m * n)) /\
  batch * (m * n) <= max_blocks * max_threads

(* General (4-type, fused-map) batched rank-3 kernel descriptor. *)
inline_for_extraction noextract
val gbkdesc
  (#ta #tb #tc #tacc : Type0) {| scalar tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#b #m #n #k : szp)
  (#lA : layout3 b m k)
  (#lB : layout3 b k n)
  (#lC : layout3 b m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor ta lA { is_global gA })
  (gB : tensor tb lB { is_global gB })
  (gC : tensor tc lC { is_global gC })
  (#eA : chest3 ta b m k)
  (#eB : chest3 tb b k n)
  (#eC : chest3 tc b m n)
  (#fA #fB : perm)
  (#_ : squash (bsize_req b m n k))
  : kernel_desc
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.gbmmcomb mapA mapB comb eC eA eB)

(* Batched rank-3 kernel descriptor, exposed so callers can launch it
   directly (e.g. asynchronously). *)
inline_for_extraction noextract
val bkdesc
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#b #m #n #k : szp)
  (#lA : layout3 b m k)
  (#lB : layout3 b k n)
  (#lC : layout3 b m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA { is_global gA })
  (gB : tensor et lB { is_global gB })
  (gC : tensor et lC { is_global gC })
  (#eA : chest3 et b m k)
  (#eB : chest3 et b k n)
  (#eC : chest3 et b m n)
  (#fA #fB : perm)
  (#_ : squash (bsize_req b m n k))
  : kernel_desc
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.bmmcomb comb eC eA eB)

(* General (4-type, fused-map) batched rank-3 GEMM. *)
inline_for_extraction noextract
fn gbmmcomb_gpu_exact
  (#ta #tb #tc #tacc : Type0) {| scalar tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (batch m n k: szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (a : tensor ta lA { is_global a })
  (b : tensor tb lB { is_global b })
  (c : tensor tc lC { is_global c })
  (#eA : chest3 ta batch _ _)
  (#eB : chest3 tb batch _ _)
  (#eC : chest3 tc batch _ _)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (a |-> Frac fA eA ** b |-> Frac fB eB)
  requires
    pure (bsize_req batch m n k) **
    on gpu_loc (c |-> eC)
  ensures
    on gpu_loc (c |-> MS.gbmmcomb mapA mapB comb eC eA eB)

(* Batched rank-3 GEMM: a single launch spawns [batch * m * n] threads
   (via [kernel_desc_n]), each computing one output cell of one page with
   a full dot product. *)
inline_for_extraction noextract
fn bmmcomb_gpu_exact
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (batch m n k: szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (a : tensor et lA { is_global a })
  (b : tensor et lB { is_global b })
  (c : tensor et lC { is_global c })
  (#eA #eB #eC : chest3 et batch _ _)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (a |-> Frac fA eA ** b |-> Frac fB eB)
  requires
    pure (bsize_req batch m n k) **
    on gpu_loc (c |-> eC)
  ensures
    on gpu_loc (c |-> MS.bmmcomb comb eC eA eB)

(* General (4-type, fused-map) rank-2 kernel descriptor. *)
inline_for_extraction noextract
val gkdesc
  (#ta #tb #tc #tacc : Type0) {| scalar tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor ta lA { is_global gA })
  (gB : tensor tb lB { is_global gB })
  (gC : tensor tc lC { is_global gC })
  (#eA : chest2 ta _ _)
  (#eB : chest2 tb _ _)
  (#eC : chest2 tc _ _)
  (#fA #fB : perm)
  (#_ : squash (size_req m n k))
  : kernel_desc
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.gmmcomb mapA mapB comb eC eA eB)

(* Rank-2 kernel descriptor, derived from [bkdesc] at batch one and
   exposed so callers can launch it directly (e.g. asynchronously, as in
   [Kuiper.Example.Async.GEMM]). *)
inline_for_extraction noextract
val kdesc
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA { is_global gA })
  (gB : tensor et lB { is_global gB })
  (gC : tensor et lC { is_global gC })
  (#eA #eB #eC : chest2 et _ _)
  (#fA #fB : perm)
  (#_ : squash (size_req m n k))
  : kernel_desc
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.mmcomb comb eC eA eB)

(* General (4-type, fused-map) rank-2 GEMM. *)
inline_for_extraction noextract
fn gmmcomb_gpu_exact
  (#ta #tb #tc #tacc : Type0) {| scalar tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor ta lA { is_global gA })
  (gB : tensor tb lB { is_global gB })
  (gC : tensor tc lC { is_global gC })
  (#eA : chest2 ta _ _)
  (#eB : chest2 tb _ _)
  (#eC : chest2 tc _ _)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (size_req m n k) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.gmmcomb mapA mapB comb eC eA eB)

(* Rank-2 GEMM: a single launch spawns [m * n] threads (via
   [kernel_desc_n]), each computing one output cell with a full dot
   product.  Derived from [bmmcomb_gpu_exact] at batch one. *)
inline_for_extraction noextract
fn mmcomb_gpu_exact
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA { is_global gA })
  (gB : tensor et lB { is_global gB })
  (gC : tensor et lC { is_global gC })
  (#eA #eB #eC : chest2 et _ _)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (size_req m n k) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.mmcomb comb eC eA eB)
