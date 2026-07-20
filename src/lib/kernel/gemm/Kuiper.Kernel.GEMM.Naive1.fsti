module Kuiper.Kernel.GEMM.Naive1

(* Native batched matrix multiplication (GEMM) over rank-3 tensors:
   (batch, m, k) and (batch, k, n) to produce (batch, m, n).

   Like Naive2, but batched: a *single* kernel launch spawns
   [batch * m * n] independent blocks (one thread each, via
   [kernel_desc_m_1]), each computing one output cell of one page with a
   full dot product. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Shape
open Kuiper.Chest
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT

(* Batched size requirement: all [batch * m * n] blocks must fit in the
   available blocks (one block per output cell, one thread per block). *)
inline_for_extraction noextract
let bsize_req (batch m n k: nat) : prop =
  SZ.fits (batch * (m * n)) /\
  batch * (m * n) <= max_blocks

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
