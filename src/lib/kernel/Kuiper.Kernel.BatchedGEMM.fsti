module Kuiper.Kernel.BatchedGEMM

(* Batched matrix multiplication (GEMM) using 3D tensors
  (batch, M, K) and (batch, K, N) to produce (batch, M, N). *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg
open Kuiper.Shape
module EMatrix3 = Kuiper.EMatrix3
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT

inline_for_extraction noextract
fn bmmcomb_gpu_exact
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (batch rows shared cols : szp)
  (#la : tlayout (batch @| rows @| shared @| INil))
  (#lb : tlayout (batch @| shared @| cols @| INil))
  (#lc : tlayout (batch @| rows @| cols @| INil))
  {| ctlayout la, ctlayout lb, ctlayout lc |}
  (a : tensor et la { is_global a })
  (b : tensor et lb { is_global b })
  (c : tensor et lc { is_global c })
  (#sa : erased (EMatrix3.t et batch rows shared))
  (#sb : erased (EMatrix3.t et batch shared cols))
  (#sc : erased (EMatrix3.t et batch rows cols))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (a |-> Frac fA sa) **
    on gpu_loc (b |-> Frac fB sb)
  requires
    pure (
      rows * cols <= max_blocks * max_threads  /\
      SZ.fits (batch * rows * cols)
    ) **
    on gpu_loc (c |-> sc)
  ensures
    on gpu_loc (c |-> MS.bmmcomb comb sc sa sb)
