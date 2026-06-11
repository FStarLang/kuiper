module Kuiper.Kernel.BatchedGEMM

(* Batched matrix multiplication using Array3.
   Inputs are Array3.t on GPU with l3_batched_row_major layout.
   Functional spec: the output is page-wise the matmul of the input pages,
   i.e. out i j k == matmul (slice_page sa i) (slice_page sb i) j k. *)

#lang-pulse
open Kuiper
module Array3 = Kuiper.Array3
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor
module EM = Kuiper.EMatrix
module EMatrix3 = Kuiper.EMatrix3
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT

(* TODO: Attempt to not use EMatrix3 and just use Chest. *)
inline_for_extraction noextract
fn bmmcomb_gpu_exact
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (batch rows shared cols : szp)
  (#la : Array3.layout batch rows shared)
  (#lb : Array3.layout batch shared cols)
  (#lc : Array3.layout batch rows cols)
  {| ctlayout la, ctlayout lb, ctlayout lc |}
  (a : Array3.t et la { Array3.is_global a })
  (b : Array3.t et lb { Array3.is_global b })
  (c : Array3.t et lc { Array3.is_global c })
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
      rows * cols <= max_blocks * max_threads /\
      SZ.fits (batch * rows * cols)
    ) ** 
    on gpu_loc (c |-> sc)
  ensures
    on gpu_loc (c |-> MS.bmmcomb comb sc sa sb)