module Kuiper.Kernel.BatchedGEMM

(* Batched matrix multiplication using Array3.
   Inputs are Array3.t on GPU with l3_batched_row_major layout.
   Functional spec: the output is page-wise the matmul of the input pages,
   i.e. out i j k == matmul (slice_page sa i) (slice_page sb i) j k. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
module EM = Kuiper.EMatrix
module EMatrix3 = Kuiper.EMatrix3
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT

(* Per-page batched matmul spec. *)
let batched_matmul
  (#et:Type) {| scalar et |}
  (#batch #rows #shared #cols : nat)
  (a : EMatrix3.t et batch rows shared)
  (b : EMatrix3.t et batch shared cols)
  : EMatrix3.t et batch rows cols
  = EMatrix3.mkM fun i j k ->
      EM.macc (MS.matmul (EMatrix3.slice_page a i)
                         (EMatrix3.slice_page b i)) j k

(* TODO: Layout polymorphism. Attempt to not use EMatrix3 and just use Chest. *)
inline_for_extraction noextract
fn batched_gemm_f32
  (batch rows shared cols : szp)
  (a : tensor f32 (l3_batched_row_major batch rows shared) { is_global a })
  (b : tensor f32 (l3_batched_row_major batch shared cols) { is_global b })
  (#sa : erased (EMatrix3.t f32 batch rows shared))
  (#sb : erased (EMatrix3.t f32 batch shared cols))
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
    )
  returns
    out : tensor f32 (l3_batched_row_major batch rows cols)
  ensures
    on gpu_loc (out |-> batched_matmul sa sb) **
    pure (is_global out)
