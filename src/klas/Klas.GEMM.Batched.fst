module Klas.GEMM.Batched

#lang-pulse
open Kuiper
module K = Kuiper.Kernel.BatchedGEMM
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT

inline_for_extraction noextract
fn batched_matmul
  (#et : Type0) {| scalar et |}
  (batch rows shared cols : szp)
  (a : tensor et (l3_batched_row_major batch rows shared) { is_global a })
  (b : tensor et (l3_batched_row_major batch shared cols) { is_global b })
  (#sa : erased (chest3 et batch rows shared))
  (#sb : erased (chest3 et batch shared cols))
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
    out : tensor et (l3_batched_row_major batch rows cols)
  ensures
    on gpu_loc (out |-> MS.batched_matmul sa sb) **
    pure (is_global out)
{
  let out = alloc0 #et (batch *^ rows *^ cols) (l3_batched_row_major batch rows cols);
  with sc0. assert on gpu_loc (out |-> sc0);
  K.bmmcomb_gpu_exact #et (MS.comb2) batch rows shared cols a b out #sa #sb #sc0 #fA #fB;
  out
}

inline_for_extraction noextract
let batched_gemm
  (#et : Type0) {| scalar et |}
  (alpha beta : et)
  (batch rows shared cols : szp {SZ.fits (batch * rows * shared) /\ SZ.fits (batch * shared * cols) /\ SZ.fits (batch * rows * cols)})
  =
    K.bmmcomb_gpu_exact #et #_ (MS.lincomb alpha beta) batch rows shared cols
    #(l3_batched_row_major batch rows shared)
    #(l3_batched_row_major batch shared cols)
    #(l3_batched_row_major batch rows cols)

let batched_gemm_f32 = batched_gemm #f32

let batched_matmul_f32 = batched_matmul #f32
