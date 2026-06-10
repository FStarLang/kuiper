module Kuiper.Kernel.Attention

#lang-pulse
open Kuiper
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg
open Kuiper.Index
open Kuiper.Real
module A3 = Kuiper.Array3
module A2 = Kuiper.Array2
module EM3 = Kuiper.EMatrix3
open Kuiper.EMatrix
module SZ = Kuiper.SizeT

module MS = Kuiper.Spec.GEMM

open Kuiper.Kernel.BatchedGEMM
open Kuiper.Kernel.RowSoftmax

fn scaled_dot_product_efficient_attention (#et : Type0) {| scalar et, floating et, real_like et |}
  (b h : szp)
  (m n : szp)
  (k kv : szp)
  (q    : A4.t et (l4_batched_row_major b h m  k ) { A4.is_global q    })
  (k_   : A4.t et (l4_batched_row_major b h n  k ) { A4.is_global k_   })
  (v    : A4.t et (l4_batched_row_major b h n  kv) { A4.is_global v    })
  (bias : A4.t et (l4_batched_row_major b h m  n ) { A4.is_global bias })
  (scale : et)
  (#sQ : erased (EM4.t et b h m k))
  (#sK : erased (EM4.t et b h n k))
  (#sV : erased (EM4.t et b h n kv))
  (#sB : erased (EM4.t et b h m n))
  (#rKT : erased (EM4.t real b h k n))
  (#fQ #fK #fV #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (q    |-> Frac fQ sQ) **
    on gpu_loc (k_   |-> Frac fK sK) **
    on gpu_loc (v    |-> Frac fV sV) **
    on gpu_loc (bias |-> Frac fB sB)
  requires
    pure (
      SZ.fits (b * h * m * kv) /\
      SZ.fits (b * h * m * n)  /\
      SZ.fits (b * h * n * k)  /\
      SZ.fits (b * h * m * k)  /\
      SZ.fits (b * h * m) /\
      (EM4.mkM (fun i j k l -> EM4.macc sK i j l k)) %~ rKT
    )
  returns
    out : A4.t et (l4_batched_row_major b h m kv) &
          A3.t et (l3_batched_row_major b h m)
  ensures
    (exists* (sO : EM4.t et b h m kv) (sL : EM3.t et b h m).
      on gpu_loc (fst out |-> sO) **
      on gpu_loc (snd out |-> sL) **
      pure (
        let attn_tile = fun i j -> attention_real
            (EM4.slice_page (EM4.to_real_matrix sQ) i j)
            (EM4.slice_page rKT i j)
            (EM4.slice_page (EM4.to_real_matrix sV) i j)
            (EM4.slice_page (EM4.to_real_matrix sB) i j)
            (to_real scale) in
        let out_spec = EM4.mkM fun i j -> macc (fst (attn_tile i j)) in 
        let lse_spec = EM3.mkM fun i j -> Seq.index (snd (attn_tile i j)) in 
        sO %~ out_spec /\ sL %~ lse_spec)) **
    pure (A4.is_global (fst out) /\ A3.is_global (snd out)) {
    admit ()
  }