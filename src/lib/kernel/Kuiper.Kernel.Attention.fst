module Kuiper.Kernel.Attention

#lang-pulse
open Kuiper
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor
open Kuiper.Index
open Kuiper.Bijection
open Kuiper.Real

module CH = Kuiper.Chest

module SZ = Kuiper.SizeT

// module MS = Kuiper.Spec.GEMM
// open Kuiper.Kernel.BatchedGEMM
// open Kuiper.Kernel.RowSoftmax
  
  #set-options "--ifuel 0"
let index_destructure_test (#et : Type0) {| floating et, real_like et, floating_real_like et |} (#b #h : szp)
  (#m #n : szp) 
  (#k : szp)
  (#sK : erased  (CH.t (b @| h @| n @| k @| INil) et))
  (di: abs (b @| h @| k @| n @| INil)): GTot et =
  let (i,(j,(k,(l,())))) = di in
  CH.acc sK (i,(j,(l,(k,()))))

#push-options "--split_queries always --ifuel 4"
fn scaled_dot_product_efficient_attention4
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (b h : szp)
  (m n : szp)
  (k kv : szp)
  (q    : tensor et (l4_batched_row_major b h m  k ) ) //{ A4.is_global q    }
  (k_   : tensor et (l4_batched_row_major b h n  k ) ) //{ A4.is_global k_   }
  (v    : tensor et (l4_batched_row_major b h n  kv) ) //{ A4.is_global v    }
  (bias : tensor et (l4_batched_row_major b h m  n ) ) //{ A4.is_global bias }
  (scale : et)
  (#sQ : erased  (CH.t (b @| h @| m @| k @| INil) et))
  (#sK : erased  (CH.t (b @| h @| n @| k @| INil) et))
  (#sV : erased  (CH.t (b @| h @| n @| kv @| INil) et))
  (#sB : erased  (CH.t (b @| h @| m @| n @| INil) et))
  (#rKT : erased (CH.t (b @| h @| k @| n @| INil) real))
  (#fQ #fK #fV #fB : perm)
  preserves
    cpu **
    (* on gpu_loc *) (q    |-> Frac fQ sQ) **
    (* on gpu_loc *) (k_   |-> Frac fK sK) **
    (* on gpu_loc *) (v    |-> Frac fV sV) **
    (* on gpu_loc *) (bias |-> Frac fB sB)
  requires
    pure (
      SZ.fits (b * h * m * kv) /\
      SZ.fits (b * h * m * n)  /\
      SZ.fits (b * h * n * k)  /\
      SZ.fits (b * h * m * k)  /\
      SZ.fits (b * h * m) /\
      ((CH.mk (b @| h @| k @| n @| INil) 
        (fun (i,(j,(k,(l,())))) -> 
          CH.acc sK (i,(j,(l,(k,())))))
        
        ) %~ (reveal rKT)) /\
      m * n <= max_blocks * max_threads /\
      m * kv <= max_blocks * max_threads /\
      b * h * m <= max_blocks
    )
  returns
    out : tensor et (l4_batched_row_major b h m kv) &
          tensor et (l3_batched_row_major b h m)
  ensures
    (exists* (sO : CH.t (b @| h @| m @| kv @| INil) et) (sL : CH.t (b @| h @| m @| INil) et).
      (* on gpu_loc *) (fst out |-> sO) **
      (* on gpu_loc *) (snd out |-> sL))
      //pure (
      //  let attn_tile = fun i j -> attention_real
      //      (CH.slice_page sQ i j)
      //      (CH.slice_page rKT i j)
      //      (CH.slice_page sV i j)
      //      (CH.slice_page sB i j)
      //      (to_real scale) in
      //  let out_spec = CH.mkM fun i j -> CH.macc (fst (attn_tile i j)) in 
      //  let lse_spec = CH.mkM fun i j -> Seq.index (snd (attn_tile i j)) in 
      //  sO %~ out_spec /\ sL %~ lse_spec))
    (* ** pure (A4.is_global (fst out) /\ A3.is_global (snd out)) *) {

//  map_loc gpu_loc ghost fn () 
//    requires 
//  tensor_fold_outer 

  admit (); 
}
