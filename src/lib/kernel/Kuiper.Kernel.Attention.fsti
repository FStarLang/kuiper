module Kuiper.Kernel.Attention

(* Shape-only specification of PyTorch's

     aten._scaled_dot_product_efficient_attention

   Inputs:
     query     : (B, H, M, K)        -- on GPU
     key       : (B, H, N, K)        -- on GPU
     value     : (B, H, N, Kv)       -- on GPU
     attn_bias : (B, H, M, N)        -- on GPU, additive bias
     scale     : et                  -- caller-provided scaling factor
                                       (PyTorch defaults to 1/sqrt(K))
     is_causal : bool                -- if true, mask above the main diagonal

   Outputs:
     out       : (B, H, M, Kv)       -- on GPU
     lse       : (B, H, M)           -- log-sum-exp, on GPU
                                       (PyTorch's `compute_log_sumexp` flag is
                                        skipped here; we always return it)

   Dropout (and the philox seed/offset outputs that accompany it) is
   intentionally omitted, as is the broadcasting behaviour of attn_bias
   (we require the full (B, H, M, N) shape). Functional correctness is
   left for a future revision; this file pins down only the shapes/types. *)

#lang-pulse
open Kuiper
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor
open Kuiper.Index
open Kuiper.Real
module A4 = Kuiper.Array4
module A3 = Kuiper.Array3
module EM4 = Kuiper.EMatrix4
module EM3 = Kuiper.EMatrix3
open Kuiper.EMatrix
module SZ = Kuiper.SizeT

module MS = Kuiper.Spec.GEMM

// TODO: FIX SPEC ON NEW API
 
(*

// TODO: feels like we shouldn't need to import these kernel impl. modules
// to have the real-value specifications of these operators - should separate out into spec modules
open Kuiper.Kernel.BatchedGEMM
open Kuiper.Kernel.RowSoftmax

// TODO: add masking support (is_causal). Need extended reals; creating the triangular matrix mask 
// means having if-then-else in the real spec, which we currently don't have approximation rules for.
// Extended reals means we can say is_causal ==> bias += (triangle of -inf)

(* Pre-softmax attention scores. *)
let attn_scores
  (#m #n #k: pos)
  (q : ematrix real m k)
  (k_ : ematrix real k n)
  (bias : ematrix real m n)
  (scale : real)
  : ematrix real m n
  = mkM fun i j -> macc (MS.matmul q k_ ) i j *. scale +. macc bias i j

(* row-wise log sum exp of scores *)
let attn_lse
  (#m #n : pos)
  (scores : ematrix real m n)
  : lseq real m
  = Seq.init m (fun i -> 
      log (rsum (seq_map exp (ematrix_row scores i))))

(* Top-level real-valued spec: (output, log-sum-exp) given real inputs. *)
let attention_real
  (#m #n #k #kv : pos)
  (q : ematrix real m k)
  (k_ : ematrix real k n)
  (v : ematrix real n kv)
  (bias : ematrix real m n)
  (scale : real)
  : GTot (ematrix real m kv & lseq real m)
  = let scores = attn_scores q k_ bias scale in
    let probs  = row_softmax_real scores in
    let out    = MS.matmul probs v in
    let lse    = attn_lse scores in
    (out, lse)

unfold
type scaled_dot_product_efficient_attention_ty
  (et : Type0) {| scalar et, real_like et |}
  =
  fn (b h : szp)
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
       (EM4.mkM (fun i j k l -> EM4.macc sK i j l k)) %~ rKT /\
       m * n <= max_blocks * max_threads /\
       b * h * m <= max_blocks
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
     pure (A4.is_global (fst out) /\ A3.is_global (snd out))

inline_for_extraction noextract
val scaled_dot_product_efficient_attention
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
   : scaled_dot_product_efficient_attention_ty et


*)