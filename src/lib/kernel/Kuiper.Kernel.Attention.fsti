module Kuiper.Kernel.Attention

(* Specification of PyTorch's

     aten._scaled_dot_product_efficient_attention

   Inputs:
     query     : (N, H, L, E)        -- on GPU
     key       : (N, H, S, E)        -- on GPU
     value     : (N, H, S, Ev)       -- on GPU
     attn_bias : (N, H, L, S)        -- on GPU, additive bias
     scale     : et                  -- caller-provided scaling factor
                                       (PyTorch defaults to 1/sqrt(K))
     is_causal : bool                -- if true, mask above the main diagonal

   Outputs:
     out       : (N, H, L, Ev)       -- on GPU
     lse       : (N, H, L)           -- log-sum-exp, on GPU
                                       (PyTorch's `compute_log_sumexp` flag is
                                        skipped here; we always return it)

   Dropout (and the philox seed/offset outputs that accompany it) is
   intentionally omitted, as is the broadcasting behaviour of attn_bias
   (we require the full (N, H, L, S) shape). *)

#lang-pulse
open Kuiper
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor
open Kuiper.Index
open Kuiper.Bijection

open Kuiper.Spec.Attention

module EM4 = Kuiper.EMatrix4
module EM3 = Kuiper.EMatrix3
module CH = Kuiper.Chest
module SZ = Kuiper.SizeT

// TODO: add masking support (is_causal). Need extended reals; creating the triangular matrix mask 
// means having if-then-else in the real spec, which we currently don't have approximation rules for.
// Extended reals means we can say is_causal ==> bias += (triangle of -inf)

unfold
type scaled_dot_product_efficient_attention_ty
  (et : Type0) {| scalar et, real_like et |} = 
  fn 
    (n h : szp)
    (l s : szp)
    (e ev : szp)
    (#lQ: tlayout    (n @| h @| l @| e @| INil) { is_full lQ }) // needed for tlayout_bij for now.
    (#lK: tlayout    (n @| h @| s @| e @| INil) { is_full lK })
    (#lV: tlayout    (n @| h @| s @| ev @| INil) { is_full lV })
    (#lbias: tlayout (n @| h @| l @| s @| INil) { is_full lbias })
    {| ctlayout lQ, ctlayout lK, ctlayout lV, ctlayout lbias |}
    (gQ    : tensor et lQ    { is_global gQ    })
    (gK    : tensor et lK    { is_global gK    })
    (gV    : tensor et lV    { is_global gV    })
    (gbias : tensor et lbias { is_global gbias })
    (scale : et)
    (#eQ : erased    (EM4.t et n h l e))
    (#eK : erased    (EM4.t et n h s e))
    (#eV : erased    (EM4.t et n h s ev))
    (#ebias : erased (EM4.t et n h l s))
    (#rKT : erased   (EM4.t real n h e s))
    (#fQ #fK #fV #fbias : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gQ    |-> Frac fQ eQ) **
    on gpu_loc (gK    |-> Frac fK eK) **
    on gpu_loc (gV    |-> Frac fV eV) **
    on gpu_loc (gbias |-> Frac fbias ebias)
  requires
    pure (
      SZ.fits (l * ev) /\
      SZ.fits (n * h * l * e) /\
      SZ.fits (n * h * s * e)  /\
      SZ.fits (n * h * s * ev)  /\
      SZ.fits (n * h * l * ev)  /\ 
      SZ.fits (n * h * l * s)  /\
      SZ.fits (n * h * l) /\
      SZ.fits (h * l) /\
      (EM4.mkM (fun i j k l -> EM4.macc eK i j l k)) %~ rKT /\
      l * s <= max_blocks * max_threads /\
      l * ev <= max_blocks * max_threads /\
      n * h * l <= max_blocks /\
      n * h * l * s <= max_blocks * max_threads
    )
  returns
    // TODO: polymorphic out & LSE layout
    out : tensor et (l4_batched_row_major n h l ev) & 
          tensor et (l3_batched_row_major n h l)
  ensures
    (exists* (eO : EM4.t et n h l ev) (eLSE : EM3.t et n h l).
      on gpu_loc (fst out |-> eO) **
      on gpu_loc (snd out |-> eLSE) **
      pure (
        let out_spec, lse_spec = attention_real_batched
            (EM4.to_real_matrix eQ)
            rKT
            (EM4.to_real_matrix eV)
            (EM4.to_real_matrix ebias)
            (to_real scale) in
          eO %~ out_spec /\ eLSE %~ lse_spec)) **
    pure (is_global (fst out) /\ is_global (snd out)) 

inline_for_extraction noextract
val scaled_dot_product_efficient_attention
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
   : scaled_dot_product_efficient_attention_ty et
