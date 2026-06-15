module Kuiper.Kernel.Attention

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


fn scaled_dot_product_efficient_attention
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (n h : szp)
  (l s : szp)
  (e ev : szp)
  (#lQ: tlayout    (n @| h @| l @| e @| INil))
  (#lK: tlayout    (n @| h @| s @| e @| INil))
  (#lV: tlayout    (n @| h @| s @| ev @| INil))
  (#lbias: tlayout (n @| h @| l @| s @| INil))
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
      SZ.fits (n * h * l * e) /\
      SZ.fits (n * h * s * e)  /\
      SZ.fits (n * h * s * ev)  /\
      SZ.fits (n * h * l * s)  /\
      SZ.fits (n * h * l) /\
      (EM4.mkM (fun i j k l -> EM4.macc eK i j l k)) %~ rKT /\
      l * s <= max_blocks * max_threads /\
      n * h * l <= max_blocks
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
    pure (is_global (fst out) /\ is_global (snd out)) {

  admit ();
}