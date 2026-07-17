module Kuiper.Kernel.OnlineSoftmaxDotprod

#lang-pulse
open Kuiper
open Kuiper.Tensor
module KS = Kuiper.Spec.Softmax
open Kuiper.DotProd

inline_for_extraction noextract
fn softmax_dotprod
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (len : szp{len <= max_blocks * max_threads})
  (#l : layout1 len) {| ctlayout l |}
  (a : array1 et l)
  (b : array1 et l)
  (r : gpu_ref et)
  (#va : chest1 et len)
  (#vb : chest1 et len)
  (ra : chest1 real len { va %~ ra })
  (rb : chest1 real len { vb %~ rb })
  (#fa #fb : perm)
  (#_: squash (chest_forallb not_nan va))
  (tid : szlt len)
  ()
  preserves
    gpu **
    a |-> Frac fa va **
    b |-> Frac fb vb
  returns
    res : et
  ensures
    pure (res %~ chest1_dotprod (KS.softmax_real ra) rb)
