module Kuiper.Kernel.OnlineSoftmaxDotprod

#lang-pulse
open Kuiper
open Kuiper.Seq.Common
open Kuiper.Array1
open Kuiper.Tensor.Layout { ctlayout }
module KS = Kuiper.Spec.Softmax
open Kuiper.DotProd

inline_for_extraction noextract
fn softmax_dotprod
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (len : szp{len <= max_blocks * max_threads})
  (#l : Kuiper.Array1.layout len) {| ctlayout l |}
  (a : array1 et l)
  (b : array1 et l)
  (r : gpu_ref et)
  (#va : erased (lseq et len))
  (#vb : erased (lseq et len))
  (ra : erased (lseq real len) { va %~ ra })
  (rb : erased (lseq real len) { vb %~ rb })
  (#fa #fb : perm)
  (#_: squash (seq_forallb not_nan va))
  (tid : szlt len)
  ()
  preserves
    gpu **
    a |-> Frac fa va **
    b |-> Frac fb vb
  returns
    res : et
  ensures
    pure (res %~ seq_dotprod (KS.softmax_real ra) rb)
