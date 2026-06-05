module Kuiper.Kernel.Softmax

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Spec.Softmax
module Array1 = Kuiper.Array1

inline_for_extraction noextract
fn softmax_gpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp {nth <= max_threads})
  (#lena : szp)
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#va : erased (lseq et lena))
  (ra : erased (lseq real lena))
  preserves
    cpu
  requires
    on gpu_loc (a |-> va) **
    pure (va %~ ra) **
    pure (lena <= max_blocks * max_threads)
  ensures
    exists* (va' : lseq et lena).
      on gpu_loc (a |-> va') **
      pure (va' %~ softmax_real ra)
