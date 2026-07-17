module Kuiper.Kernel.OnlineSoftmax

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Softmax

unfold
type softmax_notinplace_gpu_ty (et : Type0) {| floating et, real_like et, floating_real_like et |} =
  fn (nth : szp{nth <= max_threads})
     (#lenab : szp{lenab <= max_blocks * max_threads})
     (#l : layout1 lenab) {| ctlayout l |} (* TODO: let them have different layouts *)
     (a : array1 et l { is_global a })
     (b : array1 et l { is_global b })
     (#va : chest1 et lenab)
     (ra :  chest1 real lenab { va %~ ra })
     (#_: squash (chest_forallb not_nan va))
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    exists* (vb : chest1 et lenab). on gpu_loc (b |-> vb)
  ensures
    exists* (vb' : chest1 et lenab).
      on gpu_loc (b |-> vb') **
      pure (vb' %~ softmax_real ra)

(*
TODO

unfold
type softmax_notinplace_ty (et : Type0) {| floating et, real_like et, floating_real_like et  |} =
  fn (nth : szp{nth <= max_threads})
     (#lena : szp)
     (a : Vec.lvec et lena)
     (#va : erased (lseq et lena))
     (ra : erased (lseq real lena))
  preserves
    cpu **
    a |-> va
  requires
    pure (va %~ ra) **
    pure (lena <= max_blocks * max_threads)
    // ^ This could be removed
*)

inline_for_extraction noextract
val online_softmax_gpu (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  : softmax_notinplace_gpu_ty et

// TODO
// inline_for_extraction noextract
// val online_softmax (#et : Type0) {| floating et, real_like et, floating_real_like et |}
//   : softmax_notinplace_ty et
