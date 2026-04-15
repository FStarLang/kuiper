module Kuiper.Poly.Softmax

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Array1
open Kuiper.Real { rexp }
module Array1 = Kuiper.Array1
module KS = Kuiper.Seq.Common
module Vec = Pulse.Lib.Vec
open Kuiper.Tensor.Layout.Alg { l1_forward }

let sum (#et:Type0) {| scalar et |} (s:seq et) =
  KS.seq_fold_left add zero s

val sum_non_zero
    (s:seq real { forall (i:natlt (Seq.length s)). Seq.index s i >. 0.0R })
    (acc:real)
  : Lemma (requires Seq.length s > 0)
          (ensures KS.seq_fold_left add acc s >. acc)

let softmax_real (s:Seq.seq real { Seq.length s > 0 }) =
  let open KS in
  let exps = seq_map rexp s in
  let avg : real = sum exps in
  sum_non_zero exps zero;
  seq_map FStar.Real.(fun x -> x /. avg) exps

unfold
type softmax_gpu_ty (et : Type0) {| floating et, real_like et |} =
  fn (#lena : szp)
     (a : array1 et (l1_forward lena) { is_global a })
     (#va : erased (lseq et lena))
     (ra : erased (lseq real lena))
  preserves
    cpu
  requires
    on gpu_loc (a |-> va) **
    pure (lena <= max_threads) **
    pure (va %~ ra)
  ensures
    exists* (va' : lseq et lena).
      on gpu_loc (a |-> va') **
      pure (va' %~ softmax_real ra)

inline_for_extraction noextract
val softmax_gpu (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  : softmax_gpu_ty et

unfold
type softmax_ty (et : Type0) {| floating et, real_like et |} =
  fn (#lena : szp)
     (a : Vec.lvec et lena)
     (#va : erased (lseq et lena))
     (ra : erased (lseq real lena))
  preserves
    cpu
  requires
    a |-> va **
    pure (lena <= max_threads) **
    pure (va %~ ra)
  ensures
    exists* (va' : lseq et lena).
      a |-> va' **
      pure (va' %~ softmax_real ra)

inline_for_extraction noextract
val softmax (#et : Type0) {| floating et, real_like et, floating_real_like et |}
: softmax_ty et
