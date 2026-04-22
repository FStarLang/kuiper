module Kuiper.Poly.Softmax

#lang-pulse
open Kuiper
open Kuiper.Real { rexp }
open Kuiper.Seq.Common
module Vec = Pulse.Lib.Vec

// FIXME: If this is only used on reals it should be specialized.
let sum (#et:Type0) {| scalar et |} (s:seq et) =
  seq_fold_left add zero s

val sum_non_zero
    (s:seq real { forall (i:natlt (Seq.length s)). Seq.index s i >. 0.0R })
    (acc:real)
  : Lemma (requires Seq.length s > 0)
          (ensures seq_fold_left add acc s >. acc)
          [SMTPat (seq_fold_left add acc s)]

let softmax_real (s:Seq.seq real { Seq.length s > 0 }) =
  let exps = seq_map rexp s in
  let summ : real = sum exps in
  seq_map FStar.Real.(fun x -> rexp x /. summ) s

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
