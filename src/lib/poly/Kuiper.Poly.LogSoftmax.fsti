module Kuiper.Poly.LogSoftmax

#lang-pulse
open Kuiper
open Kuiper.Real { rlog }
open Kuiper.Seq.Common
module Vec = Pulse.Lib.Vec
module SM = Kuiper.Poly.Softmax

// Unfortunate to have to define and use this
let seq_refine #a (p : a -> prop)
  (s : seq a { forall i. p (s @! i) })
  : GTot (lseq (x:a{p x}) (Seq.length s))
  = Seq.init_ghost #(x:a{p x}) (Seq.length s) (fun i -> s @! i)

let seq_refine_len #a (p : a -> prop)
  (s : seq a { forall i. p (s @! i) })
  : Lemma (len (seq_refine p s) == len s)
          [SMTPat (Seq.length (seq_refine p s))]
  = ()

let seq_refine_at #a (p : a -> prop)
  (s : seq a { forall i. p (s @! i) })
  (i : natlt (Seq.length s))
  : Lemma ((seq_refine p s) @! i == s @! i)
          [SMTPat ((seq_refine p s) @! i)]
  = ()

// Log of softmax.
let log_softmax_real (s:Seq.seq real { Seq.length s > 0 }) =
  lseq_map rlog (seq_refine (fun x -> x >. 0.0R) (SM.softmax_real s))

let log_softmax_real' (s:Seq.seq real { Seq.length s > 0 }) =
  let exps = seq_map rexp s in
  let summ : real = SM.sum exps in
  lseq_map #_ #_ #(Seq.length s) FStar.Real.(fun x -> x -. rlog summ) s

unfold
type log_softmax_ty (et : Type0) {| floating et, real_like et |} =
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
      pure (va' %~ log_softmax_real ra)

inline_for_extraction noextract
val log_softmax (#et : Type0) {| floating et, real_like et, floating_real_like et |}
: log_softmax_ty et
