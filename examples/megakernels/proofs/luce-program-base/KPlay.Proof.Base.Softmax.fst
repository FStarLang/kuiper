module KPlay.Proof.Base.Softmax

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Array
open FStar.Ghost
open FStar.Real

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module Cast = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base
module Exp = FStar.Math.Exp
module S = KPlay.Proof.Shared.Attention
module Math = KPlay.Proof.Base.AttentionAlgebra
module Scale = KPlay.Proof.Base.SoftmaxScale
module Stripes = KPlay.Proof.Base.SoftmaxStripes
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
let frame (before after:Transfer.scratch) (offset count:nat{offset + count <= 16000}) : prop =
  forall (i:nat{i < 16000}).
    ~(offset <= i /\ i < offset + count) ==> Seq.index after i == Seq.index before i

inline_for_extraction noextract
let produced (n:pos{n <= 647}) (offset:nat{offset + n <= 16000})
  (after:Transfer.scratch) (scores:S.vector n) (count:nat{count <= n}) : prop =
  forall (i:nat{i < count}).
    Approx.v_approximates (Seq.index after (offset + i)) (S.softmax n scores i)

inline_for_extraction noextract
let store_step (scope:prop) (n:pos{n <= 647}) (offset:nat{offset + n <= 16000})
  (before current:Transfer.scratch) (scores:S.vector n) (count:nat{count < n}) (value:BF.t)
  : Lemma
      (requires frame before current offset count /\ (scope ==> produced n offset current scores count /\
        Approx.v_approximates value (S.softmax n scores count)))
      (ensures frame before (Seq.upd current (offset + count) value) offset (count + 1) /\
        (scope ==> produced n offset (Seq.upd current (offset + count) value) scores (count + 1))) = ()

inline_for_extraction noextract
let quotient_approx (scope:prop) (n:pos{n <= 647}) (scores:S.vector n) (shift:real)
  (i:S.index n) (value denominator:FP.t)
  : Lemma
      (requires scope ==>
        Approx.v_approximates value (Exp.exp (scores i -. shift)) /\
        Approx.v_approximates denominator (Math.exponential_sum n scores shift))
      (ensures scope ==> Approx.v_approximates (Cast.cast_f32_to_bf16 (F.div value denominator))
        (S.softmax n scores i)) =
  let under (_:scope)
    : Lemma (Approx.v_approximates (Cast.cast_f32_to_bf16 (F.div value denominator))
        (S.softmax n scores i)) =
    Approx.div_approx value denominator (Exp.exp (scores i -. shift)) (Math.exponential_sum n scores shift);
    Math.softmax_shift_value n scores shift i;
    Cast.cast_f32_to_bf16_ok (F.div value denominator) (S.softmax n scores i)
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
fn finish
  (n:sz{0 < SZ.v n /\ SZ.v n <= 647}) (offset:sz{SZ.v offset + SZ.v n <= 16000})
  (hidden:larray BF.t 16000) (values:larray FP.t 2048) (partial:larray FP.t 32)
  (#sx:erased Transfer.scratch) (#sv:erased (lseq FP.t 2048)) (#sp:erased (lseq FP.t 32))
  (#scores:erased (S.vector (SZ.v n))) (#shift:erased real) (#scope:prop)
  preserves gpu ** values |-> sv ** partial |-> sp
  requires A.pts_to hidden sx
  requires pure (scope ==>
    Scale.values_rep (SZ.v n) sv (Stripes.exponential_values (SZ.v n) scores (reveal #real shift)) /\
    Approx.v_approximates (Seq.index sp 0) (Math.exponential_sum (SZ.v n) scores (reveal #real shift)))
  ensures exists* (result:Transfer.scratch).
    A.pts_to hidden result **
    pure (frame sx result (SZ.v offset) (SZ.v n) /\ (scope ==> produced (SZ.v n) (SZ.v offset) result scores (SZ.v n)))
{
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (j:sz{SZ.v j <= SZ.v n}) (current:Transfer.scratch).
      i |-> j ** A.pts_to hidden current **
      pure (frame sx current (SZ.v offset) (SZ.v j) /\
        (scope ==> produced (SZ.v n) (SZ.v offset) current scores (SZ.v j)))
    invariant gpu ** values |-> sv ** partial |-> sp
    decreases (SZ.v n - SZ.v !i)
  {
    let j = !i;
    with current. assert A.pts_to hidden current;
    let value = values.(j);
    let denominator = partial.(0sz);
    let quotient = F.div value denominator;
    quotient_approx scope (SZ.v n) scores (reveal #real shift) (SZ.v j) value denominator;
    let rounded = Cast.cast_f32_to_bf16 quotient;
    store_step scope (SZ.v n) (SZ.v offset) sx current scores (SZ.v j) rounded;
    hidden.(SZ.add offset j) <- rounded;
    i := SZ.add j 1sz;
  }
}

inline_for_extraction noextract
fn apply
  (n:sz{0 < SZ.v n /\ SZ.v n <= 647}) (offset:sz{SZ.v offset + SZ.v n <= 16000})
  (hidden:larray BF.t 16000) (values:larray FP.t 2048) (partial:larray FP.t 32)
  (#sx:erased Transfer.scratch) (#sv:erased (lseq FP.t 2048)) (#sp:erased (lseq FP.t 32))
  (#input:erased (S.vector (SZ.v n))) (#scope:prop)
  preserves gpu
  requires A.pts_to hidden sx ** values |-> sv ** partial |-> sp
  requires pure (scope ==> Scale.hidden_rep (SZ.v n) sx (SZ.v offset) input)
  ensures exists* (result:Transfer.scratch) (vnext:lseq FP.t 2048) (pnext:lseq FP.t 32).
    A.pts_to hidden result ** values |-> vnext ** partial |-> pnext **
    pure (frame sx result (SZ.v offset) (SZ.v n) /\
      (scope ==> produced (SZ.v n) (SZ.v offset) result (Math.scaled_scores (SZ.v n) input) (SZ.v n)))
{
  let width = Scale.source_width n;
  let maximum = Scale.scale_and_maximum hidden values n offset;
  Scale.values_approx scope (SZ.v n) sv sx (SZ.v offset) input;
  // Only the free softmax shift is interpreted here, never a propagated hidden vector.
  Approx.to_real_ok maximum;
  let shift : erased real = hide #real (Approx.to_real maximum);
  Stripes.exponentiate_and_reduce n width values partial maximum #_ #sp
    #(Math.scaled_scores (SZ.v n) input) #shift #scope;
  finish n offset hidden values partial #sx #_ #_
    #(Math.scaled_scores (SZ.v n) input) #shift #scope;
}
