module KPlay.Proof.Base.Convolution

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
module Fast = Kuiper.Float32.FastMath
module Cast = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base
module Sum = Kuiper.Sum
module P = KPlay.Proof.Shared.Projection
module S = KPlay.Proof.Shared.Attention
module R = KPlay.Proof.Shared.Recurrent
module Transfer = KPlay.Proof.Base.Transfer
module SM = KPlay.Proof.Base.SIMTMath
module L2 = KPlay.Proof.Base.L2
module Scalar = KPlay.Proof.Base.Scalar

inline_for_extraction noextract
let cells = lseq BF.t 442368

inline_for_extraction noextract
let address (slot:nat{slot < 18}) (channel:P.index 6144) (tap:P.index 4) : P.index 442368 =
  slot * 24576 + channel * 4 + tap

inline_for_extraction noextract
let shifted (before:cells) (hidden:Transfer.scratch)
  (slot:nat{slot < 18}) (channel:P.index 6144) (count:nat{count <= 4}) : GTot cells =
  let start = address slot channel 0 in
  Seq.init_ghost 442368 (fun i ->
    if start <= i && i < start + count then
      if i - start < 3 then Seq.index before (i + 1) else Seq.index hidden (2048 + channel)
    else Seq.index before i)

inline_for_extraction noextract
let shifted_zero (before:cells) (hidden:Transfer.scratch) (slot:nat{slot < 18}) (channel:P.index 6144)
  : Lemma (shifted before hidden slot channel 0 == before) =
  assert (Seq.equal (shifted before hidden slot channel 0) before);
  Seq.lemma_eq_elim (shifted before hidden slot channel 0) before

inline_for_extraction noextract
let shifted_step (before:cells) (hidden:Transfer.scratch)
  (slot:nat{slot < 18}) (channel:P.index 6144) (j:nat{j < 3})
  : Lemma (
      Seq.index (shifted before hidden slot channel j) (address slot channel (j + 1)) ==
        Seq.index before (address slot channel (j + 1)) /\
      Seq.upd (shifted before hidden slot channel j) (address slot channel j)
        (Seq.index before (address slot channel (j + 1))) ==
      shifted before hidden slot channel (j + 1)) =
  assert (Seq.equal
    (Seq.upd (shifted before hidden slot channel j) (address slot channel j)
      (Seq.index before (address slot channel (j + 1))))
    (shifted before hidden slot channel (j + 1)));
  Seq.lemma_eq_elim
    (Seq.upd (shifted before hidden slot channel j) (address slot channel j)
      (Seq.index before (address slot channel (j + 1))))
    (shifted before hidden slot channel (j + 1))

inline_for_extraction noextract
let shifted_last (before:cells) (hidden:Transfer.scratch)
  (slot:nat{slot < 18}) (channel:P.index 6144)
  : Lemma (
      Seq.upd (shifted before hidden slot channel 3) (address slot channel 3) (Seq.index hidden (2048 + channel)) ==
      shifted before hidden slot channel 4) =
  assert (Seq.equal
    (Seq.upd (shifted before hidden slot channel 3) (address slot channel 3) (Seq.index hidden (2048 + channel)))
    (shifted before hidden slot channel 4));
  Seq.lemma_eq_elim
    (Seq.upd (shifted before hidden slot channel 3) (address slot channel 3) (Seq.index hidden (2048 + channel)))
    (shifted before hidden slot channel 4)

inline_for_extraction noextract
fn shift_channel
  (conv:larray BF.t 442368) (hidden:larray BF.t 16000)
  (slot:sz{SZ.v slot < 18}) (channel:sz{SZ.v channel < 6144})
  (#before:erased cells) (#vh:erased Transfer.scratch) (#hidden_fraction:perm)
  preserves gpu ** A.pts_to hidden #hidden_fraction vh
  requires A.pts_to conv before
  ensures A.pts_to conv (shifted before vh (SZ.v slot) (SZ.v channel) 4)
{
  shifted_zero before vh (SZ.v slot) (SZ.v channel);
  let start = SZ.add (SZ.mul slot 24576sz) (SZ.mul channel 4sz);
  let mut j = 0sz;
  while (!j <^ 3sz)
    invariant exists* (i:sz{SZ.v i <= 3}).
      j |-> i ** A.pts_to conv (shifted before vh (SZ.v slot) (SZ.v channel) (SZ.v i))
    invariant gpu
    decreases (3 - SZ.v !j)
  {
    let i = !j;
    shifted_step before vh (SZ.v slot) (SZ.v channel) (SZ.v i);
    let previous = conv.(SZ.add start (SZ.add i 1sz));
    conv.(SZ.add start i) <- previous;
    j := SZ.add i 1sz;
  };
  shifted_last before vh (SZ.v slot) (SZ.v channel);
  let raw = hidden.(SZ.add 2048sz channel);
  conv.(SZ.add start 3sz) <- raw;
}

inline_for_extraction noextract
let real_at (values:erased (R.vector 4)) (i:P.index 4) : GTot real =
  (reveal #(R.vector 4) values) i

inline_for_extraction noextract
let terms (weights taps:R.vector 4) : R.vector 4 =
  fun i -> taps i *. weights i

inline_for_extraction noextract
let sum_step (weights taps:R.vector 4) (count:nat{count < 4})
  : Lemma (
      Sum.sum #real #Sum.real_add_semigroup 0 (count + 1) (terms weights taps) ==
      Sum.sum #real #Sum.real_add_semigroup 0 count (terms weights taps) +. taps count *. weights count) =
  Sum.sum_pop_right #real #Sum.real_add_semigroup 0 (count + 1) (terms weights taps)

inline_for_extraction noextract
let tap_rep (before:cells) (hidden:Transfer.scratch)
  (slot:nat{slot < 18}) (channel:P.index 6144) (taps:R.vector 4) : prop =
  (forall (j:nat{j < 3}). Approx.v_approximates (Seq.index before (address slot channel (j + 1))) (taps j)) /\
  Approx.v_approximates (Seq.index hidden (2048 + channel)) (taps 3)

inline_for_extraction noextract
let shifted_rep (scope:prop) (before:cells) (hidden:Transfer.scratch)
  (slot:nat{slot < 18}) (channel:P.index 6144) (taps:R.vector 4)
  : Lemma
      (requires scope ==> tap_rep before hidden slot channel taps)
      (ensures scope ==> forall (j:P.index 4).
        Approx.v_approximates (Seq.index (shifted before hidden slot channel 4) (address slot channel j)) (taps j)) = ()

inline_for_extraction noextract
let activation_approx (scope:prop) (sum:FP.t) (logical:real)
  : Lemma
      (requires scope ==> Approx.v_approximates sum logical)
      (ensures scope ==> Approx.v_approximates
        (Cast.cast_f32_to_bf16 (Scalar.silu (L2.rounded sum))) (R.silu logical)) =
  let under (_:scope)
    : Lemma (Approx.v_approximates
        (Cast.cast_f32_to_bf16 (Scalar.silu (L2.rounded sum))) (R.silu logical)) =
    L2.rounded_approx sum logical;
    Scalar.silu_approx (L2.rounded sum) logical;
    Cast.cast_f32_to_bf16_ok (Scalar.silu (L2.rounded sum)) (R.silu logical)
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
fn compute_channel
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000) (conv:larray BF.t 442368)
  (base:sz{SZ.v base + 19458048 <= 752393024})
  (slot:sz{SZ.v slot < 18}) (channel:sz{SZ.v channel < 6144})
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch) (#before:erased cells)
  (#weight_fraction #hidden_fraction:perm)
  (#wr #taps:erased (R.vector 4)) (#scope:prop)
  preserves gpu ** A.pts_to weights #weight_fraction vw ** A.pts_to hidden #hidden_fraction vh
  requires A.pts_to conv before
  requires pure (scope ==>
    tap_rep before vh (SZ.v slot) (SZ.v channel) taps /\
    (forall (j:P.index 4).
      Approx.v_approximates (Seq.index vw (SZ.v base + 19433472 + 4 * SZ.v channel + j)) (real_at wr j)))
  returns result:BF.t
  ensures A.pts_to conv (shifted before vh (SZ.v slot) (SZ.v channel) 4)
  ensures pure (scope ==> Approx.v_approximates result
    (R.silu (Sum.sum #real #Sum.real_add_semigroup 0 4 (terms wr taps))))
{
  shift_channel conv hidden slot channel;
  shifted_rep scope before vh (SZ.v slot) (SZ.v channel) taps;
  let start = SZ.add (SZ.mul slot 24576sz) (SZ.mul channel 4sz);
  let weight_start = SZ.add (SZ.add base (SZ.uint_to_t 19433472)) (SZ.mul channel 4sz);
  let mut j = 0sz;
  let mut sum:FP.t = F.zero #FP.t;
  while (!j <^ 4sz)
    invariant exists* (i:sz{SZ.v i <= 4}) (current:FP.t).
      j |-> i ** sum |-> current **
      pure (scope ==> Approx.v_approximates current (Sum.sum #real #Sum.real_add_semigroup 0 (SZ.v i) (terms wr taps)))
    invariant gpu ** A.pts_to weights #weight_fraction vw **
      A.pts_to conv (shifted before vh (SZ.v slot) (SZ.v channel) 4)
    decreases (4 - SZ.v !j)
  {
    let i = !j;
    let left = conv.(SZ.add start i);
    let right = weights.(SZ.add weight_start i);
    let old = !sum;
    let next = Fast.fma_rn (Cast.cast_bf16_to_f32 left) (Cast.cast_bf16_to_f32 right) old;
    SM.conditional_fma scope left right old next (real_at taps (SZ.v i)) (real_at wr (SZ.v i))
      (Sum.sum #real #Sum.real_add_semigroup 0 (SZ.v i) (terms wr taps));
    sum_step wr taps (SZ.v i);
    sum := next;
    j := SZ.add i 1sz;
  };
  let completed = !j;
  assert pure (SZ.v completed == 4);
  let sum = !sum;
  activation_approx scope sum (Sum.sum #real #Sum.real_add_semigroup 0 4 (terms wr taps));
  let x = Cast.cast_bf16_to_f32 (Cast.cast_f32_to_bf16 sum);
  let activated = Scalar.silu x;
  Cast.cast_f32_to_bf16 activated
}
