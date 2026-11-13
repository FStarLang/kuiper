module KPlay.Proof.Base.ConvolutionLoop

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
module Approx = Kuiper.Approximates.Base
module Sum = Kuiper.Sum
module P = KPlay.Proof.Shared.Projection
module S = KPlay.Proof.Shared.Attention
module R = KPlay.Proof.Shared.Recurrent
module Native = KPlay.Proof.Base.Convolution
module Transfer = KPlay.Proof.Base.Transfer
module Algebra = KPlay.Proof.Base.Algebra

inline_for_extraction noextract
let after (before:Native.cells) (hidden:Transfer.scratch)
  (slot:nat{slot < 18}) (count:nat{count <= 6144}) : GTot Native.cells =
  Seq.init_ghost 442368 (fun i ->
    if slot * 24576 <= i && i < slot * 24576 + count * 4 then
      let j = i - slot * 24576 in
      if j % 4 < 3 then Seq.index before (i + 1) else Seq.index hidden (2048 + j / 4)
    else Seq.index before i)

inline_for_extraction noextract
let zero (before:Native.cells) (hidden:Transfer.scratch) (slot:nat{slot < 18})
  : Lemma (after before hidden slot 0 == before) =
  assert (Seq.equal (after before hidden slot 0) before);
  Seq.lemma_eq_elim (after before hidden slot 0) before

inline_for_extraction noextract
let hidden_frame (before current:Transfer.scratch) (count:nat{count <= 6144}) : prop =
  forall (i:P.index 16000).
    ~(2048 <= i /\ i < 2048 + count) ==> Seq.index current i == Seq.index before i

inline_for_extraction noextract
let step (before:Native.cells) (initial current:Transfer.scratch)
  (slot:nat{slot < 18}) (channel:P.index 6144)
  : Lemma
      (requires hidden_frame initial current channel)
      (ensures Native.shifted (after before initial slot channel) current slot channel 4 ==
        after before initial slot (channel + 1)) =
  assert (Seq.equal (Native.shifted (after before initial slot channel) current slot channel 4)
    (after before initial slot (channel + 1)));
  Seq.lemma_eq_elim (Native.shifted (after before initial slot channel) current slot channel 4)
    (after before initial slot (channel + 1))

inline_for_extraction noextract
let input_rep (conv:Native.cells) (hidden:Transfer.scratch) (weights:Transfer.packed)
  (slot:nat{slot < 18}) (base:nat{base + 19458048 <= 752393024})
  (old wr:R.matrix 6144 4) (raw:R.vector 6144) : prop =
  (forall (channel:P.index 6144) (tap:P.index 4).
    Approx.v_approximates (Seq.index conv (Native.address slot channel tap)) (old channel tap) /\
    Approx.v_approximates (Seq.index weights (base + 19433472 + 4 * channel + tap)) (wr channel tap)) /\
  (forall (channel:P.index 6144).
    Approx.v_approximates (Seq.index hidden (2048 + channel)) (raw channel))

inline_for_extraction noextract
let shifted_taps (old:R.matrix 6144 4) (raw:R.vector 6144) (channel:P.index 6144) : R.vector 4 =
  fun tap -> if tap < 3 then old channel (tap + 1) else raw channel

inline_for_extraction noextract
let row_at (values:erased (R.matrix 6144 4)) (channel:P.index 6144) : GTot (R.vector 4) =
  (reveal #(R.matrix 6144 4) values) channel

inline_for_extraction noextract
let channel_input (scope:prop) (conv:Native.cells) (initial current:Transfer.scratch) (weights:Transfer.packed)
  (slot:nat{slot < 18}) (base:nat{base + 19458048 <= 752393024})
  (old wr:R.matrix 6144 4) (raw:R.vector 6144) (channel:P.index 6144)
  : Lemma
      (requires (scope ==> input_rep conv initial weights slot base old wr raw) /\ hidden_frame initial current channel)
      (ensures scope ==> Native.tap_rep (after conv initial slot channel) current slot channel (shifted_taps old raw channel) /\
        (forall (j:P.index 4).
          Approx.v_approximates (Seq.index weights (base + 19433472 + 4 * channel + j)) (wr channel j))) = ()

inline_for_extraction noextract
let convolution_value (old wr:R.matrix 6144 4) (raw:R.vector 6144) (channel:P.index 6144) : GTot real =
  R.convolve wr (R.shift_convolution old raw) channel

inline_for_extraction noextract
let convolution_sum (old wr:R.matrix 6144 4) (raw:R.vector 6144) (channel:P.index 6144)
  : Lemma (
      Sum.sum #real #Sum.real_add_semigroup 0 4
        (Native.terms (wr channel) (shifted_taps old raw channel)) ==
      Sum.sum #real #Sum.real_add_semigroup 0 4
        (fun tap -> wr channel tap *. R.shift_convolution old raw channel tap)) =
  assert (
    Sum.sum #real #Sum.real_add_semigroup 0 4
      (Native.terms (wr channel) (shifted_taps old raw channel)) ==
    Sum.sum #real #Sum.real_add_semigroup 0 4
      (fun tap -> wr channel tap *. R.shift_convolution old raw channel tap)) by (
    FStar.Tactics.V2.apply_lemma (`(Algebra.sum_ext #4));
    FStar.Tactics.V2.norm [zeta; iota; delta_only [
      "KPlay.Proof.Base.Convolution.terms";
      "KPlay.Proof.Base.ConvolutionLoop.shifted_taps";
      "KPlay.Proof.Shared.Recurrent.shift_convolution"]];
    FStar.Tactics.V2.smt ())

#push-options "--fuel 0 --ifuel 0"

inline_for_extraction noextract
let convolution_equation (old wr:R.matrix 6144 4) (raw:R.vector 6144) (channel:P.index 6144)
  : Lemma (
      R.silu (Sum.sum #real #Sum.real_add_semigroup 0 4
        (Native.terms (wr channel) (shifted_taps old raw channel))) ==
      convolution_value old wr raw channel) =
  convolution_sum old wr raw channel;
  norm_spec [delta_once [
    "KPlay.Proof.Base.ConvolutionLoop.convolution_value";
    "KPlay.Proof.Shared.Recurrent.convolve"]]
    (convolution_value old wr raw channel)

#pop-options

inline_for_extraction noextract
let output_rep (hidden:Transfer.scratch) (old wr:R.matrix 6144 4) (raw:R.vector 6144)
  (count:nat{count <= 6144}) : prop =
  forall (channel:nat{channel < count}).
    Approx.v_approximates (Seq.index hidden (2048 + channel)) (convolution_value old wr raw channel)

inline_for_extraction noextract
let store_step (scope:prop) (initial current:Transfer.scratch) (old wr:R.matrix 6144 4) (raw:R.vector 6144)
  (channel:P.index 6144) (value:BF.t)
  : Lemma
      (requires hidden_frame initial current channel /\ (scope ==> output_rep current old wr raw channel /\
        Approx.v_approximates value (convolution_value old wr raw channel)))
      (ensures hidden_frame initial (Seq.upd current (2048 + channel) value) (channel + 1) /\
        (scope ==> output_rep (Seq.upd current (2048 + channel) value) old wr raw (channel + 1))) = ()

inline_for_extraction noextract
fn apply
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000) (conv:larray BF.t 442368)
  (base:sz{SZ.v base + 19458048 <= 752393024}) (slot:sz{SZ.v slot < 18})
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch) (#before:erased Native.cells)
  (#fraction:perm) (#old #wr:erased (R.matrix 6144 4)) (#raw:erased (R.vector 6144)) (#scope:prop)
  preserves gpu ** A.pts_to weights #fraction vw
  requires A.pts_to hidden vh ** A.pts_to conv before
  requires pure (scope ==> input_rep before vh vw (SZ.v slot) (SZ.v base) old wr raw)
  ensures exists* (result:Transfer.scratch).
    A.pts_to hidden result ** A.pts_to conv (after before vh (SZ.v slot) 6144) **
    pure (hidden_frame vh result 6144 /\ (scope ==> output_rep result old wr raw 6144))
{
  zero before vh (SZ.v slot);
  let mut i = 0sz;
  while (!i <^ 6144sz)
    invariant exists* (channel:sz{SZ.v channel <= 6144}) (current:Transfer.scratch).
      i |-> channel ** A.pts_to hidden current ** A.pts_to conv (after before vh (SZ.v slot) (SZ.v channel)) **
      pure (hidden_frame vh current (SZ.v channel) /\ (scope ==> output_rep current old wr raw (SZ.v channel)))
    invariant gpu ** A.pts_to weights #fraction vw
    decreases (6144 - SZ.v !i)
  {
    let channel = !i;
    with current. assert A.pts_to hidden current;
    channel_input scope before vh current vw (SZ.v slot) (SZ.v base) old wr raw (SZ.v channel);
    let value = Native.compute_channel weights hidden conv base slot channel
      #vw #current #(after before vh (SZ.v slot) (SZ.v channel))
      #fraction #1.0R #(row_at wr (SZ.v channel)) #(shifted_taps old raw (SZ.v channel)) #scope;
    step before vh current (SZ.v slot) (SZ.v channel);
    convolution_equation old wr raw (SZ.v channel);
    store_step scope vh current old wr raw (SZ.v channel) value;
    hidden.(SZ.add 2048sz channel) <- value;
    i := SZ.add channel 1sz;
  }
}
