module KPlay.Proof.Base.RecurrentGating

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
module Root = FStar.Math.Sqrt
module P = KPlay.Proof.Shared.Projection
module O = KPlay.Proof.Shared.Observations
module R = KPlay.Proof.Shared.Recurrent
module H = KPlay.Proof.Base.RecurrentHeadModel
module State = KPlay.Proof.Base.RecurrentStateMath
module RMS = KPlay.Proof.Base.RMSNative
module Norm = KPlay.Proof.Base.Norm
module L2 = KPlay.Proof.Base.L2
module Scalar = KPlay.Proof.Base.Scalar
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
let rms_bridge (values:R.vector 128)
  : Lemma (R.rms_scale #128 values ==
      Root.rsqrt (P.rms_argument #128 P.reference_epsilon values)) =
  assert (R.rms_scale #128 values ==
    Root.rsqrt (P.rms_argument #128 P.reference_epsilon values)) by (
    FStar.Tactics.V2.norm [zeta; iota; delta_only [
      "KPlay.Proof.Shared.Recurrent.rms_scale"; "KPlay.Proof.Shared.Recurrent.sum_squares";
      "KPlay.Proof.Shared.Recurrent.epsilon"; "KPlay.Proof.Shared.Projection.rms_argument";
      "KPlay.Proof.Shared.Projection.real_sum"; "KPlay.Proof.Shared.Projection.reference_epsilon"]];
    FStar.Tactics.V2.trefl ())

inline_for_extraction noextract
let value (before:Transfer.scratch) (weights:Transfer.packed)
  (base:nat{base + 19458208 <= 752393024}) (head:P.index 16) (i:P.index 128) : GTot BF.t =
  let scale = Norm.source_scale 128 (RMS.input 128 before (10400 + head * 128)) in
  let normalized = L2.rounded (F.mul (Cast.cast_bf16_to_f32 (Seq.index before (10400 + head * 128 + i))) scale) in
  let gained = L2.rounded (F.mul normalized (Cast.cast_bf16_to_f32 (Seq.index weights (base + 19458080 + i)))) in
  let gate = Scalar.silu (Cast.cast_bf16_to_f32 (Seq.index before (8192 + head * 128 + i))) in
  Cast.cast_f32_to_bf16 (F.mul gained gate)

inline_for_extraction noextract
let inputs (before:Transfer.scratch) (weights:Transfer.packed)
  (base:nat{base + 19458208 <= 752393024}) (head:P.index 16) (x:H.inputs) : prop =
  State.core_prefix before head x 128 /\
  (forall (i:P.index 128).
    Approx.v_approximates (Seq.index before (8192 + head * 128 + i)) (x.H.gate i) /\
    Approx.v_approximates (Seq.index weights (base + 19458080 + i)) (x.H.norm_weight i))

inline_for_extraction noextract
let value_approx (before:Transfer.scratch) (weights:Transfer.packed)
  (base:nat{base + 19458208 <= 752393024}) (head:P.index 16) (x:H.inputs) (i:P.index 128)
  : Lemma
      (requires inputs before weights base head x)
      (ensures Approx.v_approximates (value before weights base head i) (H.gated x i)) =
  Norm.source_scale_approx 128 (RMS.input 128 before (10400 + head * 128)) (H.core x);
  rms_bridge (H.core x);
  let scale = Norm.source_scale 128 (RMS.input 128 before (10400 + head * 128)) in
  let real_scale = R.rms_scale #128 (H.core x) in
  let raw = Seq.index before (10400 + head * 128 + i) in
  let weight = Seq.index weights (base + 19458080 + i) in
  let gate = Seq.index before (8192 + head * 128 + i) in
  Cast.cast_bf16_to_f32_ok raw (H.core x i);
  Approx.a_mul (Cast.cast_bf16_to_f32 raw) scale (H.core x i) real_scale;
  L2.rounded_approx (F.mul (Cast.cast_bf16_to_f32 raw) scale) (H.core x i *. real_scale);
  let normalized = L2.rounded (F.mul (Cast.cast_bf16_to_f32 raw) scale) in
  Cast.cast_bf16_to_f32_ok weight (x.H.norm_weight i);
  Approx.a_mul normalized (Cast.cast_bf16_to_f32 weight) (H.core x i *. real_scale) (x.H.norm_weight i);
  L2.rounded_approx (F.mul normalized (Cast.cast_bf16_to_f32 weight))
    ((H.core x i *. real_scale) *. x.H.norm_weight i);
  let gained = L2.rounded (F.mul normalized (Cast.cast_bf16_to_f32 weight)) in
  Cast.cast_bf16_to_f32_ok gate (x.H.gate i);
  Scalar.silu_approx (Cast.cast_bf16_to_f32 gate) (x.H.gate i);
  Approx.a_mul gained (Scalar.silu (Cast.cast_bf16_to_f32 gate))
    ((H.core x i *. real_scale) *. x.H.norm_weight i) (R.silu (x.H.gate i));
  Cast.cast_f32_to_bf16_ok (F.mul gained (Scalar.silu (Cast.cast_bf16_to_f32 gate))) (H.gated x i)

inline_for_extraction noextract
let writes (before:Transfer.scratch) (weights:Transfer.packed)
  (base:nat{base + 19458208 <= 752393024}) (head:P.index 16) (count:nat{count <= 128})
  : GTot Transfer.scratch =
  Seq.init_ghost 16000 (fun i ->
    if 10400 + head * 128 <= i && i < 10400 + head * 128 + count
    then value before weights base head (i - 10400 - head * 128)
    else Seq.index before i)

inline_for_extraction noextract
let writes_zero (before:Transfer.scratch) (weights:Transfer.packed)
  (base:nat{base + 19458208 <= 752393024}) (head:P.index 16)
  : Lemma (writes before weights base head 0 == before) =
  assert (Seq.equal (writes before weights base head 0) before);
  Seq.lemma_eq_elim (writes before weights base head 0) before

inline_for_extraction noextract
let writes_step (before:Transfer.scratch) (weights:Transfer.packed)
  (base:nat{base + 19458208 <= 752393024}) (head:P.index 16) (i:P.index 128)
  : Lemma (
      Seq.index (writes before weights base head i) (10400 + head * 128 + i) ==
        Seq.index before (10400 + head * 128 + i) /\
      Seq.index (writes before weights base head i) (8192 + head * 128 + i) ==
        Seq.index before (8192 + head * 128 + i) /\
      Seq.upd (writes before weights base head i) (10400 + head * 128 + i) (value before weights base head i) ==
      writes before weights base head (i + 1)) =
  assert (Seq.equal
    (Seq.upd (writes before weights base head i) (10400 + head * 128 + i) (value before weights base head i))
    (writes before weights base head (i + 1)));
  Seq.lemma_eq_elim
    (Seq.upd (writes before weights base head i) (10400 + head * 128 + i) (value before weights base head i))
    (writes before weights base head (i + 1))

inline_for_extraction noextract
let complete (scope:prop) (before:Transfer.scratch) (weights:Transfer.packed)
  (base:nat{base + 19458208 <= 752393024}) (head:P.index 16) (x:H.inputs)
  : Lemma
      (requires scope ==> inputs before weights base head x)
      (ensures State.hidden_frame before (writes before weights base head 128) head 128 /\
        (scope ==> forall (i:P.index 128).
          Approx.v_approximates (Seq.index (writes before weights base head 128) (10400 + head * 128 + i))
            (H.gated x i))) =
  let under (_:scope)
    : Lemma (forall (i:P.index 128).
        Approx.v_approximates (Seq.index (writes before weights base head 128) (10400 + head * 128 + i))
          (H.gated x i)) =
    FStar.Classical.forall_intro (value_approx before weights base head x)
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
fn write_scaled
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000)
  (base:sz{SZ.v base + 19458208 <= 752393024}) (head:sz{SZ.v head < 16})
  (scale:FP.t)
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch) (#fraction:perm)
  preserves gpu ** A.pts_to weights #fraction vw
  requires A.pts_to hidden vh
  requires pure (scale == Norm.source_scale 128 (RMS.input 128 vh (10400 + SZ.v head * 128)))
  ensures A.pts_to hidden (writes vh vw (SZ.v base) (SZ.v head) 128)
{
  let output = SZ.add 10400sz (SZ.mul head 128sz);
  let gates = SZ.add 8192sz (SZ.mul head 128sz);
  let weight_offset = SZ.add base (SZ.uint_to_t 19458080);
  writes_zero vh vw (SZ.v base) (SZ.v head);
  let mut i = 0sz;
  while (!i <^ 128sz)
    invariant exists* (j:sz{SZ.v j <= 128}).
      i |-> j ** A.pts_to hidden (writes vh vw (SZ.v base) (SZ.v head) (SZ.v j))
    invariant gpu ** A.pts_to weights #fraction vw
    decreases (128 - SZ.v !i)
  {
    let j = !i;
    writes_step vh vw (SZ.v base) (SZ.v head) (SZ.v j);
    let raw = hidden.(SZ.add output j);
    let normalized = Cast.cast_bf16_to_f32 (Cast.cast_f32_to_bf16 (F.mul (Cast.cast_bf16_to_f32 raw) scale));
    let weight = weights.(SZ.add weight_offset j);
    let gained = Cast.cast_bf16_to_f32 (Cast.cast_f32_to_bf16 (F.mul normalized (Cast.cast_bf16_to_f32 weight)));
    let gate = hidden.(SZ.add gates j);
    hidden.(SZ.add output j) <- Cast.cast_f32_to_bf16 (F.mul gained (Scalar.silu (Cast.cast_bf16_to_f32 gate)));
    i := SZ.add j 1sz;
  }
}

inline_for_extraction noextract
fn apply
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000) (partial:larray FP.t 256)
  (base:sz{SZ.v base + 19458208 <= 752393024}) (head:sz{SZ.v head < 16})
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch) (#vp:erased (lseq FP.t 256)) (#fraction:perm)
  preserves gpu ** A.pts_to weights #fraction vw
  requires A.pts_to hidden vh ** partial |-> vp
  ensures exists* (temporary:lseq FP.t 256). partial |-> temporary **
    A.pts_to hidden (writes vh vw (SZ.v base) (SZ.v head) 128)
{
  let output = SZ.add 10400sz (SZ.mul head 128sz);
  let scale = RMS.scale 128sz hidden output partial;
  write_scaled weights hidden base head scale;
}

inline_for_extraction noextract
fn source_apply
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000)
  (base:sz{SZ.v base + 19458208 <= 752393024}) (head:sz{SZ.v head < 16})
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch) (#fraction:perm)
  preserves gpu ** A.pts_to weights #fraction vw
  requires A.pts_to hidden vh
  ensures A.pts_to hidden (writes vh vw (SZ.v base) (SZ.v head) 128)
{
  let output = SZ.add 10400sz (SZ.mul head 128sz);
  let scale = RMS.source_scale 128sz hidden output;
  write_scaled weights hidden base head scale;
}
