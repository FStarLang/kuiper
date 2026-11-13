module KPlay.Proof.Base.RecurrentScalars

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open FStar.Real
module Seq = FStar.Seq
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module Neg = Kuiper.Float32.NativeNeg
module Cast = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base
module Lit = Kuiper.Float32.Literals
module Exp = FStar.Math.Exp
module Root = FStar.Math.Sqrt
module Softplus = Luce.Softplus
module Model = KPlay.Proof.Shared.Model
module P = KPlay.Proof.Shared.Projection
module O = KPlay.Proof.Shared.Observations
module R = KPlay.Proof.Shared.Recurrent
module Scalar = KPlay.Proof.Base.Scalar
module L2 = KPlay.Proof.Base.L2
module RMS = KPlay.Proof.Base.RMSNative
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
let query_coefficient : FP.t = F.of_literal "0.08838834764831845"

inline_for_extraction noextract
let query_coefficient_approx ()
  : Lemma (Approx.v_approximates query_coefficient (Root.rsqrt 128.0R)) =
  Root.sqrt_square 128.0R;
  Root.sqrt_positive 128.0R;
  assert (88388346.0R /. 1000000000.0R <=. Root.rsqrt 128.0R /\
    Root.rsqrt 128.0R <=. 88388348.0R /. 1000000000.0R);
  assert_norm (Lit.common_normal_cell "0.08838834764831845"
    88388346 1000000000 88388348 1000000000 11863283 (-27));
  FP.literal_interval_approx "0.08838834764831845"
    88388346 1000000000 88388348 1000000000 11863283 (-27) (Root.rsqrt 128.0R)

inline_for_extraction noextract
let beta (value:BF.t) : FP.t =
  L2.rounded (Scalar.sigmoid (Cast.cast_bf16_to_f32 value))

inline_for_extraction noextract
let beta_approx (value:BF.t) (r:real)
  : Lemma
      (requires Approx.v_approximates value r)
      (ensures Approx.v_approximates (beta value) (R.sigmoid r)) =
  Cast.cast_bf16_to_f32_ok value r;
  Scalar.sigmoid_approx (Cast.cast_bf16_to_f32 value) r;
  L2.rounded_approx (Scalar.sigmoid (Cast.cast_bf16_to_f32 value)) (R.sigmoid r)

inline_for_extraction noextract
let decay (a bias log:BF.t) : FP.t =
  let combined = F.add (Cast.cast_bf16_to_f32 a) (Cast.cast_bf16_to_f32 bias) in
  let softplus = Softplus.softplus20 combined in
  F.fexp (F.mul (Neg.neg (F.fexp (Cast.cast_bf16_to_f32 log))) softplus)

inline_for_extraction noextract
let decay_approx (a bias log:BF.t) (ar br lr:real)
  : Lemma
      (requires Approx.v_approximates a ar /\ Approx.v_approximates bias br /\ Approx.v_approximates log lr)
      (ensures Approx.v_approximates (decay a bias log)
        (Exp.exp ((0.0R -. Exp.exp lr) *. R.softplus (ar +. br)))) =
  Cast.cast_bf16_to_f32_ok a ar;
  Cast.cast_bf16_to_f32_ok bias br;
  Cast.cast_bf16_to_f32_ok log lr;
  Approx.a_add (Cast.cast_bf16_to_f32 a) (Cast.cast_bf16_to_f32 bias) ar br;
  let combined = F.add (Cast.cast_bf16_to_f32 a) (Cast.cast_bf16_to_f32 bias) in
  Softplus.softplus20_approx combined (ar +. br);
  Model.authorized_softplus_model (ar +. br);
  Approx.exp_approx (Cast.cast_bf16_to_f32 log) lr;
  Neg.neg_approx (F.fexp (Cast.cast_bf16_to_f32 log)) (Exp.exp lr);
  Approx.a_mul (Neg.neg (F.fexp (Cast.cast_bf16_to_f32 log))) (Softplus.softplus20 combined)
    (0.0R -. Exp.exp lr) (R.softplus (ar +. br));
  Approx.exp_approx (F.mul (Neg.neg (F.fexp (Cast.cast_bf16_to_f32 log))) (Softplus.softplus20 combined))
    ((0.0R -. Exp.exp lr) *. R.softplus (ar +. br))

inline_for_extraction noextract
let query (before:Transfer.scratch) (head:P.index 16) (i:P.index 128) : GTot FP.t =
  let scale = L2.source_scale (RMS.input 128 before (2048 + head * 128)) in
  F.mul (L2.rounded (F.mul (Cast.cast_bf16_to_f32 (Seq.index before (2048 + head * 128 + i))) scale))
    query_coefficient

inline_for_extraction noextract
let key (before:Transfer.scratch) (head:P.index 16) (i:P.index 128) : GTot FP.t =
  let scale = L2.source_scale (RMS.input 128 before (4096 + head * 128)) in
  L2.rounded (F.mul (Cast.cast_bf16_to_f32 (Seq.index before (4096 + head * 128 + i))) scale)

inline_for_extraction noextract
let query_approx (before:Transfer.scratch) (head:P.index 16) (logical:R.vector 128) (i:P.index 128)
  : Lemma
      (requires O.vector_approximates (RMS.input 128 before (2048 + head * 128)) logical)
      (ensures Approx.v_approximates (query before head i)
        (R.l2_normalize logical i *. Root.rsqrt 128.0R)) =
  L2.scale_approx (RMS.input 128 before (2048 + head * 128)) logical;
  let value = Seq.index before (2048 + head * 128 + i) in
  let scale = L2.source_scale (RMS.input 128 before (2048 + head * 128)) in
  let real_scale = Root.rsqrt (R.sum_squares #128 logical +. R.epsilon) in
  Cast.cast_bf16_to_f32_ok value (logical i);
  Approx.a_mul (Cast.cast_bf16_to_f32 value) scale (logical i) real_scale;
  L2.rounded_approx (F.mul (Cast.cast_bf16_to_f32 value) scale) (logical i *. real_scale);
  query_coefficient_approx ();
  Approx.a_mul (L2.rounded (F.mul (Cast.cast_bf16_to_f32 value) scale)) query_coefficient
    (logical i *. real_scale) (Root.rsqrt 128.0R)

inline_for_extraction noextract
let key_approx (before:Transfer.scratch) (head:P.index 16) (logical:R.vector 128) (i:P.index 128)
  : Lemma
      (requires O.vector_approximates (RMS.input 128 before (4096 + head * 128)) logical)
      (ensures Approx.v_approximates (key before head i) (R.l2_normalize logical i)) =
  L2.scale_approx (RMS.input 128 before (4096 + head * 128)) logical;
  let value = Seq.index before (4096 + head * 128 + i) in
  let scale = L2.source_scale (RMS.input 128 before (4096 + head * 128)) in
  let real_scale = Root.rsqrt (R.sum_squares #128 logical +. R.epsilon) in
  Cast.cast_bf16_to_f32_ok value (logical i);
  Approx.a_mul (Cast.cast_bf16_to_f32 value) scale (logical i) real_scale;
  L2.rounded_approx (F.mul (Cast.cast_bf16_to_f32 value) scale) (logical i *. real_scale)
