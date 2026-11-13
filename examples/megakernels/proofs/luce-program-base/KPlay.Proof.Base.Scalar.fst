module KPlay.Proof.Base.Scalar

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
module F = Kuiper.Floating
module Neg = Kuiper.Float32.NativeNeg
module Cast = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base
module Exp = FStar.Math.Exp
module R = KPlay.Proof.Shared.Recurrent
module P = KPlay.Proof.Shared.Projection
module Transfer = KPlay.Proof.Base.Transfer
module L2 = KPlay.Proof.Base.L2

inline_for_extraction noextract
let silu (x:FP.t) : FP.t =
  F.div x (F.add (F.one #FP.t) (F.fexp (Neg.neg x)))

inline_for_extraction noextract
let sigmoid (x:FP.t) : FP.t =
  F.div (F.one #FP.t) (F.add (F.one #FP.t) (F.fexp (Neg.neg x)))

inline_for_extraction noextract
let denominator_approx (x:FP.t) (r:real)
  : Lemma
      (requires Approx.v_approximates x r)
      (ensures Approx.v_approximates
        (F.add (F.one #FP.t) (F.fexp (Neg.neg x))) (1.0R +. Exp.exp (0.0R -. r)) /\
        1.0R +. Exp.exp (0.0R -. r) >. 0.0R) =
  Neg.neg_approx x r;
  Approx.exp_approx (Neg.neg x) (0.0R -. r);
  Exp.exp_positive (0.0R -. r);
  Approx.a_add (F.one #FP.t) (F.fexp (Neg.neg x)) 1.0R (Exp.exp (0.0R -. r))

inline_for_extraction noextract
let silu_approx (x:FP.t) (r:real)
  : Lemma
      (requires Approx.v_approximates x r)
      (ensures Approx.v_approximates (silu x) (R.silu r)) =
  denominator_approx x r;
  Approx.div_approx x (F.add (F.one #FP.t) (F.fexp (Neg.neg x)))
    r (1.0R +. Exp.exp (0.0R -. r));
  assert (r /. (1.0R +. Exp.exp (0.0R -. r)) ==
    r *. (1.0R /. (1.0R +. Exp.exp (0.0R -. r))))

inline_for_extraction noextract
let sigmoid_approx (x:FP.t) (r:real)
  : Lemma
      (requires Approx.v_approximates x r)
      (ensures Approx.v_approximates (sigmoid x) (R.sigmoid r)) =
  denominator_approx x r;
  Approx.div_approx (F.one #FP.t) (F.add (F.one #FP.t) (F.fexp (Neg.neg x)))
    1.0R (1.0R +. Exp.exp (0.0R -. r))

type operation = | Residual | Mlp | AttentionGate

inline_for_extraction noextract
let width (op:operation) : pos =
  match op with | Residual -> 1024 | Mlp -> 3584 | AttentionGate -> 2048

inline_for_extraction noextract
let destination (op:operation) : nat =
  match op with | Residual -> 0 | Mlp -> 2048 | AttentionGate -> 10400

inline_for_extraction noextract
let left_offset (op:operation) : nat =
  match op with | Residual -> 0 | Mlp -> 2048 | AttentionGate -> 8192

inline_for_extraction noextract
let right_offset (op:operation) : nat =
  match op with | Residual -> 12500 | Mlp -> 5632 | AttentionGate -> 10400

inline_for_extraction noextract
let value (op:operation) (left right:BF.t) : BF.t =
  let x = Cast.cast_bf16_to_f32 left in
  let y = Cast.cast_bf16_to_f32 right in
  match op with
  | Residual -> Cast.cast_f32_to_bf16 (F.add x y)
  | Mlp -> Cast.cast_f32_to_bf16 (F.mul (L2.rounded (silu x)) y)
  | AttentionGate -> Cast.cast_f32_to_bf16 (F.mul y (L2.rounded (sigmoid x)))

inline_for_extraction noextract
let logical (op:operation) (left right:real) : GTot real =
  match op with
  | Residual -> left +. right
  | Mlp -> R.silu left *. right
  | AttentionGate -> right *. R.sigmoid left

inline_for_extraction noextract
let value_approx (op:operation) (left right:BF.t) (x y:real)
  : Lemma
      (requires Approx.v_approximates left x /\ Approx.v_approximates right y)
      (ensures Approx.v_approximates (value op left right) (logical op x y)) =
  Cast.cast_bf16_to_f32_ok left x;
  Cast.cast_bf16_to_f32_ok right y;
  let a = Cast.cast_bf16_to_f32 left in
  let b = Cast.cast_bf16_to_f32 right in
  match op with
  | Residual ->
      Approx.a_add a b x y;
      Cast.cast_f32_to_bf16_ok (F.add a b) (x +. y)
  | Mlp ->
      silu_approx a x;
      L2.rounded_approx (silu a) (R.silu x);
      Approx.a_mul (L2.rounded (silu a)) b (R.silu x) y;
      Cast.cast_f32_to_bf16_ok (F.mul (L2.rounded (silu a)) b) (R.silu x *. y)
  | AttentionGate ->
      sigmoid_approx a x;
      L2.rounded_approx (sigmoid a) (R.sigmoid x);
      Approx.a_mul b (L2.rounded (sigmoid a)) y (R.sigmoid x);
      Cast.cast_f32_to_bf16_ok (F.mul b (L2.rounded (sigmoid a))) (y *. R.sigmoid x)

inline_for_extraction noextract
let bounds (op:operation)
  : Lemma (
      destination op + width op <= 16000 /\
      left_offset op + width op <= 16000 /\
      right_offset op + width op <= 16000) = ()

inline_for_extraction noextract
let writes (op:operation) (before:Transfer.scratch) (count:nat{count <= width op})
  : GTot Transfer.scratch =
  Seq.init_ghost 16000 (fun i ->
    if destination op <= i && i < destination op + count then
      let j = i - destination op in
      value op (Seq.index before (left_offset op + j)) (Seq.index before (right_offset op + j))
    else Seq.index before i)

inline_for_extraction noextract
let writes_zero (op:operation) (before:Transfer.scratch)
  : Lemma (writes op before 0 == before) =
  assert (Seq.equal (writes op before 0) before);
  Seq.lemma_eq_elim (writes op before 0) before

inline_for_extraction noextract
let writes_step (op:operation) (before:Transfer.scratch) (i:P.index (width op))
  : Lemma (
      Seq.index (writes op before i) (left_offset op + i) == Seq.index before (left_offset op + i) /\
      Seq.index (writes op before i) (right_offset op + i) == Seq.index before (right_offset op + i) /\
      Seq.upd (writes op before i) (destination op + i)
        (value op (Seq.index before (left_offset op + i)) (Seq.index before (right_offset op + i))) ==
      writes op before (i + 1)) =
  bounds op;
  assert (Seq.equal
    (Seq.upd (writes op before i) (destination op + i)
      (value op (Seq.index before (left_offset op + i)) (Seq.index before (right_offset op + i))))
    (writes op before (i + 1)));
  Seq.lemma_eq_elim
    (Seq.upd (writes op before i) (destination op + i)
      (value op (Seq.index before (left_offset op + i)) (Seq.index before (right_offset op + i))))
    (writes op before (i + 1))

inline_for_extraction noextract
fn apply
  (op:operation) (hidden:larray BF.t 16000) (#before:erased Transfer.scratch)
  preserves gpu
  requires A.pts_to hidden before
  ensures A.pts_to hidden (writes op before (width op))
{
  bounds op;
  writes_zero op before;
  let n = SZ.uint_to_t (width op);
  let left = SZ.uint_to_t (left_offset op);
  let right = SZ.uint_to_t (right_offset op);
  let output = SZ.uint_to_t (destination op);
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (j:sz{SZ.v j <= width op}).
      i |-> j ** A.pts_to hidden (writes op before (SZ.v j))
    invariant gpu
    decreases (width op - SZ.v !i)
  {
    let j = !i;
    writes_step op before (SZ.v j);
    let a = hidden.(SZ.add left j);
    let b = hidden.(SZ.add right j);
    hidden.(SZ.add output j) <- value op a b;
    i := SZ.add j 1sz;
  }
}

inline_for_extraction noextract
let outside_frame (op:operation) (before:Transfer.scratch) (i:P.index 16000)
  : Lemma (
      ~(destination op <= i /\ i < destination op + width op) ==>
      Seq.index (writes op before (width op)) i == Seq.index before i) = ()
