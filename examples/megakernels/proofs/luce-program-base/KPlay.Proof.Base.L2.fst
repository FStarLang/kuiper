module KPlay.Proof.Base.L2

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open FStar.Real
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module Cast = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base
module Sum = Kuiper.Sum
module Root = FStar.Math.Sqrt
module P = KPlay.Proof.Shared.Projection
module O = KPlay.Proof.Shared.Observations
module R = KPlay.Proof.Shared.Recurrent
module Norm = KPlay.Proof.Base.Norm
module Math = KPlay.Proof.Base.ReductionMath

inline_for_extraction noextract
let rounded (x:FP.t) : FP.t =
  Cast.cast_bf16_to_f32 (Cast.cast_f32_to_bf16 x)

inline_for_extraction noextract
let rounded_approx (x:FP.t) (r:real)
  : Lemma
      (requires Approx.v_approximates x r)
      (ensures Approx.v_approximates (rounded x) r) =
  Cast.cast_f32_to_bf16_ok x r;
  Cast.cast_bf16_to_f32_ok (Cast.cast_f32_to_bf16 x) r

inline_for_extraction noextract
let rec group (x:P.vector BF.t 128) (g:P.index 16) (count:nat{count <= 8})
  : GTot FP.t (decreases count) =
  if count = 0 then F.zero #FP.t
  else F.add (group x g (count - 1)) (rounded (Norm.square (x (8 * g + count - 1))))

inline_for_extraction noextract
let grouped (x:P.vector BF.t 128) : P.vector FP.t 256 =
  fun i -> if i < 16 then group x i 8 else F.zero #FP.t

inline_for_extraction noextract
let source_sum (x:P.vector BF.t 128) : GTot FP.t =
  Norm.tree 16 (grouped x)

inline_for_extraction noextract
let source_scale (x:P.vector BF.t 128) : GTot FP.t =
  rounded (F.rsqrt (rounded (F.add (rounded (source_sum x)) Norm.epsilon)))

inline_for_extraction noextract
let rec group_approx (x:P.vector BF.t 128) (r:P.vector real 128)
  (g:P.index 16) (count:nat{count <= 8})
  : Lemma
      (requires O.vector_approximates x r)
      (ensures Approx.v_approximates (group x g count)
        (Sum.sum (8 * g) (8 * g + count) (Norm.squares #128 r)))
      (decreases count) =
  if count > 0 then (
    group_approx x r g (count - 1);
    let i = 8 * g + count - 1 in
    Norm.square_approx (x i) (r i);
    rounded_approx (Norm.square (x i)) (r i *. r i);
    Sum.sum_pop_right (8 * g) (8 * g + count) (Norm.squares #128 r);
    Approx.a_add (group x g (count - 1)) (rounded (Norm.square (x i)))
      (Sum.sum (8 * g) (8 * g + count - 1) (Norm.squares #128 r)) (r i *. r i)
  )

inline_for_extraction noextract
let real_groups (r:P.vector real 128) : P.vector real 256 =
  fun i -> if i < 16 then Sum.sum (8 * i) (8 * i + 8) (Norm.squares #128 r) else 0.0R

inline_for_extraction noextract
let rec groups_sum (r:P.vector real 128) (count:nat{count <= 16})
  : Lemma (
      Sum.sum 0 count (real_groups r) == Sum.sum 0 (8 * count) (Norm.squares #128 r))
      (decreases count) =
  if count > 0 then (
    groups_sum r (count - 1);
    Sum.sum_pop_right 0 count (real_groups r);
    Sum.sum_split 0 (8 * count) (Norm.squares #128 r) (8 * (count - 1))
  )

inline_for_extraction noextract
let sum_formula (r:P.vector real 128)
  : Lemma (Norm.partial_sum 16 (real_groups r) == R.sum_squares #128 r) =
  Math.sum_kuiper_prefix #256 (real_groups r) 16;
  groups_sum r 16;
  assert (R.sum_squares #128 r == Sum.sum 0 128 (Norm.squares #128 r)) by (
    FStar.Tactics.V2.norm [zeta; iota; delta_only [
      "KPlay.Proof.Shared.Recurrent.sum_squares"; "KPlay.Proof.Base.Norm.squares"]];
    FStar.Tactics.V2.trefl ());
  norm_spec [delta_once ["KPlay.Proof.Base.Norm.partial_sum"]]
    (Norm.partial_sum 16 (real_groups r))

inline_for_extraction noextract
let source_sum_approx (x:P.vector BF.t 128) (r:P.vector real 128)
  : Lemma
      (requires O.vector_approximates x r)
      (ensures Approx.v_approximates (source_sum x) (R.sum_squares #128 r)) =
  let at_index (i:P.index 16)
    : Lemma (Approx.v_approximates (grouped x i) (real_groups r i)) =
    group_approx x r i 8
  in
  FStar.Classical.forall_intro at_index;
  Norm.tree_approx 16 (grouped x) (real_groups r);
  sum_formula r

inline_for_extraction noextract
let scale_approx (x:P.vector BF.t 128) (r:P.vector real 128)
  : Lemma
      (requires O.vector_approximates x r)
      (ensures (
        let sum = R.sum_squares #128 r in
        Approx.v_approximates (source_scale x) (Root.rsqrt (sum +. R.epsilon)))) =
  let sum = R.sum_squares #128 r in
  source_sum_approx x r;
  rounded_approx (source_sum x) sum;
  Norm.epsilon_approx ();
  Approx.a_add (rounded (source_sum x)) Norm.epsilon sum R.epsilon;
  rounded_approx (F.add (rounded (source_sum x)) Norm.epsilon) (sum +. R.epsilon);
  Approx.rsqrt_approx (rounded (F.add (rounded (source_sum x)) Norm.epsilon)) (sum +. R.epsilon);
  rounded_approx (F.rsqrt (rounded (F.add (rounded (source_sum x)) Norm.epsilon)))
    (Root.rsqrt (sum +. R.epsilon))
