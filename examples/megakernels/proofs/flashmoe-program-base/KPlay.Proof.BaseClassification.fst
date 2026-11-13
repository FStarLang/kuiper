module KPlay.Proof.BaseClassification

open Kuiper

module S = FStar.Seq
module F = Kuiper.Floating
module B = Kuiper.Floating.Bounds
module Bounds = Kuiper.Float32.Bounds
module Cast = Kuiper.Float.Casts
module I64 = FStar.Int64
module Classical = FStar.Classical
module Projection = KPlay.Proof.Shared.ProjectionSoftmax
module Capacity = KPlay.Proof.Shared.Capacity
module BP = KPlay.Proof.BaseProjection

(* The guarded conclusion preserves the existing unbounded numerical theorem. *)
inline_for_extraction noextract
let source_fma_bound (x w:bf16) (acc result:f32) (k:nat{k < 512})
  : Lemma
      (requires B.fma_post
        (Cast.fcast #bf16 #f32 x) (Cast.fcast #bf16 #f32 w) acc result)
      (ensures
        Projection.finite_bounded_bf16 32L x /\
        Projection.finite_bounded_bf16 4L w /\
        B.magnitude (128 * k) acc ==>
        B.magnitude (128 * (k + 1)) result)
  =
  let bounded (_:(
      Projection.finite_bounded_bf16 32L x /\
      Projection.finite_bounded_bf16 4L w /\
      B.magnitude (128 * k) acc))
    : Lemma (B.magnitude (128 * (k + 1)) result)
    =
    I64.v_inj (I64.int_to_t 32) 32L;
    I64.v_inj (I64.int_to_t 4) 4L;
    Cast.bf16_to_f32_magnitude 32 x;
    Cast.bf16_to_f32_magnitude 4 w
  in
  Classical.impl_intro bounded

inline_for_extraction noextract
let score_row =
  scores:lseq f32 8{
    forall (e:nat{e < 8}). B.magnitude 65536 (S.index scores e)}

inline_for_extraction noextract
let native_scores (scores:score_row) : Capacity.native_row 8 =
  fun e -> S.index scores e

inline_for_extraction noextract
let rec score_scan_matches (scores:score_row) (n:pos{n <= 8})
  : Lemma
      (ensures BP.maximum_index scores n ==
        Capacity.selection_prefix (native_scores scores) n)
      (decreases n)
  = if n = 1 then () else score_scan_matches scores (n - 1)

inline_for_extraction noextract
let maximum_dominates (scores:score_row) (e:nat{e < 8})
  : Lemma (
      F.lte (S.index scores e)
        (S.index scores (BP.maximum_index scores 8)))
  =
  score_scan_matches scores 8;
  Capacity.native_selection_law (native_scores scores) e

inline_for_extraction noextract
let exponential_range (scores:score_row) (e:nat{e < 8})
  : Lemma (B.unit (S.index (BP.exponentials scores) e))
  =
  let score = S.index scores e in
  let maximum = S.index scores (BP.maximum_index scores 8) in
  maximum_dominates scores e;
  Bounds.sub_bounds score maximum;
  assert (B.magnitude 131072 (F.sub score maximum));
  Bounds.exp_nonpositive (F.sub score maximum)

inline_for_extraction noextract
let exponentials_range (scores:score_row)
  : Lemma (
      (forall (e:nat{e < 8}). B.unit (S.index (BP.exponentials scores) e)) /\
      S.index (BP.exponentials scores) (BP.maximum_index scores 8) == F.one)
  =
  Classical.forall_intro (exponential_range scores);
  let maximum = S.index scores (BP.maximum_index scores 8) in
  F.eq_spec maximum maximum;
  Bounds.sub_bounds maximum maximum;
  Bounds.exp_nonpositive (F.sub maximum maximum)

inline_for_extraction noextract
let sum_step (sum term:f32) (k:nat{k < 8})
  : Lemma
      (requires B.nonnegative k sum /\ B.unit term)
      (ensures
        B.nonnegative (k + 1) (F.add sum term) /\
        F.lte sum (F.add sum term) /\ F.lte term (F.add sum term))
  =
  B.nonnegative_magnitude k sum;
  B.nonnegative_unit term;
  B.nonnegative_magnitude 1 term;
  Bounds.add_bounds k 1 sum term

inline_for_extraction noextract
let rec sum_prefix_range (terms:lseq f32 8) (n:nat{n <= 8})
  : Lemma
      (requires forall (e:nat{e < 8}). B.unit (S.index terms e))
      (ensures
        B.nonnegative n (BP.sum_prefix terms n) /\
        (forall (e:nat{e < n}). F.lte (S.index terms e) (BP.sum_prefix terms n)))
      (decreases n)
  =
  if n = 0 then Bounds.zero ()
  else (
    sum_prefix_range terms (n - 1);
    let previous = BP.sum_prefix terms (n - 1) in
    let term = S.index terms (n - 1) in
    sum_step previous term (n - 1);
    let cell (e:nat{e < n})
      : Lemma (F.lte (S.index terms e) (BP.sum_prefix terms n))
      =
      if e = n - 1 then ()
      else Capacity.native_order3 (S.index terms e) previous (F.add previous term)
    in
    Classical.forall_intro cell)

inline_for_extraction noextract
let denominator_range (scores:score_row)
  : Lemma (
      B.nonnegative 8 (BP.sum_prefix (BP.exponentials scores) 8) /\
      F.lte F.one (BP.sum_prefix (BP.exponentials scores) 8))
  =
  exponentials_range scores;
  sum_prefix_range (BP.exponentials scores) 8

inline_for_extraction noextract
let normalized_range (scores:score_row)
  : Lemma (
      forall (e:nat{e < 8}). B.unit (S.index (BP.normalized_values scores) e))
  =
  exponentials_range scores;
  denominator_range scores;
  let terms = BP.exponentials scores in
  let sum = BP.sum_prefix terms 8 in
  Bounds.one ();
  Bounds.div_unit F.one sum;
  let cell (e:nat{e < 8})
    : Lemma (B.unit (S.index (BP.normalized_values scores) e))
    = Bounds.mul_unit (S.index terms e) (F.div F.one sum)
  in
  Classical.forall_intro cell
