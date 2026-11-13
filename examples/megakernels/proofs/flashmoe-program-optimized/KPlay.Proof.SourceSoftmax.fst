module KPlay.Proof.SourceSoftmax
#lang-pulse

open Kuiper
open Kuiper.Seq.Common

module S = FStar.Seq
module C = Kuiper.Chest
module F = Kuiper.Floating
module A = Kuiper.Approximates.Base
module B = Kuiper.Floating.Bounds
module Bounds = Kuiper.Float32.Bounds
module FM = Kuiper.Float32.FastMath
module Math = KPlay.Proof.OnlineMath
module Class = KPlay.Proof.OnlineClassification
module Order = KPlay.Proof.Shared.Capacity
module SM = Kuiper.Spec.Softmax
module I32 = FStar.Int32
module IC = FStar.Int.Cast
module SZ = Kuiper.SizeT
module PA = Pulse.Lib.Array

inline_for_extraction noextract
let index (value:i32{0 <= I32.v value /\ I32.v value <= 2048})
  : result:sz{SZ.v result == I32.v value} =
  SZ.uint32_to_sizet (IC.int32_to_uint32 value)

noextract type score_row =
  values:lseq f32 8{forall (e:natlt 8). B.magnitude 65536 (S.index values e)}

noextract type probability_row =
  values:lseq f32 8{forall (e:natlt 8). B.unit (S.index values e)}

inline_for_extraction noextract
let maximum_extends (scores:score_row) (n:nat{n < 8}) (old:f32)
  : Lemma
      (requires
        (old == Class.negative_infinity \/ B.magnitude 65536 old) /\
        (forall (j:natlt 8). j < n ==> F.lte (S.index scores j) old))
      (ensures (
        let next = F.fmax old (S.index scores n) in
        B.magnitude 65536 next /\
        (forall (j:natlt 8). j < n + 1 ==> F.lte (S.index scores j) next)))
  =
  Class.maximum_step old (S.index scores n);
  F.neg_kind #f32 F.infinity;
  let cell (j:natlt 8)
    : Lemma (j < n + 1 ==> F.lte (S.index scores j) (F.fmax old (S.index scores n)))
    =
    if j < n then
      Order.native_order3 (S.index scores j) old (F.fmax old (S.index scores n))
  in FStar.Classical.forall_intro cell

inline_for_extraction noextract
fn source_normalizer (row:larray f32 8) (#scores:erased score_row)
  preserves gpu ** PA.pts_to row scores
  returns md:(f32 & f32)
  ensures pure (
    B.magnitude 65536 (fst md) /\
    B.nonnegative 8 (snd md) /\ F.lte F.one (snd md) /\
    (forall (j:natlt 8). F.lte (S.index scores j) (fst md)) /\
    (forall (r:lseq real 8).
      seq_approximates scores r ==> A.v_approximates (snd md) (Math.denominator r 8 (fst md))))
{
  Bounds.zero ();
  let mut maximum:f32 = Class.negative_infinity;
  let mut denominator:f32 = F.zero;
  let mut expert = 0l;
  while (I32.lt !expert 8l)
    invariant exists* (n:i32{0 <= I32.v n /\ I32.v n <= 8}) (m d:f32).
      expert |-> n ** maximum |-> m ** denominator |-> d **
      pure (
        (if I32.v n = 0 then m == Class.negative_infinity /\ d == F.zero
         else B.magnitude 65536 m /\ F.lte F.one d) /\
        B.nonnegative (I32.v n) d /\
        (forall (j:natlt 8). j < I32.v n ==> F.lte (S.index scores j) m) /\
        (forall (r:lseq real 8).
          seq_approximates scores r ==>
          A.v_approximates d (Math.denominator r (I32.v n) m)))
    decreases (8 - I32.v !expert)
  {
    let n = !expert;
    let old = !maximum;
    let previous = !denominator;
    let x = row.(index n);
    let next = F.fmax old x;
    maximum_extends scores (I32.v n) old;
    Class.maximum_step old x;
    A.to_real_ok old;
    A.to_real_ok next;
    let delta_m = FM.sub_rn old next;
    let adjustment = FM.exp delta_m;
    if (I32.eq n 0l) {
      Class.seed_shift next delta_m adjustment;
      ()
    } else {
      Class.exp_shift old next delta_m adjustment;
      ()
    };
    let delta_x = FM.sub_rn x next;
    let numerator = FM.exp delta_x;
    Class.exp_shift x next delta_x numerator;
    let result = FM.fma_rn previous adjustment numerator;
    Class.denominator_step (I32.v n) old next x previous adjustment numerator result;
    assert pure (
      forall (r:lseq real 8).
        seq_approximates scores r ==>
        A.v_approximates result
          (Math.denominator r (I32.v n) old *.
            Kuiper.Real.exp (A.to_real old -. A.to_real next) +.
            Kuiper.Real.exp (S.index r (I32.v n) -. A.to_real next)));
    Math.denominator_step_all #8 (I32.v n) old next;
    maximum := next;
    denominator := result;
    expert := I32.add n 1l;
  };
  (!maximum, !denominator)
}

inline_for_extraction noextract
fn source_normalize (row:larray f32 8) (#scores:erased score_row)
  preserves gpu
  requires PA.pts_to row scores
  returns probabilities:erased probability_row
  ensures PA.pts_to row probabilities **
    pure (forall (r:lseq real 8).
      seq_approximates scores r ==>
      seq_approximates probabilities (SM.softmax_real_seq r))
{
  let (maximum, denominator) = source_normalizer row #scores;
  A.to_real_ok maximum;
  let mut expert = 0l;
  while (I32.lt !expert 8l)
    invariant exists* (n:i32{0 <= I32.v n /\ I32.v n <= 8}) (values:lseq f32 8).
      expert |-> n ** PA.pts_to row values **
      pure (
        (forall (j:natlt 8).
          if j < I32.v n then B.unit (S.index values j)
          else S.index values j == S.index scores j) /\
        (forall (r:lseq real 8) (j:natlt 8).
          seq_approximates scores r /\ j < I32.v n ==>
          A.v_approximates (S.index values j) (S.index (SM.softmax_real_seq r) j)))
    decreases (8 - I32.v !expert)
  {
    let n = !expert;
    let score = row.(index n);
    let delta = FM.sub_rn score maximum;
    let numerator = FM.exp delta;
    Class.exp_shift score maximum delta numerator;
    let probability = FM.divide numerator denominator;
    Math.normalized_value_all #8 maximum (I32.v n);
    assert pure (B.unit probability);
    assert pure (
      forall (r:lseq real 8).
        seq_approximates scores r ==>
        A.v_approximates probability (S.index (SM.softmax_real_seq r) (I32.v n)));
    row.(index n) <- probability;
    expert := I32.add n 1l;
  };
  with probabilities. assert PA.pts_to row probabilities;
  FStar.Ghost.hide (probabilities <: probability_row)
}
