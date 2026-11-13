module KPlay.Functional

#lang-pulse

open Kuiper
open Pulse.Lib.Send

module Shared = KPlay.SharedSpec
module Projection = KPlay.Proof.Shared.ProjectionSoftmax
module Argmax = KPlay.Proof.Shared.Argmax
module Capacity = KPlay.Proof.Shared.Capacity
module Native = KPlay.Proof.Shared.Native
module BP = KPlay.Proof.BaseProjection
module BC = KPlay.Proof.BaseCapacity
module Classification = KPlay.Proof.BaseClassification
module History = KPlay.Proof.BaseHistory
module Storage = KPlay.Proof.BaseStorage
module Caller = KPlay.Proof.BaseCaller
module SourceNative = KPlay.Proof.BaseNative
module SourceReturn = KPlay.Proof.BaseReturn
module Epoch = Kuiper.Epoch
module Stream = Kuiper.Kernel.Stream
module C = Kuiper.Chest
module S = FStar.Seq
module A = Kuiper.Approximates.Base
module F = Kuiper.Floating
module B = Kuiper.Floating.Bounds
module Bounds = Kuiper.Float32.Bounds
module Cast = Kuiper.Float.Casts
module Fast = Kuiper.Float32.FastMath
module GEMM = Kuiper.Spec.GEMM
module I32 = FStar.Int32
module I64 = FStar.Int64
module IC = FStar.Int.Cast
module SZ = Kuiper.SizeT
module PA = Pulse.Lib.Array
module KA = Kuiper.Array
module Input = Kuiper.Tensor.Input
module RT = Kuiper.Tensor.Runtime
module Results = Kuiper.Tensor.Results
module Pledge = Pulse.Lib.Pledge
module Classical = FStar.Classical

inline_for_extraction noextract
let input_models_are_bound tokens weights xs ws x w =
  BP.chest2_extensional (token_model xs) x;
  BP.chest2_extensional (weight_model ws) w;
  Input.view_metadata tokens;
  Input.view_metadata weights

inline_for_extraction noextract
let token_row_cell (xs:token_values) (t:Shared.token_index)
  (h:nat{h < 512})
  : Lemma (
      S.index (token_row xs t) h == S.index xs (t * 512 + h) /\
      S.index (token_row xs t) h == C.acc2 (token_model xs) t h)
  = S.lemma_index_slice xs (t * 512) (t * 512 + 512) h

inline_for_extraction noextract
let source_row_admissible xs ws t =
  let token_cell (h:nat{h < 512})
    : Lemma (Projection.finite_bounded_bf16 32L (S.index (token_row xs t) h))
    = token_row_cell xs t h
  in
  let weight_cell (i:nat{i < 8 * 512})
    : Lemma (Projection.finite_bounded_bf16 4L (S.index ws i))
    =
    assert (i / 512 < 8 /\ i % 512 < 512);
    assert (i == (i / 512) * 512 + i % 512);
    assert (S.index ws i == C.acc2 (weight_model ws) (i % 512) (i / 512))
  in
  Classical.forall_intro token_cell;
  Classical.forall_intro weight_cell

inline_for_extraction noextract
let projection_row_matches xs ws t =
  Classical.forall_intro (token_row_cell xs t);
  let left = BP.projection_row #512 #8 (token_row xs t) ws in
  let right =
    C.chest2_row (Projection.projection true (token_model xs) (weight_model ws)) t in
  let cell (e:Shared.expert_index)
    : Lemma (C.acc1 left e == C.acc1 right e)
    =
    Projection.projection_entry true (token_model xs) (weight_model ws) t e;
    GEMM.__gmatmul_single_congr 0.0R ( *. ) ( +. )
      (C.to_real_chest (BP.token_view (token_row xs t)))
      (C.to_real_chest (BP.weight_view #512 #8 ws))
      (C.to_real_chest (token_model xs)) (C.to_real_chest (weight_model ws))
      0 e t e 512
  in
  Classical.forall_intro cell;
  BP.chest1_extensional left right

inline_for_extraction noextract
let projection_cell_matches (xs:token_values) (ws:weight_values)
  (t:Shared.token_index) (e:Shared.expert_index)
  : Lemma (
      BP.projection_prefix #512 #8 (token_row xs t) ws e 512 ==
        C.acc2 (Projection.projection true (token_model xs) (weight_model ws)) t e)
  =
  projection_row_matches xs ws t;
  assert (C.acc1 (BP.projection_row #512 #8 (token_row xs t) ws) e ==
    BP.projection_prefix #512 #8 (token_row xs t) ws e 512);
  assert (C.acc1
    (C.chest2_row (Projection.projection true (token_model xs) (weight_model ws)) t) e ==
    C.acc2 (Projection.projection true (token_model xs) (weight_model ws)) t e)

inline_for_extraction noextract
let source_softmax_gather xs ws t scores =
  projection_row_matches xs ws t;
  BP.normalization_models scores (BP.projection_row #512 #8 (token_row xs t) ws);
  let row = BP.normalized_values scores in
  let chosen = BP.maximum_index row 8 in
  Shared.reference_probability_positive (token_model xs) (weight_model ws) t chosen;
  let model = C.acc2
    (Projection.projection_softmax true (token_model xs) (weight_model ws)) t chosen in
  assert (A.v_approximates (S.index row chosen) model);
  BC.selected_weight_represents (S.index row chosen) model

inline_for_extraction noextract
let rec scan_matches_family (row:classified_row) (n:pos{n <= 8})
  : Lemma
      (ensures
        BP.maximum_index row n == Argmax.best_prefix (argmax_row row) n)
      (decreases n)
  = if n = 1 then () else scan_matches_family row (n - 1)

inline_for_extraction noextract
let source_selection_compatible row =
  scan_matches_family row 8;
  Argmax.best_prefix_correct (argmax_row row) 8;
  Shared.selector_compatibility (argmax_row row) 8

inline_for_extraction noextract
let source_update_is_native token chosen row before =
  let rank = I64.add (S.index before.BC.counts chosen) 1L in
  BC.updated_state_is_native Capacity.Smoke token chosen row
    (BC.selected_weight (S.index row chosen)) before rank

inline_for_extraction noextract
let source_update_refines_prefix b token selected row before =
  BC.incoming_count_bounds Capacity.Smoke b (I64.v token) before;
  let chosen = b.Capacity.selected_expert (I64.v token) in
  source_update_is_native token chosen row before;
  BC.native_step_refines_prefix Capacity.Smoke b (I64.v token)
    selected row before (source_after token chosen row before)

inline_for_extraction noextract
let first_token_transition row e c =
  let chosen = BP.maximum_index row 8 in
  source_update_is_native 0L chosen row first_live_state;
  BC.packed_offset_injective 32 e chosen c 0

inline_for_extraction noextract
let scan_successor (row:lseq f32 8) (n:pos{n < 8})
  : Lemma (
      BP.maximum_index row (n + 1) ==
        (if F.lt (S.index row (BP.maximum_index row n)) (S.index row n)
         then n else BP.maximum_index row n))
  = ()

inline_for_extraction noextract
fn source_fma_step
  (token:sz{SZ.v token < 256})
  (expert:sz{SZ.v expert < 8})
  (hidden:sz{SZ.v hidden < 512})
  (tokens:larray bf16 (256 * 512))
  (weights:larray bf16 (8 * 512))
  (#token_perm #weight_perm:perm)
  (#xs:erased token_values) (#ws:erased weight_values)
  (accumulator:f32)
  preserves gpu **
    PA.pts_to tokens #token_perm xs ** PA.pts_to weights #weight_perm ws
  requires pure (A.v_approximates accumulator
    (BP.projection_prefix #512 #8 (token_row xs (SZ.v token)) ws
      (SZ.v expert) (SZ.v hidden)))
  returns next:f32
  ensures pure (
    A.v_approximates next
      (BP.projection_prefix #512 #8 (token_row xs (SZ.v token)) ws
        (SZ.v expert) (SZ.v hidden + 1)) /\
    (Shared.input_domain (token_model xs) (weight_model ws) /\
      B.magnitude (128 * SZ.v hidden) accumulator ==>
      B.magnitude (128 * (SZ.v hidden + 1)) next))
{
  let xv = tokens.(token *^ 512sz +^ hidden);
  let wv = weights.(expert *^ 512sz +^ hidden);
  let xf = Cast.fcast #bf16 #f32 xv;
  let wf = Cast.fcast #bf16 #f32 wv;
  Shared.bf16_promotion_model xv;
  Shared.bf16_promotion_model wv;
  token_row_cell xs (SZ.v token) (SZ.v hidden);
  Classical.move_requires (source_row_admissible xs ws) (SZ.v token);
  BP.projection_prefix_step #512 #8 (token_row xs (SZ.v token)) ws (SZ.v expert) (SZ.v hidden);
  let next = Fast.fma_rn xf wf accumulator;
  Classification.source_fma_bound xv wv accumulator next (SZ.v hidden);
  next
}

inline_for_extraction noextract
fn source_first_fma
  (token:sz{SZ.v token < 256}) (expert:sz{SZ.v expert < 8})
  (tokens:larray bf16 (256 * 512)) (weights:larray bf16 (8 * 512))
  (#token_perm #weight_perm:perm)
  (#xs:erased token_values) (#ws:erased weight_values)
  preserves gpu **
    PA.pts_to tokens #token_perm xs ** PA.pts_to weights #weight_perm ws
  returns next:f32
  ensures pure (A.v_approximates next
    (BP.projection_prefix #512 #8 (token_row xs (SZ.v token)) ws (SZ.v expert) 1))
{
  BP.projection_prefix_zero #512 #8 (token_row xs (SZ.v token)) ws (SZ.v expert);
  source_fma_step token expert 0sz tokens weights #token_perm #weight_perm #xs #ws
    (F.zero #f32)
}

inline_for_extraction noextract
fn source_dot_product
  (token:sz{SZ.v token < 256}) (expert:sz{SZ.v expert < 8})
  (tokens:larray bf16 (256 * 512)) (weights:larray bf16 (8 * 512))
  (#token_perm #weight_perm:perm)
  (#xs:erased token_values) (#ws:erased weight_values)
  preserves gpu **
    PA.pts_to tokens #token_perm xs ** PA.pts_to weights #weight_perm ws
  returns score:f32
  ensures pure (
    A.v_approximates score
      (C.acc2 (Projection.projection true (token_model xs) (weight_model ws))
        (SZ.v token) (SZ.v expert)) /\
    (Shared.input_domain (token_model xs) (weight_model ws) ==>
      B.magnitude 65536 score))
{
  BP.projection_prefix_zero #512 #8 (token_row xs (SZ.v token)) ws (SZ.v expert);
  Bounds.zero ();
  let mut hidden = 0l;
  let mut accumulator:f32 = F.zero #f32;
  while (I32.lt !hidden 512l)
    invariant exists* (h:i32{0 <= I32.v h /\ I32.v h <= 512}) (value:f32).
      hidden |-> h ** accumulator |-> value **
      pure (
        A.v_approximates value
          (BP.projection_prefix #512 #8 (token_row xs (SZ.v token)) ws
            (SZ.v expert) (I32.v h)) /\
        (Shared.input_domain (token_model xs) (weight_model ws) ==>
          B.magnitude (128 * I32.v h) value))
    decreases (512 - I32.v !hidden)
  {
    let h = !hidden;
    let value = !accumulator;
    let next = source_fma_step token expert (i32_index h)
      tokens weights #token_perm #weight_perm #xs #ws value;
    accumulator := next;
    hidden := I32.add h 1l;
  };
  let score = !accumulator;
  projection_cell_matches xs ws (SZ.v token) (SZ.v expert);
  score
}

inline_for_extraction noextract
fn source_select
  (row:larray f32 8) (#row_perm:perm) (#values:erased (lseq f32 8))
  preserves gpu ** PA.pts_to row #row_perm values
  returns selected:i32
  ensures pure (
    0 <= I32.v selected /\ I32.v selected < 8 /\
    I32.v selected == BP.maximum_index values 8)
{
  let mut selected = 0l;
  let mut expert = 1l;
  while (I32.lt !expert 8l)
    invariant exists* (j:i32{1 <= I32.v j /\ I32.v j <= 8})
      (best:i32{0 <= I32.v best /\ I32.v best < I32.v j}).
      expert |-> j ** selected |-> best **
      pure (I32.v best == BP.maximum_index values (I32.v j))
    decreases (8 - I32.v !expert)
  {
    let j = !expert;
    let best = !selected;
    let previous = row.(i32_index best);
    let candidate = row.(i32_index j);
    scan_successor values (I32.v j);
    let next:i32 =
      (if F.lt previous candidate then (j <: i32) else (best <: i32));
    selected := next;
    expert := I32.add j 1l;
  };
  !selected
}

inline_for_extraction noextract
fn source_capacity_step
  (token:i64{0 <= I64.v token /\ I64.v token < 256})
  (selected:i32{0 <= I32.v selected /\ I32.v selected < 8})
  (row:larray f32 8) (#row_perm:perm) (#values:erased (lseq f32 8))
  (counts:larray i64 8) (indices:larray i64 256)
  (probabilities:larray f32 256) (survival:larray i32 2048)
  (#before:erased (safe_state (I64.v token)))
  preserves gpu ** PA.pts_to row #row_perm values
  requires
    PA.pts_to counts (FStar.Ghost.reveal before).BC.counts **
    PA.pts_to indices (FStar.Ghost.reveal before).BC.indices **
    PA.pts_to probabilities (FStar.Ghost.reveal before).BC.weights **
    KA.pts_to_slice survival 0 ((I64.v token + 1) * 8)
      (FStar.Ghost.reveal before).BC.flags
  ensures
    PA.pts_to counts (source_after token (I32.v selected) values before).counts **
    PA.pts_to indices (source_after token (I32.v selected) values before).indices **
    PA.pts_to probabilities (source_after token (I32.v selected) values before).weights **
    KA.pts_to_slice survival 0 ((I64.v token + 1) * 8)
      (source_after token (I32.v selected) values before).flags **
    pure (BC.native_step Capacity.Smoke (I64.v token) (I32.v selected)
      values before (source_after token (I32.v selected) values before))
{
  let chosen = i32_index selected;
  let probability = row.(chosen);
  let selected_sum = F.add (F.zero #f32) probability;
  let weight = F.div probability selected_sum;
  let previous_count = counts.(chosen);
  let rank = I64.add previous_count 1L;
  counts.(chosen) <- rank;
  source_update_is_native token (I32.v selected) values before;
  if (I64.lte rank 32L) {
    let destination = chosen *^ 32sz +^ (i64_index rank -^ 1sz);
    indices.(destination) <- token;
    probabilities.(destination) <- weight;
    let flag_offset = i64_index token *^ 8sz +^ chosen;
    KA.slice_write survival flag_offset 1l;
  }
}

inline_for_extraction noextract
fn source_token_step
  (token:i64{0 <= I64.v token /\ I64.v token < 256})
  (row:larray f32 8) (#row_perm:perm) (#values:erased (lseq f32 8))
  (counts:larray i64 8) (indices:larray i64 256)
  (probabilities:larray f32 256) (survival:larray i32 2048)
  (#before:erased (safe_state (I64.v token)))
  preserves gpu ** PA.pts_to row #row_perm values
  requires
    PA.pts_to counts (FStar.Ghost.reveal before).BC.counts **
    PA.pts_to indices (FStar.Ghost.reveal before).BC.indices **
    PA.pts_to probabilities (FStar.Ghost.reveal before).BC.weights **
    KA.pts_to_slice survival 0 ((I64.v token + 1) * 8)
      (FStar.Ghost.reveal before).BC.flags
  ensures
    PA.pts_to counts (source_after token (BP.maximum_index values 8) values before).counts **
    PA.pts_to indices (source_after token (BP.maximum_index values 8) values before).indices **
    PA.pts_to probabilities (source_after token (BP.maximum_index values 8) values before).weights **
    KA.pts_to_slice survival 0 ((I64.v token + 1) * 8)
      (source_after token (BP.maximum_index values 8) values before).flags **
    pure (BC.native_step Capacity.Smoke (I64.v token) (BP.maximum_index values 8)
      values before (source_after token (BP.maximum_index values 8) values before))
{
  let selected = source_select row #row_perm #values;
  source_capacity_step token selected row #row_perm #values
    counts indices probabilities survival #before;
}

inline_for_extraction noextract
fn source_project_row
  (token:sz{SZ.v token < 256})
  (tokens:larray bf16 (256 * 512)) (weights:larray bf16 (8 * 512))
  (#token_perm #weight_perm:perm)
  (#xs:erased token_values) (#ws:erased weight_values)
  (row:larray f32 8) (survival:larray i32 2048)
  (#old_row:erased (lseq f32 8))
  (#old_flags:erased (lseq i32 ((SZ.v token + 1) * 8)))
  preserves gpu **
    PA.pts_to tokens #token_perm xs ** PA.pts_to weights #weight_perm ws
  requires
    pure (Shared.input_domain (token_model xs) (weight_model ws)) **
    PA.pts_to row old_row **
    KA.pts_to_slice survival 0 ((SZ.v token + 1) * 8) old_flags
  ensures exists* (scores:Classification.score_row).
    PA.pts_to row scores **
    KA.pts_to_slice survival 0 ((SZ.v token + 1) * 8)
      (zero_current_flags (SZ.v token) old_flags) **
    pure (C.seq_to_chest1 scores %~
      BP.projection_row #512 #8 (token_row xs (SZ.v token)) ws)
{
  let mut expert = 0l;
  while (I32.lt !expert 8l)
    invariant exists* (j:i32{0 <= I32.v j /\ I32.v j <= 8})
      (scores:lseq f32 8) (flags:lseq i32 ((SZ.v token + 1) * 8)).
      expert |-> j ** PA.pts_to row scores **
      KA.pts_to_slice survival 0 ((SZ.v token + 1) * 8) flags **
      pure (
        (forall (e:Shared.expert_index). e < I32.v j ==>
          B.magnitude 65536 (S.index scores e) /\
          A.v_approximates (S.index scores e)
            (BP.projection_prefix #512 #8 (token_row xs (SZ.v token)) ws e 512)) /\
        (forall (i:nat{i < (SZ.v token + 1) * 8}).
          S.index flags i ==
            (if SZ.v token * 8 <= i && i < SZ.v token * 8 + I32.v j
             then 0l else S.index old_flags i)))
    decreases (8 - I32.v !expert)
  {
    let j = !expert;
    let score = source_dot_product token (i32_index j)
      tokens weights #token_perm #weight_perm #xs #ws;
    projection_cell_matches xs ws (SZ.v token) (I32.v j);
    row.(i32_index j) <- score;
    KA.slice_write survival (token *^ 8sz +^ i32_index j) 0l;
    expert := I32.add j 1l;
  };
  with scores flags.
    assert PA.pts_to row scores **
      KA.pts_to_slice survival 0 ((SZ.v token + 1) * 8) flags;
  assert pure (S.equal flags (zero_current_flags (SZ.v token) old_flags));
  assert pure (C.seq_to_chest1 scores %~
    BP.projection_row #512 #8 (token_row xs (SZ.v token)) ws);
}

inline_for_extraction noextract
fn source_maximum (row:larray f32 8) (#scores:erased (lseq f32 8))
  preserves gpu ** PA.pts_to row scores
  returns maximum:f32
  ensures pure (maximum == S.index scores (BP.maximum_index scores 8))
{
  let first = row.(0sz);
  let mut maximum = first;
  let mut expert = 1l;
  while (I32.lt !expert 8l)
    invariant exists* (j:i32{1 <= I32.v j /\ I32.v j <= 8}) (value:f32).
      expert |-> j ** maximum |-> value **
      pure (value == S.index scores (BP.maximum_index scores (I32.v j)))
    decreases (8 - I32.v !expert)
  {
    let j = !expert;
    let candidate = row.(i32_index j);
    let previous = !maximum;
    scan_successor scores (I32.v j);
    if (F.lt previous candidate) {
      maximum := candidate;
    };
    expert := I32.add j 1l;
  };
  !maximum
}

inline_for_extraction noextract
let sum_successor (terms:lseq f32 8) (n:nat{n < 8})
  : Lemma (
      BP.sum_prefix terms (n + 1) ==
        F.add (BP.sum_prefix terms n) (S.index terms n))
  = ()

inline_for_extraction noextract
fn source_exponentiate_row
  (row:larray f32 8) (#scores:erased Classification.score_row)
  (maximum:f32{maximum == S.index scores (BP.maximum_index scores 8)})
  preserves gpu
  requires PA.pts_to row scores
  returns sum:f32
  ensures
    PA.pts_to row (BP.exponentials scores) **
    pure (sum == BP.sum_prefix (BP.exponentials scores) 8 /\
      B.nonnegative 8 sum /\ F.lte F.one sum)
{
  Bounds.zero ();
  let mut expert = 0l;
  let mut sum:f32 = F.zero #f32;
  while (I32.lt !expert 8l)
    invariant exists* (j:i32{0 <= I32.v j /\ I32.v j <= 8})
      (value:f32) (values:lseq f32 8).
      expert |-> j ** sum |-> value ** PA.pts_to row values **
      pure (
        value == BP.sum_prefix (BP.exponentials scores) (I32.v j) /\
        B.nonnegative (I32.v j) value /\
        (forall (e:Shared.expert_index).
          S.index values e ==
            (if e < I32.v j then S.index (BP.exponentials scores) e
             else S.index scores e)))
    decreases (8 - I32.v !expert)
  {
    let j = !expert;
    let score = row.(i32_index j);
    let numerator = F.fexp (F.sub score maximum);
    Classification.exponential_range scores (I32.v j);
    row.(i32_index j) <- numerator;
    let previous = !sum;
    sum_successor (BP.exponentials scores) (I32.v j);
    Classification.sum_step previous numerator (I32.v j);
    sum := F.add previous numerator;
    expert := I32.add j 1l;
  };
  with values. assert PA.pts_to row values;
  assert pure (S.equal values (BP.exponentials scores));
  Classification.denominator_range scores;
  !sum
}

inline_for_extraction noextract
fn source_softmax (row:larray f32 8) (#scores:erased Classification.score_row)
  preserves gpu
  requires PA.pts_to row scores
  ensures PA.pts_to row (produced_row scores)
{
  let maximum = source_maximum row
    #(FStar.Ghost.hide ((FStar.Ghost.reveal scores) <: lseq f32 8));
  let sum = source_exponentiate_row row #scores maximum;
  Bounds.one ();
  Bounds.div_unit F.one sum;
  let inverse = F.div (F.one #f32) sum;
  let mut expert = 0l;
  while (I32.lt !expert 8l)
    invariant exists* (j:i32{0 <= I32.v j /\ I32.v j <= 8}) (values:lseq f32 8).
      expert |-> j ** PA.pts_to row values **
      pure (forall (e:Shared.expert_index).
        S.index values e ==
          (if e < I32.v j then S.index (BP.normalized_values scores) e
           else S.index (BP.exponentials scores) e))
    decreases (8 - I32.v !expert)
  {
    let j = !expert;
    let numerator = row.(i32_index j);
    Classification.exponential_range scores (I32.v j);
    Bounds.mul_unit numerator inverse;
    row.(i32_index j) <- F.mul numerator inverse;
    expert := I32.add j 1l;
  };
  with values. assert PA.pts_to row values;
  assert pure (S.equal values (BP.normalized_values scores));
}

inline_for_extraction noextract
fn source_producer_consumer_step
  (token:i64{0 <= I64.v token /\ I64.v token < 256})
  (tokens:larray bf16 (256 * 512)) (weights:larray bf16 (8 * 512))
  (#token_perm #weight_perm:perm)
  (#xs:erased token_values) (#ws:erased weight_values)
  (row:larray f32 8)
  (counts:larray i64 8) (indices:larray i64 256)
  (probabilities:larray f32 256) (survival:larray i32 2048)
  (#old_row:erased (lseq f32 8))
  (#before:erased (safe_state (I64.v token)))
  preserves gpu **
    PA.pts_to tokens #token_perm xs ** PA.pts_to weights #weight_perm ws
  requires
    pure (Shared.input_domain (token_model xs) (weight_model ws)) **
    PA.pts_to row old_row **
    PA.pts_to counts (FStar.Ghost.reveal before).BC.counts **
    PA.pts_to indices (FStar.Ghost.reveal before).BC.indices **
    PA.pts_to probabilities (FStar.Ghost.reveal before).BC.weights **
    KA.pts_to_slice survival 0 ((I64.v token + 1) * 8)
      (FStar.Ghost.reveal before).BC.flags
  ensures exists* (scores:Classification.score_row).
    PA.pts_to row (produced_row scores) **
    PA.pts_to counts (completed_source_state token scores before).counts **
    PA.pts_to indices (completed_source_state token scores before).indices **
    PA.pts_to probabilities (completed_source_state token scores before).weights **
    KA.pts_to_slice survival 0 ((I64.v token + 1) * 8)
      (completed_source_state token scores before).flags **
    pure (
      source_row_certificate xs ws (I64.v token) scores /\
      Argmax.stable_maximum (argmax_row (produced_row scores)) 8
        (BP.maximum_index (produced_row scores) 8) /\
      BC.native_step Capacity.Smoke (I64.v token)
        (BP.maximum_index (produced_row scores) 8) (produced_row scores)
        (zeroed_state (I64.v token) before) (completed_source_state token scores before) /\
      (let chosen = BP.maximum_index (produced_row scores) 8 in
       let model = C.acc2
         (Projection.projection_softmax true (token_model xs) (weight_model ws))
         (I64.v token) chosen in
       model >. 0.0R /\
       A.v_approximates (BC.selected_weight (S.index (produced_row scores) chosen))
         (model /. (0.0R +. model))))
{
  source_project_row (i64_index token) tokens weights
    #token_perm #weight_perm #xs #ws row survival #old_row
    #(FStar.Ghost.hide ((FStar.Ghost.reveal before).BC.flags));
  with scores. assert PA.pts_to row scores;
  source_softmax row #scores;
  projection_row_matches xs ws (I64.v token);
  BP.normalization_models scores (BP.projection_row #512 #8 (token_row xs (I64.v token)) ws);
  source_selection_compatible (produced_row scores);
  source_softmax_gather xs ws (I64.v token) scores;
  source_token_step token row #1.0R
    #(FStar.Ghost.hide (produced_row scores))
    counts indices probabilities survival
    #(FStar.Ghost.hide (zeroed_state (I64.v token) before));
}

inline_for_extraction noextract
fn source_project_uninitialized_row
  (token:sz{SZ.v token < 256})
  (tokens:larray bf16 (256 * 512)) (weights:larray bf16 (8 * 512))
  (#token_perm #weight_perm:perm)
  (#xs:erased token_values) (#ws:erased weight_values)
  (row:larray f32 8) (survival:larray i32 2048)
  (#old_flags:erased (lseq i32 ((SZ.v token + 1) * 8)))
  preserves gpu **
    PA.pts_to tokens #token_perm xs ** PA.pts_to weights #weight_perm ws
  requires
    pure (Shared.input_domain (token_model xs) (weight_model ws)) **
    PA.pts_to_uninit row 8 **
    KA.pts_to_slice survival 0 ((SZ.v token + 1) * 8) old_flags
  ensures exists* (scores:Classification.score_row).
    PA.pts_to row scores **
    KA.pts_to_slice survival 0 ((SZ.v token + 1) * 8)
      (zero_current_flags (SZ.v token) old_flags) **
    pure (C.seq_to_chest1 scores %~
      BP.projection_row #512 #8 (token_row xs (SZ.v token)) ws)
{
  unfold PA.pts_to_uninit row 8;
  let mut expert = 0l;
  while (I32.lt !expert 8l)
    invariant exists* (j:i32{0 <= I32.v j /\ I32.v j <= 8})
      (raw:lseq (option f32) 8) (flags:lseq i32 ((SZ.v token + 1) * 8)).
      expert |-> j ** PA.pts_to_mask row raw (fun _ -> True) **
      KA.pts_to_slice survival 0 ((SZ.v token + 1) * 8) flags **
      pure (
        (forall (e:Shared.expert_index). e < I32.v j ==>
          Some? (S.index raw e) /\
          B.magnitude 65536 (Storage.value (S.index raw e)) /\
          A.v_approximates (Storage.value (S.index raw e))
            (BP.projection_prefix #512 #8 (token_row xs (SZ.v token)) ws e 512)) /\
        (forall (i:nat{i < (SZ.v token + 1) * 8}).
          S.index flags i ==
            (if SZ.v token * 8 <= i && i < SZ.v token * 8 + I32.v j
             then 0l else S.index old_flags i)))
    decreases (8 - I32.v !expert)
  {
    let j = !expert;
    let score = source_dot_product token (i32_index j)
      tokens weights #token_perm #weight_perm #xs #ws;
    projection_cell_matches xs ws (SZ.v token) (I32.v j);
    PA.mask_write row (i32_index j) score;
    KA.slice_write survival (token *^ 8sz +^ i32_index j) 0l;
    expert := I32.add j 1l;
  };
  PA.from_mask row;
  with scores flags.
    assert PA.pts_to row scores **
      KA.pts_to_slice survival 0 ((SZ.v token + 1) * 8) flags;
  assert pure (S.equal flags (zero_current_flags (SZ.v token) old_flags));
  assert pure (S.length scores == 8);
  assert pure (C.seq_to_chest1 (scores <: lseq f32 8) %~
    BP.projection_row #512 #8 (token_row xs (SZ.v token)) ws);
}

inline_for_extraction noextract
let source_history_certificate xs ws t scores =
  projection_row_matches xs ws t;
  BP.normalization_models scores (BP.projection_row #512 #8 (token_row xs t) ws);
  source_selection_compatible (produced_row scores)

inline_for_extraction noextract
let zero_extended_state_matches n st current =
  assert (S.equal
    (zero_current_flags n (S.append st.BC.flags current))
    (S.append st.BC.flags (S.create 8 0l)))

inline_for_extraction noextract
fn source_recorded_token
  (token:i64{0 <= I64.v token /\ I64.v token < 256})
  (tokens:larray bf16 (256 * 512)) (weights:larray bf16 (8 * 512))
  (#token_perm #weight_perm:perm)
  (#xs:erased token_values) (#ws:erased weight_values)
  (counts:larray i64 8) (indices:larray i64 256)
  (probabilities:larray f32 256) (survival:larray i32 2048)
  (#h:erased (History.history (I64.v token)))
  (#st:erased (history_state (I64.v token)))
  (#current:erased (lseq i32 8))
  preserves gpu **
    PA.pts_to tokens #token_perm xs ** PA.pts_to weights #weight_perm ws
  requires
    pure (Shared.input_domain (token_model xs) (weight_model ws) /\
      History.valid (token_model xs) (weight_model ws) h /\ History.prefix_ok h st) **
    PA.pts_to counts (FStar.Ghost.reveal st).BC.counts **
    PA.pts_to indices (FStar.Ghost.reveal st).BC.indices **
    PA.pts_to probabilities (FStar.Ghost.reveal st).BC.weights **
    KA.pts_to_slice survival 0 ((I64.v token + 1) * 8)
      (S.append (FStar.Ghost.reveal st).BC.flags current)
  returns scores:erased Classification.score_row
  ensures
    PA.pts_to counts
      (completed_source_state token scores (current_state (I64.v token) st current)).counts **
    PA.pts_to indices
      (completed_source_state token scores (current_state (I64.v token) st current)).indices **
    PA.pts_to probabilities
      (completed_source_state token scores (current_state (I64.v token) st current)).weights **
    KA.pts_to_slice survival 0 ((I64.v token + 1) * 8)
      (completed_source_state token scores (current_state (I64.v token) st current)).flags **
    pure (
      History.valid (token_model xs) (weight_model ws) (History.push h scores) /\
      History.prefix_ok (History.push h scores)
        (completed_source_state token scores (current_state (I64.v token) st current)))
{
  zero_extended_state_matches (I64.v token) st current;
  let before = FStar.Ghost.hide (current_state (I64.v token) st current);
  let mut row : PA.array f32 = [| 8sz |];
  source_project_uninitialized_row (i64_index token) tokens weights
    #token_perm #weight_perm #xs #ws row survival
    #(FStar.Ghost.hide (S.append (FStar.Ghost.reveal st).BC.flags current));
  with scores. assert PA.pts_to row scores;
  source_softmax row #scores;
  source_history_certificate xs ws (I64.v token) scores;
  source_token_step token row #1.0R #(FStar.Ghost.hide (produced_row scores))
    counts indices probabilities survival
    #(FStar.Ghost.hide (zeroed_state (I64.v token) before));
  History.valid_push (token_model xs) (weight_model ws) h scores;
  History.advance h scores st (completed_source_state token scores before);
  PA.to_mask row;
  FStar.Ghost.hide (scores <: Classification.score_row)
}

inline_for_extraction noextract
ghost
fn pack_loop_state
  (n:History.prefix) (xs:token_values) (ws:weight_values)
  (counts:larray i64 8) (indices:larray i64 256)
  (probabilities:larray f32 256) (survival:larray i32 2048)
  (h:History.history n) (st:history_state n) (tail:lseq i32 (2048 - n * 8))
  requires
    PA.pts_to counts st.BC.counts ** PA.pts_to indices st.BC.indices **
    PA.pts_to probabilities st.BC.weights **
    KA.pts_to_slice survival 0 (n * 8) st.BC.flags **
    KA.pts_to_slice survival (n * 8) 2048 tail **
    pure (History.valid (token_model xs) (weight_model ws) h /\ History.prefix_ok h st)
  ensures loop_state n xs ws counts indices probabilities survival
{
  fold loop_payload n xs ws counts indices probabilities survival h st tail;
  Pulse.Lib.Core.intro_exists
    (fun tail -> loop_payload n xs ws counts indices probabilities survival h st tail) tail;
  Pulse.Lib.Core.intro_exists
    (fun st -> exists* tail.
      loop_payload n xs ws counts indices probabilities survival h st tail) st;
  Pulse.Lib.Core.intro_exists
    (fun h -> exists* st tail.
      loop_payload n xs ws counts indices probabilities survival h st tail) h;
  fold loop_state n xs ws counts indices probabilities survival;
}

inline_for_extraction noextract
fn source_write_demand
  (counts:larray i64 8) (demand:larray i32 8)
  (#values:erased final_counts) (#before:erased (lseq i32 8))
  preserves gpu ** PA.pts_to counts values
  requires PA.pts_to demand before
  returns result:erased (lseq i32 8)
  ensures PA.pts_to demand result **
    pure (forall (e:Shared.expert_index).
      I32.v (S.index result e) == I64.v (S.index values e))
{
  let mut expert = 0l;
  while (I32.lt !expert 8l)
    invariant exists* (j:i32{0 <= I32.v j /\ I32.v j <= 8}) (ds:lseq i32 8).
      expert |-> j ** PA.pts_to demand ds **
      pure (forall (e:Shared.expert_index). e < I32.v j ==>
        I32.v (S.index ds e) == I64.v (S.index values e))
    decreases (8 - I32.v !expert)
  {
    let j = !expert;
    let count = counts.(i32_index j);
    demand.(i32_index j) <- demand_cast count;
    expert := I32.add j 1l;
  };
  with ds. assert PA.pts_to demand ds;
  FStar.Ghost.hide (ds <: lseq i32 8)
}

inline_for_extraction noextract
fn source_router
  (tokens:larray bf16 (256 * 512)) (weights:larray bf16 (8 * 512))
  (#token_perm #weight_perm:perm)
  (#xs:erased token_values) (#ws:erased weight_values)
  (demand:larray i32 8) (indices:larray i64 256)
  (probabilities:larray f32 256) (survival:larray i32 2048)
  (#old_demand:erased (lseq i32 8)) (#old_indices:erased (lseq i64 256))
  (#old_probabilities:erased (lseq f32 256)) (#old_survival:erased (lseq i32 2048))
  preserves gpu **
    PA.pts_to tokens #token_perm xs ** PA.pts_to weights #weight_perm ws
  requires
    pure (Shared.input_domain (token_model xs) (weight_model ws)) **
    PA.pts_to demand old_demand ** PA.pts_to indices old_indices **
    PA.pts_to probabilities old_probabilities ** PA.pts_to survival old_survival
  ensures exists* (ds:lseq i32 8) (st:BC.physical_state Capacity.Smoke 256).
    PA.pts_to demand ds ** PA.pts_to indices st.indices **
    PA.pts_to probabilities st.weights ** PA.pts_to survival st.flags **
    pure (Shared.task_relation (token_model xs) (weight_model ws)
      (token_model xs) (weight_model ws) (History.output ds st))
{
  let mut counts : PA.array i64 = [| 0L; 8sz |];
  Storage.initialize_slots indices probabilities #old_indices #old_probabilities;
  History.initial_invariant ();
  KA.array_to_slice survival;
  S.append_empty_l (FStar.Ghost.reveal old_survival);
  KA.slice_split survival #1.0R
    #(FStar.Ghost.hide (S.empty <: S.seq i32))
    #(FStar.Ghost.hide ((FStar.Ghost.reveal old_survival) <: S.seq i32)) 0 0 2048;
  pack_loop_state 0 xs ws counts indices probabilities survival
    (S.empty <: History.history 0) History.initial old_survival;
  let mut token = 0l;
  while (I32.lt !token 256l)
    invariant exists* (t:i32{0 <= I32.v t /\ I32.v t <= 256}).
      token |-> t ** loop_state (I32.v t) xs ws counts indices probabilities survival
    decreases (256 - I32.v !token)
  {
    let t = !token;
    unfold loop_state (I32.v t) xs ws counts indices probabilities survival;
    with h st tail.
      assert loop_payload (I32.v t) xs ws counts indices probabilities survival h st tail;
    unfold loop_payload (I32.v t) xs ws counts indices probabilities survival h st tail;
    let current:erased (lseq i32 8) = FStar.Ghost.hide (S.slice tail 0 8);
    let remaining:erased (lseq i32 (2048 - (I32.v t + 1) * 8)) =
      FStar.Ghost.hide (S.slice tail 8 (2048 - I32.v t * 8));
    assert pure (S.equal tail (S.append current remaining));
    KA.slice_split survival #1.0R
      #(FStar.Ghost.hide ((FStar.Ghost.reveal current) <: S.seq i32))
      #(FStar.Ghost.hide ((FStar.Ghost.reveal remaining) <: S.seq i32))
      (I32.v t * 8) ((I32.v t + 1) * 8) 2048;
    KA.slice_concat survival #1.0R
      #(FStar.Ghost.hide (st.BC.flags <: S.seq i32))
      #(FStar.Ghost.hide ((FStar.Ghost.reveal current) <: S.seq i32))
      0 (I32.v t * 8) ((I32.v t + 1) * 8);
    let native_token = IC.int32_to_int64 t;
    let scores = source_recorded_token native_token tokens weights
      #token_perm #weight_perm #xs #ws counts indices probabilities survival
      #(FStar.Ghost.hide h) #(FStar.Ghost.hide st) #current;
    let after = FStar.Ghost.hide
      (completed_source_state native_token scores (current_state (I32.v t) st current));
    let next = I32.add t 1l;
    assert pure (I32.v next == I32.v t + 1);
    pack_loop_state (I32.v t + 1) xs ws counts indices probabilities survival
      (History.push h scores) after remaining;
    rewrite loop_state (I32.v t + 1) xs ws counts indices probabilities survival
      as loop_state (I32.v next) xs ws counts indices probabilities survival;
    token := next;
  };
  with finished. assert token |-> finished;
  assert pure (I32.v finished == 256);
  rewrite loop_state (I32.v finished) xs ws counts indices probabilities survival
    as loop_state 256 xs ws counts indices probabilities survival;
  unfold loop_state 256 xs ws counts indices probabilities survival;
  with h st tail.
    assert loop_payload 256 xs ws counts indices probabilities survival h st tail;
  unfold loop_payload 256 xs ws counts indices probabilities survival h st tail;
  KA.slice_concat survival #1.0R #(FStar.Ghost.hide (st.BC.flags <: S.seq i32))
    #(FStar.Ghost.hide (tail <: S.seq i32)) 0 2048 2048;
  assert pure (S.equal (S.append st.BC.flags tail) st.BC.flags);
  KA.slice_to_array survival;
  let ds = source_write_demand counts demand
    #(FStar.Ghost.hide (st.BC.counts <: final_counts)) #old_demand;
  History.final_refines (token_model xs) (weight_model ws) h st ds;
  ()
}

inline_for_extraction noextract
let native_models_are_exact xs ws =
  assert_norm (token_model xs == SourceNative.token_model xs);
  assert_norm (weight_model ws == SourceNative.weight_model ws)

inline_for_extraction noextract
fn source_gpu_body
  (#xs:erased token_values) (#ws:erased weight_values) (#fp #fw:perm)
  (tokens:SourceNative.garray bf16 131072) (weights:SourceNative.garray bf16 4096)
  (demand:SourceNative.garray i32 8) (indices:SourceNative.garray i64 256)
  (probabilities:SourceNative.garray f32 256) (survival:SourceNative.garray i32 2048)
  ()
  preserves gpu
  requires SourceNative.device_pre xs ws fp fw
    tokens weights demand indices probabilities survival
  ensures SourceNative.device_post xs ws fp fw
    tokens weights demand indices probabilities survival
{
  native_models_are_exact xs ws;
  unfold SourceNative.device_pre xs ws fp fw tokens weights demand indices probabilities survival;
  with ds is ps ss.
    assert PA.pts_to demand ds ** PA.pts_to indices is **
      PA.pts_to probabilities ps ** PA.pts_to survival ss;
  source_router tokens weights #fp #fw #xs #ws demand indices probabilities survival
    #(FStar.Ghost.hide ds) #(FStar.Ghost.hide is) #(FStar.Ghost.hide ps) #(FStar.Ghost.hide ss);
  fold SourceNative.device_post xs ws fp fw tokens weights demand indices probabilities survival;
}

inline_for_extraction noextract
fn source_execution_prefix
  (tokens:SourceNative.token_argument) (weights:SourceNative.weight_argument)
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (#fp #fw:perm) (#xs:erased token_values) (#ws:erased weight_values)
  (frame:slprop) {| is_send frame |} ()
  requires
    Native.call_resources stream epoch tokens weights fp fw xs ws frame **
    pure (Shared.input_domain (token_model xs) (weight_model ws))
  returns result:Caller.allocation
  ensures
    cpu ** Input.reference tokens ** Input.reference weights **
    SourceNative.after_call stream epoch tokens weights xs ws fp fw frame result
{
  native_models_are_exact xs ws;
  unfold Native.call_resources stream epoch tokens weights fp fw xs ws frame;
  fold SourceNative.before_call stream epoch tokens weights xs ws fp fw frame;
  SourceNative.source_parameter_scopes stream #epoch tokens weights #xs #ws #fp #fw
    (source_gpu_body #xs #ws #fp #fw) frame ()
}

inline_for_extraction noextract
fn source_return_parameter_body
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (tokens:SourceNative.token_argument) (weights:SourceNative.weight_argument)
  (#xs:erased token_values) (#ws:erased weight_values) (#fp #fw:perm)
  (frame:slprop) {| is_send frame |} ()
  requires cpu ** Input.reference weights ** Input.reference tokens **
    SourceNative.before_call stream epoch tokens weights xs ws fp fw frame
  returns result:source_result
  ensures cpu ** Input.reference weights ** Input.reference tokens **
    source_result_post stream epoch tokens weights xs ws fp fw frame result
{
  let allocation = SourceNative.source_parameter_body stream #epoch
    tokens weights #xs #ws #fp #fw (source_gpu_body #xs #ws #fp #fw) frame ();
  unfold SourceNative.after_call stream epoch tokens weights xs ws fp fw frame allocation;
  match allocation {
    Caller.Failed error -> {
      fold source_result_post stream epoch tokens weights xs ws fp fw frame
        (Allocation_failed error);
      Allocation_failed error
    }
    Caller.Allocated outputs -> {
      let result = SourceReturn.finish_source_return stream #(Epoch.epoch_next epoch)
        tokens weights #xs #ws #fp #fw outputs frame ();
      fold source_result_post stream epoch tokens weights xs ws fp fw frame
        (Native_return result);
      Native_return result
    }
  }
}

inline_for_extraction noextract
fn source_return_weight_parameter
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (tokens:SourceNative.token_argument) (weights:SourceNative.weight_argument)
  (#xs:erased token_values) (#ws:erased weight_values) (#fp #fw:perm)
  (frame:slprop) {| is_send frame |} ()
  requires cpu ** Input.reference tokens **
    SourceNative.before_call stream epoch tokens weights xs ws fp fw frame
  returns result:source_result
  ensures cpu ** Input.reference tokens **
    source_result_post stream epoch tokens weights xs ws fp fw frame result
{
  Input.with_parameter weights
    (Input.reference tokens **
      SourceNative.before_call stream epoch tokens weights xs ws fp fw frame)
    (fun result -> Input.reference tokens **
      source_result_post stream epoch tokens weights xs ws fp fw frame result)
    (source_return_parameter_body stream #epoch tokens weights #xs #ws #fp #fw frame)
}

inline_for_extraction noextract
fn source_return_compatibility
  (tokens:SourceNative.token_argument) (weights:SourceNative.weight_argument)
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (#fp #fw:perm) (#xs:erased token_values) (#ws:erased weight_values)
  (frame:slprop) {| is_send frame |} ()
  requires
    Native.call_resources stream epoch tokens weights fp fw xs ws frame **
    pure (Shared.input_domain (token_model xs) (weight_model ws))
  returns result:source_result
  ensures
    cpu ** Input.reference tokens ** Input.reference weights **
    source_result_post stream epoch tokens weights xs ws fp fw frame result
{
  native_models_are_exact xs ws;
  unfold Native.call_resources stream epoch tokens weights fp fw xs ws frame;
  fold SourceNative.before_call stream epoch tokens weights xs ws fp fw frame;
  Input.with_parameter tokens
    (SourceNative.before_call stream epoch tokens weights xs ws fp fw frame)
    (source_result_post stream epoch tokens weights xs ws fp fw frame)
    (source_return_weight_parameter stream #epoch tokens weights #xs #ws #fp #fw frame)
}

inline_for_extraction noextract
let source_post_is_program_post
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (tokens:SourceNative.token_argument) (weights:SourceNative.weight_argument)
  (xs:token_values) (ws:weight_values) (fp fw:perm)
  (x:Shared.token_matrix) (w:Shared.weight_matrix) (frame:slprop)
  (result:source_result)
  : Lemma
      (requires token_model xs == x /\ weight_model ws == w)
      (ensures
        (cpu ** Input.reference tokens ** Input.reference weights **
          source_result_post stream epoch tokens weights xs ws fp fw frame result) ==
        program_post stream epoch tokens weights xs ws fp fw x w frame result)
  =
  native_models_are_exact xs ws;
  match result with
  | Allocation_failed _ -> ()
  | Native_return outcome ->
    match outcome with
    | Results.Failed _ -> ()
    | Results.Constructed value ->
      assert_norm (SourceNative.output_post (SourceReturn.handles value) xs ws ==
        program_output (SourceNative.token_model xs) (SourceNative.weight_model ws) value);
      assert (SourceNative.output_post (SourceReturn.handles value) xs ws ==
        program_output x w value)

inline_for_extraction noextract
fn program_refines
  (tokens weights:Input.argument bf16)
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (#fp #fw:perm) (#xs:erased token_values) (#ws:erased weight_values)
  (#x:erased Shared.token_matrix) (#w:erased Shared.weight_matrix)
  (frame:slprop) {| is_send frame |} ()
  requires
    Native.call_resources stream epoch tokens weights fp fw xs ws frame **
    pure (Native.input_binding tokens weights xs ws x w /\ Shared.input_domain x w)
  returns result:source_result
  ensures program_post stream epoch tokens weights xs ws fp fw x w frame result
{
  input_models_are_bound tokens weights xs ws x w;
  let result = source_return_compatibility tokens weights stream #epoch #fp #fw #xs #ws frame ();
  source_post_is_program_post stream epoch tokens weights xs ws fp fw x w frame result;
  rewrite (cpu ** Input.reference tokens ** Input.reference weights **
    source_result_post stream epoch tokens weights xs ws fp fw frame result)
    as program_post stream epoch tokens weights xs ws fp fw x w frame result;
  result
}
