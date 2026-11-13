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
module Cast = Kuiper.Float.Casts
module IC = FStar.Int.Cast
module I32 = FStar.Int32
module I64 = FStar.Int64
module SZ = Kuiper.SizeT
module PA = Pulse.Lib.Array
module KA = Kuiper.Array
module Input = Kuiper.Tensor.Input
module RT = Kuiper.Tensor.Runtime
module Results = Kuiper.Tensor.Results
module Pledge = Pulse.Lib.Pledge

inline_for_extraction noextract
let token_values = lseq bf16 (256 * 512)

inline_for_extraction noextract
let weight_values = lseq bf16 (8 * 512)

inline_for_extraction noextract
let token_model (values:token_values) : Shared.token_matrix =
  C.mk2 (fun t h -> S.index values (t * 512 + h))

inline_for_extraction noextract
let weight_model (values:weight_values) : Shared.weight_matrix =
  C.mk2 (fun h e -> S.index values (e * 512 + h))

inline_for_extraction noextract
let token_row (values:token_values) (t:Shared.token_index) : lseq bf16 512 =
  S.slice values (t * 512) (t * 512 + 512)

inline_for_extraction noextract
let signed_output_dtype : RT.dtype I32.t = RT.Int32

inline_for_extraction noextract
val input_models_are_bound
  (tokens weights:Input.argument bf16)
  (xs:token_values) (ws:weight_values)
  (x:Shared.token_matrix) (w:Shared.weight_matrix)
  : Lemma
      (requires Native.input_binding tokens weights xs ws x w)
      (ensures
        token_model xs == x /\ weight_model ws == w /\
        PA.length (Input.data tokens) == 256 * 512 /\
        PA.length (Input.data weights) == 8 * 512 /\
        PA.offset_of (Input.data tokens) ==
          PA.offset_of (Input.backing tokens) + Input.offset tokens /\
        PA.offset_of (Input.data weights) ==
          PA.offset_of (Input.backing weights) + Input.offset weights)

inline_for_extraction noextract
val source_row_admissible
  (xs:token_values) (ws:weight_values) (t:Shared.token_index)
  : Lemma
      (requires Shared.input_domain (token_model xs) (weight_model ws))
      (ensures BP.row_domain #512 #8 (token_row xs t) ws)

inline_for_extraction noextract
val projection_row_matches
  (xs:token_values) (ws:weight_values) (t:Shared.token_index)
  : Lemma (
      BP.projection_row #512 #8 (token_row xs t) ws ==
        C.chest2_row
          (Projection.projection true (token_model xs) (weight_model ws)) t)

(* A local producer invariant, not a native caller premise: the supplied
   scores are the cells at base_s.cu:31, modeled from the same BF16 inputs. *)
inline_for_extraction noextract
val source_softmax_gather
  (xs:token_values) (ws:weight_values) (t:Shared.token_index)
  (scores:lseq f32 8)
  : Lemma
      (requires C.seq_to_chest1 scores %~ BP.projection_row #512 #8 (token_row xs t) ws)
      (ensures (
        let row = BP.normalized_values scores in
        let chosen = BP.maximum_index row 8 in
        let model = C.acc2
          (Projection.projection_softmax true (token_model xs) (weight_model ws))
          t chosen in
        model >. 0.0R /\
        A.v_approximates (S.index row chosen) model /\
        A.v_approximates (BC.selected_weight (S.index row chosen))
          (model /. (0.0R +. model))))

(* Exactly the accepted argmax row precondition, discharged by source_softmax.
   source_select also checks the native recurrence without this refinement. *)
inline_for_extraction noextract
let classified_row =
  row:lseq f32 8{
    forall (e:Shared.expert_index).
      F.kind (S.index row e) == F.Finite /\
      F.lte (F.zero #f32) (S.index row e) /\
      F.lte (S.index row e) (F.one #f32)}

inline_for_extraction noextract
let argmax_row (row:classified_row) : Argmax.row 8 =
  fun e -> S.index row e

inline_for_extraction noextract
let capacity_row (row:classified_row) : Capacity.native_row 8 =
  fun e -> S.index row e

inline_for_extraction noextract
val source_selection_compatible (row:classified_row)
  : Lemma (
      BP.maximum_index row 8 == Argmax.top1_index (argmax_row row) /\
      BP.maximum_index row 8 == Capacity.reference_selection (capacity_row row) /\
      Argmax.stable_maximum (argmax_row row) 8 (BP.maximum_index row 8))

inline_for_extraction noextract
let counts_bounded (t:Shared.token_index)
  (before:BC.physical_state Capacity.Smoke (t + 1)) : prop =
  forall (e:Shared.expert_index).
    0 <= I64.v (S.index before.counts e) /\
    I64.v (S.index before.counts e) <= t

inline_for_extraction noextract
let safe_state (t:Shared.token_index) =
  before:BC.physical_state Capacity.Smoke (t + 1){counts_bounded t before}

inline_for_extraction noextract
let source_after
  (token:i64{0 <= I64.v token /\ I64.v token < 256})
  (chosen:Shared.expert_index) (row:lseq f32 8)
  (before:safe_state (I64.v token))
  : BC.physical_state Capacity.Smoke (I64.v token + 1) =
  let rank = I64.add (S.index before.counts chosen) 1L in
  BC.updated_state Capacity.Smoke token chosen
    (BC.selected_weight (S.index row chosen)) before rank

inline_for_extraction noextract
val source_update_is_native
  (token:i64{0 <= I64.v token /\ I64.v token < 256})
  (chosen:Shared.expert_index) (row:lseq f32 8)
  (before:safe_state (I64.v token))
  : Lemma (
      BC.native_step Capacity.Smoke (I64.v token) chosen row before
        (source_after token chosen row before))

(* This is the induction-step adapter, not an assumed family receipt.
   The implementation below proves native_step for the actual in-place stores. *)
inline_for_extraction noextract
val source_update_refines_prefix
  (b:Capacity.selected_boundary Capacity.Smoke)
  (token:i64{0 <= I64.v token /\ I64.v token < 256})
  (selected:lseq i32 1) (row:lseq f32 8)
  (before:BC.physical_state Capacity.Smoke (I64.v token + 1))
  : Lemma
      (requires BC.input_representation Capacity.Smoke b (I64.v token)
        selected row before)
      (ensures
        counts_bounded (I64.v token) before /\
        BC.state_representation Capacity.Smoke (I64.v token + 1)
          (source_after token (b.selected_expert (I64.v token)) row before)
          (Capacity.prefix_state b (I64.v token + 1)))

inline_for_extraction noextract
let first_live_state : safe_state 0 = {
  BC.counts = S.create 8 0L;
  BC.indices = S.create 256 (-1L);
  BC.weights = S.create 256 (F.zero #f32);
  BC.flags = S.create 8 0l
}

inline_for_extraction noextract
val first_token_transition
  (row:lseq f32 8) (e:Shared.expert_index) (c:Shared.slot_index)
  : Lemma (
      let chosen = BP.maximum_index row 8 in
      let after = source_after 0L chosen row first_live_state in
      I64.v (S.index after.counts e) == (if e = chosen then 1 else 0) /\
      I64.v (S.index after.indices (e * 32 + c)) ==
        (if e = chosen && c = 0 then 0 else -1) /\
      S.index after.weights (e * 32 + c) ==
        (if e = chosen && c = 0
         then BC.selected_weight (S.index row chosen) else F.zero #f32) /\
      I32.v (S.index after.flags e) == (if e = chosen then 1 else 0))

inline_for_extraction noextract
let i32_index (value:i32{0 <= I32.v value})
  : result:sz{SZ.v result == I32.v value} =
  SZ.uint32_to_sizet (IC.int32_to_uint32 value)

inline_for_extraction noextract
let i64_index (value:i64{0 <= I64.v value /\ I64.v value < 4294967296})
  : result:sz{SZ.v result == I64.v value} =
  SZ.uint32_to_sizet (IC.int64_to_uint32 value)

inline_for_extraction noextract
let demand_cast (value:i64{0 <= I64.v value /\ I64.v value <= 256})
  : result:i32{I32.v result == I64.v value} =
  IC.int64_to_int32 value

(* Proof slices of the retained source, not replacement launches/wrappers. *)
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

inline_for_extraction noextract
fn source_select
  (row:larray f32 8) (#row_perm:perm) (#values:erased (lseq f32 8))
  preserves gpu ** PA.pts_to row #row_perm values
  returns selected:i32
  ensures pure (
    0 <= I32.v selected /\ I32.v selected < 8 /\
    I32.v selected == BP.maximum_index values 8)

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

inline_for_extraction noextract
let zero_current_flags (t:Shared.token_index) (flags:lseq i32 ((t + 1) * 8))
  : lseq i32 ((t + 1) * 8) =
  S.init ((t + 1) * 8) (fun i -> if i < t * 8 then S.index flags i else 0l)

inline_for_extraction noextract
let zeroed_state (t:Shared.token_index) (before:safe_state t) : safe_state t = {
  BC.counts = before.counts;
  BC.indices = before.indices;
  BC.weights = before.weights;
  BC.flags = zero_current_flags t before.flags
}

inline_for_extraction noextract
let produced_row (scores:Classification.score_row)
  : row:classified_row{row == BP.normalized_values scores} =
  Classification.normalized_range scores;
  BP.normalized_values scores

inline_for_extraction noextract
let completed_source_state
  (token:i64{0 <= I64.v token /\ I64.v token < 256})
  (scores:Classification.score_row) (before:safe_state (I64.v token))
  : BC.physical_state Capacity.Smoke (I64.v token + 1) =
  let row = produced_row scores in
  source_after token (BP.maximum_index row 8) row (zeroed_state (I64.v token) before)

inline_for_extraction noextract
let source_row_certificate
  (xs:token_values) (ws:weight_values) (t:Shared.token_index)
  (scores:Classification.score_row) : prop =
  C.seq_to_chest1 scores %~ BP.projection_row #512 #8 (token_row xs t) ws /\
  C.seq_to_chest1 (produced_row scores) %~
    C.chest2_row
      (Projection.projection_softmax true (token_model xs) (weight_model ws)) t

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

inline_for_extraction noextract
fn source_maximum (row:larray f32 8) (#scores:erased (lseq f32 8))
  preserves gpu ** PA.pts_to row scores
  returns maximum:f32
  ensures pure (maximum == S.index scores (BP.maximum_index scores 8))

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

inline_for_extraction noextract
fn source_softmax (row:larray f32 8) (#scores:erased Classification.score_row)
  preserves gpu
  requires PA.pts_to row scores
  ensures PA.pts_to row (produced_row scores)

(* One token's actual in-place source schedule, not the synchronous family ABI. *)
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

inline_for_extraction noextract
val source_history_certificate
  (xs:token_values) (ws:weight_values) (t:Shared.token_index)
  (scores:Classification.score_row)
  : Lemma
      (requires C.seq_to_chest1 scores %~ BP.projection_row #512 #8 (token_row xs t) ws)
      (ensures History.certificate (token_model xs) (weight_model ws) t scores)

inline_for_extraction noextract
val zero_extended_state_matches
  (n:Shared.token_index) (st:BC.physical_state Capacity.Smoke n) (current:lseq i32 8)
  : Lemma
      (requires History.bounded_counts st)
      (ensures
        counts_bounded n (History.extend st current) /\
        zeroed_state n (History.extend st current) == History.zero_extend st)

inline_for_extraction noextract
let history_state (n:History.prefix) =
  st:BC.physical_state Capacity.Smoke n{History.bounded_counts st}

inline_for_extraction noextract
let current_state (n:Shared.token_index) (st:history_state n) (current:lseq i32 8)
  : safe_state n =
  zero_extended_state_matches n st current;
  History.extend st current

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

inline_for_extraction noextract
let loop_payload (n:History.prefix) (xs:token_values) (ws:weight_values)
  (counts:larray i64 8) (indices:larray i64 256)
  (probabilities:larray f32 256) (survival:larray i32 2048)
  (h:History.history n) (st:history_state n) (tail:lseq i32 (2048 - n * 8)) : slprop =
  PA.pts_to counts st.BC.counts **
  PA.pts_to indices st.BC.indices **
  PA.pts_to probabilities st.BC.weights **
  KA.pts_to_slice survival 0 (n * 8) st.BC.flags **
  KA.pts_to_slice survival (n * 8) 2048 tail **
  pure (History.valid (token_model xs) (weight_model ws) h /\ History.prefix_ok h st)

inline_for_extraction noextract
let loop_state (n:History.prefix) (xs:token_values) (ws:weight_values)
  (counts:larray i64 8) (indices:larray i64 256)
  (probabilities:larray f32 256) (survival:larray i32 2048) : slprop =
  exists* (h:History.history n) (st:history_state n) (tail:lseq i32 (2048 - n * 8)).
    loop_payload n xs ws counts indices probabilities survival h st tail

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

inline_for_extraction noextract
let final_counts = values:lseq i64 8{
  forall (e:Shared.expert_index).
    0 <= I64.v (S.index values e) /\ I64.v (S.index values e) <= 256}

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

inline_for_extraction noextract
val native_models_are_exact (xs:token_values) (ws:weight_values)
  : Lemma (
      token_model xs == SourceNative.token_model xs /\
      weight_model ws == SourceNative.weight_model ws)

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

(* Fixed GPU/runtime composition. Native std::vector result construction is
   a separate source boundary, so this is not the complete program theorem. *)
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

inline_for_extraction noextract noeq
type source_result =
  | Allocation_failed : error:RT.allocation_error -> source_result
  | Native_return : result:SourceReturn.outcome -> source_result

inline_for_extraction noextract
let source_result_post
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (tokens:SourceNative.token_argument) (weights:SourceNative.weight_argument)
  (xs:token_values) (ws:weight_values) (fp fw:perm) (frame:slprop)
  (result:source_result) : slprop =
  Stream.stream_live stream ** RT.runtime stream **
  (match result with
   | Allocation_failed _ ->
     Epoch.epoch_live stream epoch **
     Pulse.Lib.Pledge.pledge0 (Epoch.epoch_done stream epoch)
       (Native.input_cells tokens weights fp fw xs ws ** frame)
   | Native_return outcome ->
     Epoch.epoch_live stream (Epoch.epoch_next epoch) **
     SourceReturn.after_return stream (Epoch.epoch_next epoch)
       tokens weights xs ws fp fw frame outcome)

(* Compatibility checkpoint, not the independently accepted program_refines.
   The fixed source body supplies the return premises inside both parameters. *)
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

inline_for_extraction noextract
let program_output
  (x:Shared.token_matrix) (w:Shared.weight_matrix) (value:SourceReturn.vector)
  : slprop =
  exists* (values:SourceNative.result_values).
    Native.output_cells (SourceReturn.handles value) values **
    pure (
      Native.output_binding (SourceReturn.handles value) values
        (SourceNative.public_view values) /\
      Shared.task_relation x w x w (SourceNative.public_view values))

inline_for_extraction noextract
let program_post
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (tokens weights:Input.argument bf16)
  (xs:token_values) (ws:weight_values) (fp fw:perm)
  (x:Shared.token_matrix) (w:Shared.weight_matrix) (frame:slprop)
  (result:source_result) : slprop =
  cpu ** Input.reference tokens ** Input.reference weights **
  Stream.stream_live stream ** RT.runtime stream **
  (match result with
   | Allocation_failed _ ->
     Epoch.epoch_live stream epoch **
     Pledge.pledge0 (Epoch.epoch_done stream epoch)
       (Native.input_cells tokens weights fp fw xs ws ** frame)
   | Native_return outcome ->
     Epoch.epoch_live stream (Epoch.epoch_next epoch) **
     (match outcome with
      | Results.Failed _ ->
        Pledge.pledge0 (Epoch.epoch_done stream (Epoch.epoch_next epoch))
          (Native.input_cells tokens weights fp fw xs ws ** frame)
      | Results.Constructed value ->
        Native.output_owners stream (SourceReturn.handles value) **
        pure (Caller.metadata (SourceReturn.handles value)) **
        Pledge.pledge0 (Epoch.epoch_done stream (Epoch.epoch_next epoch))
          ((Native.input_cells tokens weights fp fw xs ws **
            program_output x w value) ** frame)))

(* Complete retained source, including both allocation and return failures.
   The only logical input adapter is the frozen actual-cell binding. *)
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
