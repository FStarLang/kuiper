module KPlay.SharedSpec

open FStar.Real

module C = Kuiper.Chest
module F = Kuiper.Floating
module F32 = Kuiper.Float32
module BF16 = Kuiper.BFloat16
module A = Kuiper.Approximates.Base
module Cast = Kuiper.Float.Casts
module Classical = FStar.Classical
module Projection = KPlay.Proof.Shared.ProjectionSoftmax
module Argmax = KPlay.Proof.Shared.Argmax
module Capacity = KPlay.Proof.Shared.Capacity
module Native = KPlay.Proof.Shared.Native

inline_for_extraction noextract
let token_matrix = Projection.token_matrix true

inline_for_extraction noextract
let weight_matrix = Projection.weight_matrix true

inline_for_extraction noextract
let token_index = Capacity.token_index Capacity.Smoke

inline_for_extraction noextract
let expert_index = Capacity.expert_index Capacity.Smoke

inline_for_extraction noextract
let slot_index = Capacity.slot_index Capacity.Smoke

inline_for_extraction noextract
let public_outputs = Capacity.public_outputs Capacity.Smoke

inline_for_extraction noextract
let input_domain (tokens:token_matrix) (weights:weight_matrix) : prop =
  Projection.input_domain true tokens weights

(* These are observations of one execution, not caller inputs. Each role
   must bind them to its actual produced cells before using this relation. *)
inline_for_extraction noextract noeq
type producer_observation = {
  projected_scores: token_index -> expert_index -> F32.t;
  softmax_inputs: token_index -> expert_index -> F32.t;
  probability_rows: token_index -> expert_index -> F32.t
}

inline_for_extraction noextract
let producer_post (tokens:token_matrix) (weights:weight_matrix)
  (actual:producer_observation) : prop =
  (forall t e.
    actual.softmax_inputs t e == actual.projected_scores t e /\
    F.is_finite (actual.projected_scores t e) /\
    A.v_approximates (actual.projected_scores t e)
      (C.acc2 (Projection.projection true tokens weights) t e)) /\
  (forall t e.
    F.kind (actual.probability_rows t e) == F.Finite /\
    F.lte F.zero (actual.probability_rows t e) /\
    F.lte (actual.probability_rows t e) F.one /\
    A.v_approximates (actual.probability_rows t e)
      (C.acc2 (Projection.projection_softmax true tokens weights) t e))

inline_for_extraction noextract
let numerical_execution (tokens:token_matrix) (weights:weight_matrix) =
  actual:producer_observation{producer_post tokens weights actual}

inline_for_extraction noextract
let projection_softmax_edge
  (tokens:token_matrix) (weights:weight_matrix)
  (actual:numerical_execution tokens weights)
  (t:token_index) (e:expert_index)
  : Lemma (
      actual.softmax_inputs t e == actual.projected_scores t e /\
      A.v_approximates (actual.softmax_inputs t e)
        (C.acc2 (Projection.projection true tokens weights) t e))
  = ()

inline_for_extraction noextract
let checked_rows (#tokens:token_matrix) (#weights:weight_matrix)
  (actual:numerical_execution tokens weights) : Argmax.probabilities 256 8 =
  fun t e -> actual.probability_rows t e

inline_for_extraction noextract
let native_rows (#s:nat) (#e:pos) (rows:Argmax.probabilities s e)
  : Argmax.index s -> Capacity.native_row e =
  fun t j -> rows t j

inline_for_extraction noextract
let rec selector_compatibility (#n:pos) (row:Argmax.row n)
  (k:pos{k <= n})
  : Lemma
      (ensures Argmax.best_prefix #n row k ==
        Capacity.selection_prefix #n (fun e -> row e) k)
      (decreases k)
  =
  if k = 1 then () else selector_compatibility #n row (k - 1)

inline_for_extraction noextract
let reference_probability_positive
  (tokens:token_matrix) (weights:weight_matrix)
  (t:token_index) (e:expert_index)
  : Lemma (
      C.acc2 (Projection.projection_softmax true tokens weights) t e >. 0.0R)
  =
  Projection.probability_entry true tokens weights t e;
  Projection.row_normalizer_positive true
    (Projection.projection true tokens weights) t;
  Kuiper.Real.exp_positive
    (C.acc2 (Projection.projection true tokens weights) t e)

(* The positive witness is the existing softmax model at the ACTUAL native
   selected index. It is not to_real of the gathered FP32 cell. *)
inline_for_extraction noextract
let selected_boundary (tokens:token_matrix) (weights:weight_matrix)
  (actual:numerical_execution tokens weights)
  : GTot (Capacity.selected_boundary Capacity.Smoke) =
  let rows = native_rows #256 #8 (checked_rows actual) in
  {
    Capacity.probability_rows = rows;
    Capacity.selected_expert =
      (fun t -> Capacity.reference_selection #8 (rows t));
    Capacity.selected_probability =
      (fun t -> actual.probability_rows t
        (Capacity.reference_selection #8 (rows t)));
    Capacity.selected_model =
      (fun t ->
        let e = Capacity.reference_selection #8 (rows t) in
        reference_probability_positive tokens weights t e;
        C.acc2 (Projection.projection_softmax true tokens weights) t e)
  }

inline_for_extraction noextract
let selection_consumes_producer
  (tokens:token_matrix) (weights:weight_matrix)
  (actual:numerical_execution tokens weights) (t:token_index)
  : Lemma (
      let b = selected_boundary tokens weights actual in
      let rows = checked_rows actual in
      b.selected_expert t == (Argmax.select #256 #8 rows).selected_expert t /\
      b.selected_probability t == actual.probability_rows t (b.selected_expert t) /\
      b.selected_model t ==
        C.acc2 (Projection.projection_softmax true tokens weights)
          t (b.selected_expert t) /\
      Argmax.stable_maximum #8 (rows t) 8 (b.selected_expert t) /\
      A.v_approximates (b.selected_probability t)
        ((Argmax.select #256 #8 rows).selected_probability t))
  =
  let rows = checked_rows actual in
  selector_compatibility #8 (rows t) 8;
  Argmax.select_correct #256 #8 rows t

inline_for_extraction noextract
let bf16_promotion_model (value:BF16.t)
  : Lemma (
      A.v_approximates (Cast.fcast #BF16.t #F32.t value)
        (A.to_real value))
  =
  A.to_real_ok value;
  Cast.fcast_approx #BF16.t #F32.t value (A.to_real value)

inline_for_extraction noextract noeq
type normalization_observation = {
  selected_experts: token_index -> expert_index;
  gathered_probabilities: token_index -> F32.t;
  selected_sums: token_index -> F32.t;
  combine_probabilities: token_index -> F32.t
}

inline_for_extraction noextract
let normalization_post (b:Capacity.selected_boundary Capacity.Smoke)
  (actual:normalization_observation) : prop =
  forall t.
    actual.selected_experts t == b.selected_expert t /\
    actual.gathered_probabilities t == b.selected_probability t /\
    A.v_approximates (actual.selected_sums t)
      (Capacity.selected_set_sum (b.selected_model t)) /\
    A.v_approximates (actual.combine_probabilities t)
      (Capacity.normalized_probability b t)

(* Realize the normalization interface using precisely the gathered cell
   and a certified realization of its singleton sum. Other realizations
   must prove the same numerical relation, not a different operation. *)
inline_for_extraction noextract
let normalization_from_division
  (b:Capacity.selected_boundary Capacity.Smoke)
  (sums:token_index -> F32.t)
  (_:squash (forall t. A.v_approximates (sums t)
       (Capacity.selected_set_sum (b.selected_model t))))
  : actual:normalization_observation{normalization_post b actual} =
  let each (t:token_index)
    : Lemma (A.v_approximates
        (F.div (b.selected_probability t) (sums t))
        (Capacity.normalized_probability b t))
    = Capacity.normalization_approx b t (sums t)
  in
  Classical.forall_intro each;
  {
    selected_experts = b.selected_expert;
    gathered_probabilities = b.selected_probability;
    selected_sums = sums;
    combine_probabilities =
      (fun t -> F.div (b.selected_probability t) (sums t))
  }

inline_for_extraction noextract
let singleton_normalization (b:Capacity.selected_boundary Capacity.Smoke)
  : actual:normalization_observation{normalization_post b actual} =
  Classical.forall_intro (Capacity.singleton_sum_approx b);
  normalization_from_division b b.selected_probability ()

inline_for_extraction noextract
let packing_post (b:Capacity.selected_boundary Capacity.Smoke)
  (normalization:normalization_observation) (out:public_outputs) : prop =
  forall e c.
    let owner = Capacity.slot_owner b 256 e c in
    if owner < 0 then A.v_approximates (out.probabilities e c) 0.0R
    else out.probabilities e c == normalization.combine_probabilities owner

inline_for_extraction noextract
let postcondition
  (tokens:token_matrix) (weights:weight_matrix)
  (tokens_after:token_matrix) (weights_after:weight_matrix)
  (actual:numerical_execution tokens weights)
  (normalization:normalization_observation) (out:public_outputs) : prop =
  let b = selected_boundary tokens weights actual in
  input_domain tokens weights /\
  tokens_after == tokens /\ weights_after == weights /\
  normalization_post b normalization /\
  packing_post b normalization out /\
  Capacity.route_post b out

(* The public task is relation-valued. Different certified native rows may
   select different experts and therefore have different capacity histories. *)
inline_for_extraction noextract
let task_relation
  (tokens:token_matrix) (weights:weight_matrix)
  (tokens_after:token_matrix) (weights_after:weight_matrix)
  (out:public_outputs) : prop =
  exists (actual:numerical_execution tokens weights)
         (normalization:normalization_observation).
    postcondition tokens weights tokens_after weights_after
      actual normalization out

inline_for_extraction noextract
let postcondition_instantiates_task
  (tokens:token_matrix) (weights:weight_matrix)
  (tokens_after:token_matrix) (weights_after:weight_matrix)
  (actual:numerical_execution tokens weights)
  (normalization:normalization_observation) (out:public_outputs)
  : Lemma
      (requires postcondition tokens weights tokens_after weights_after
        actual normalization out)
      (ensures task_relation tokens weights tokens_after weights_after out)
  = ()

inline_for_extraction noextract
let complete_from_prefix_run
  (tokens:token_matrix) (weights:weight_matrix)
  (tokens_after:token_matrix) (weights_after:weight_matrix)
  (actual:numerical_execution tokens weights)
  (normalization:normalization_observation) (out:public_outputs)
  : Lemma
      (requires (
        let b = selected_boundary tokens weights actual in
        input_domain tokens weights /\
        tokens_after == tokens /\ weights_after == weights /\
        normalization_post b normalization /\
        packing_post b normalization out /\
        Capacity.state_representation
          (Capacity.prefix_run b 0 256 (Capacity.initial_state Capacity.Smoke)) out))
      (ensures postcondition tokens weights tokens_after weights_after
        actual normalization out)
  =
  let b = selected_boundary tokens weights actual in
  Capacity.completed_run_law b out;
  Projection.projection_softmax_satisfies_postcondition true tokens weights

inline_for_extraction noextract
let reference_output_cells
  (tokens:token_matrix) (weights:weight_matrix)
  (tokens_after:token_matrix) (weights_after:weight_matrix)
  (actual:numerical_execution tokens weights)
  (normalization:normalization_observation) (out:public_outputs)
  (t:token_index) (e:expert_index) (c:slot_index)
  : Lemma
      (requires postcondition tokens weights tokens_after weights_after
        actual normalization out)
      (ensures (
        let b = selected_boundary tokens weights actual in
        FStar.Int32.v (out.expert_demand e) == Capacity.assigned_count b 256 e /\
        FStar.Int64.v (out.token_indices e c) == Capacity.slot_owner b 256 e c /\
        A.v_approximates (out.probabilities e c)
          (Capacity.slot_probability b 256 e c) /\
        FStar.Int32.v (out.survival t e) ==
          (if b.selected_expert t = e then 1 else 0) *
          (if Capacity.assigned_count b (t + 1) e <= 32 then 1 else 0)))
  =
  Capacity.survival_reference_law
    (selected_boundary tokens weights actual) 256 t e

inline_for_extraction noextract
let same_rows_same_capacity
  (tokens:token_matrix) (weights:weight_matrix)
  (a b:numerical_execution tokens weights) (out:public_outputs)
  : Lemma
      (requires
        (forall t e. a.probability_rows t e == b.probability_rows t e) /\
        Capacity.route_post (selected_boundary tokens weights a) out)
      (ensures Capacity.route_post (selected_boundary tokens weights b) out)
  =
  Capacity.route_post_native_extensional
    (selected_boundary tokens weights a)
    (selected_boundary tokens weights b) out
