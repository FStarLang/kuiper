module KPlay.Proof.BaseHistory

open Kuiper

module S = FStar.Seq
module C = Kuiper.Chest
module F = Kuiper.Floating
module A = Kuiper.Approximates.Base
module I32 = FStar.Int32
module I64 = FStar.Int64
module Classical = FStar.Classical
module Shared = KPlay.SharedSpec
module Projection = KPlay.Proof.Shared.ProjectionSoftmax
module Capacity = KPlay.Proof.Shared.Capacity
module BP = KPlay.Proof.BaseProjection
module BC = KPlay.Proof.BaseCapacity
module Classification = KPlay.Proof.BaseClassification

inline_for_extraction noextract
let prefix = n:nat{n <= 256}

inline_for_extraction noextract
let history (n:prefix) = lseq Classification.score_row n

inline_for_extraction noextract
let row (scores:Classification.score_row) : lseq f32 8 =
  BP.normalized_values scores

inline_for_extraction noextract
let choice (scores:Classification.score_row) : Shared.expert_index =
  BP.maximum_index (row scores) 8

inline_for_extraction noextract
let native_row (scores:Classification.score_row) : Capacity.native_row 8 =
  Classification.normalized_range scores;
  fun e -> S.index (row scores) e

inline_for_extraction noextract
let gathered (scores:Classification.score_row) : f32 =
  S.index (row scores) (choice scores)

inline_for_extraction noextract
let combined (scores:Classification.score_row) : f32 =
  BC.selected_weight (gathered scores)

inline_for_extraction noextract
let certificate (x:Shared.token_matrix) (w:Shared.weight_matrix)
  (t:Shared.token_index) (scores:Classification.score_row) : prop =
  C.seq_to_chest1 scores %~
    C.chest2_row (Projection.projection true x w) t /\
  C.seq_to_chest1 (row scores) %~
    C.chest2_row (Projection.projection_softmax true x w) t /\
  choice scores == Capacity.reference_selection (native_row scores)

inline_for_extraction noextract
let valid (x:Shared.token_matrix) (w:Shared.weight_matrix)
  (#n:prefix) (h:history n) : prop =
  forall (t:nat{t < n}). certificate x w t (S.index h t)

inline_for_extraction noextract
let push (#n:nat{n < 256}) (h:history n) (scores:Classification.score_row)
  : history (n + 1) =
  S.append h (S.create 1 scores)

inline_for_extraction noextract
let valid_push (x:Shared.token_matrix) (w:Shared.weight_matrix)
  (#n:nat{n < 256}) (h:history n) (scores:Classification.score_row)
  : Lemma
      (requires valid x w h /\ certificate x w n scores)
      (ensures valid x w (push h scores))
  =
  let cell (t:nat{t < n + 1})
    : Lemma (certificate x w t (S.index (push h scores) t))
    = if t < n then () else assert (t == n)
  in Classical.forall_intro cell

inline_for_extraction noextract
let compatible (#n:prefix) (h:history n)
  (b:Capacity.selected_boundary Capacity.Smoke) : prop =
  forall (t:nat{t < n}).
    b.selected_expert t == choice (S.index h t) /\
    (forall (e:Shared.expert_index).
      b.probability_rows t e == S.index (row (S.index h t)) e)

inline_for_extraction noextract
let bounded_counts (#n:prefix) (st:BC.physical_state Capacity.Smoke n) : prop =
  forall (e:Shared.expert_index).
    0 <= I64.v (S.index st.counts e) /\ I64.v (S.index st.counts e) <= n

(* Quantifying over extensions does not fabricate future observations.
   A matching full boundary is constructed only after the actual loop. *)
inline_for_extraction noextract
let prefix_representation (#n:prefix) (h:history n)
  (st:BC.physical_state Capacity.Smoke n) : prop =
  forall (b:Capacity.selected_boundary Capacity.Smoke).
    compatible h b ==>
    BC.state_representation Capacity.Smoke n st (Capacity.prefix_state b n)

inline_for_extraction noextract
let packed_values (#n:prefix) (h:history n)
  (st:BC.physical_state Capacity.Smoke n) : prop =
  forall (e:Shared.expert_index) (c:Shared.slot_index).
    let i = e * 32 + c in
    let owner = I64.v (S.index st.indices i) in
    if owner < 0 then owner == -1 /\ S.index st.weights i == F.zero
    else owner < n /\ S.index st.weights i == combined (S.index h owner)

inline_for_extraction noextract
let prefix_ok (#n:prefix) (h:history n)
  (st:BC.physical_state Capacity.Smoke n) : prop =
  bounded_counts st /\ prefix_representation h st /\ packed_values h st

inline_for_extraction noextract
let initial : BC.physical_state Capacity.Smoke 0 = {
  BC.counts = S.create 8 0L;
  BC.indices = S.create 256 (-1L);
  BC.weights = S.create 256 (F.zero #f32);
  BC.flags = S.empty
}

inline_for_extraction noextract
let initial_invariant ()
  : Lemma (prefix_ok (S.empty <: history 0) initial)
  =
  let boundary (b:Capacity.selected_boundary Capacity.Smoke)
    : Lemma (
        BC.state_representation Capacity.Smoke 0 initial (Capacity.prefix_state b 0))
    = Capacity.initial_prefix_law b
  in Classical.forall_intro boundary

inline_for_extraction noextract
let extend (#n:nat{n < 256})
  (st:BC.physical_state Capacity.Smoke n) (current:lseq i32 8)
  : BC.physical_state Capacity.Smoke (n + 1) = {
  BC.counts = st.counts;
  BC.indices = st.indices;
  BC.weights = st.weights;
  BC.flags = S.append st.flags current
}

inline_for_extraction noextract
let zero_extend (#n:nat{n < 256}) (st:BC.physical_state Capacity.Smoke n)
  : BC.physical_state Capacity.Smoke (n + 1) =
  extend st (S.create 8 0l)

inline_for_extraction noextract
let extend_representation (#n:nat{n < 256})
  (b:Capacity.selected_boundary Capacity.Smoke)
  (st:BC.physical_state Capacity.Smoke n)
  : Lemma
      (requires BC.state_representation Capacity.Smoke n st (Capacity.prefix_state b n))
      (ensures BC.state_representation Capacity.Smoke (n + 1)
        (zero_extend st) (Capacity.prefix_state b n))
  =
  let cell (t:Shared.token_index) (e:Shared.expert_index)
    : Lemma (
        if t < n + 1 then
          I32.v (S.index (zero_extend st).BC.flags (t * 8 + e)) ==
            Capacity.prefix_survival b n t e
        else Capacity.prefix_survival b n t e == 0)
    =
    if t < n then (
      assert (BC.survival_offset Capacity.Smoke n t e == t * 8 + e);
      assert (I32.v (S.index st.flags (t * 8 + e)) ==
        (Capacity.prefix_state b n).survival_cells t e);
      assert (S.length st.flags == n * 8);
      assert (S.index (zero_extend st).BC.flags (t * 8 + e) ==
        S.index st.flags (t * 8 + e))
    ) else if t = n then (
      assert (S.length st.flags == n * 8);
      assert (S.index (zero_extend st).BC.flags (t * 8 + e) == 0l)
    ) else ()
  in Classical.forall_intro_2 cell

inline_for_extraction noextract
let compatible_prefix (#n:nat{n < 256}) (h:history n)
  (scores:Classification.score_row) (b:Capacity.selected_boundary Capacity.Smoke)
  : Lemma
      (requires compatible (push h scores) b)
      (ensures compatible h b /\
        b.selected_expert n == choice scores /\
        BC.row_representation Capacity.Smoke b n (row scores))
  =
  let cell (t:nat{t < n})
    : Lemma (
        b.selected_expert t == choice (S.index h t) /\
        (forall (e:Shared.expert_index).
          b.probability_rows t e == S.index (row (S.index h t)) e))
    = ()
  in Classical.forall_intro cell

inline_for_extraction noextract
let advance (#n:nat{n < 256}) (h:history n)
  (scores:Classification.score_row)
  (before:BC.physical_state Capacity.Smoke n)
  (after:BC.physical_state Capacity.Smoke (n + 1))
  : Lemma
      (requires prefix_ok h before /\
        BC.native_step Capacity.Smoke n (choice scores) (row scores)
          (zero_extend before) after)
      (ensures prefix_ok (push h scores) after)
  =
  let count (e:Shared.expert_index)
    : Lemma (
        0 <= I64.v (S.index after.counts e) /\ I64.v (S.index after.counts e) <= n + 1)
    = ()
  in Classical.forall_intro count;
  let prefix (b:Capacity.selected_boundary Capacity.Smoke)
    : Lemma
        (requires compatible (push h scores) b)
        (ensures BC.state_representation Capacity.Smoke (n + 1) after
          (Capacity.prefix_state b (n + 1)))
    =
    compatible_prefix h scores b;
    extend_representation b before;
    let selected = S.create 1 (I32.int_to_t (choice scores)) in
    BC.native_step_refines_prefix Capacity.Smoke b n selected (row scores)
      (zero_extend before) after
  in
  Classical.forall_intro (Classical.move_requires prefix);
  let slot (e:Shared.expert_index) (c:Shared.slot_index)
    : Lemma (
        let i = e * 32 + c in
        let owner = I64.v (S.index after.indices i) in
        if owner < 0 then owner == -1 /\ S.index after.weights i == F.zero
        else owner < n + 1 /\
          S.index after.weights i == combined (S.index (push h scores) owner))
    =
    let i = BC.slot_offset Capacity.Smoke e c in
    assert (i == e * 32 + c);
    assert ((zero_extend before).BC.counts == before.counts /\
      (zero_extend before).BC.indices == before.indices /\
      (zero_extend before).BC.weights == before.weights);
    if e = choice scores && I64.v (S.index before.counts e) = c then (
      assert (I64.v (S.index after.indices i) == n);
      assert (S.index after.weights i == BC.selected_weight (S.index (row scores) (choice scores)));
      assert (S.index (push h scores) n == scores)
    ) else (
      assert (S.index after.indices i == S.index before.indices i /\
        S.index after.weights i == S.index before.weights i);
      let owner = I64.v (S.index before.indices i) in
      if owner < 0 then (
        assert (owner == -1 /\ S.index before.weights i == F.zero)
      ) else (
        assert (owner < n /\ S.index before.weights i == combined (S.index h owner));
        assert (S.index (push h scores) owner == S.index h owner)
      )
    )
  in Classical.forall_intro_2 slot

inline_for_extraction noextract
let observation (h:history 256) : Shared.producer_observation = {
  Shared.projected_scores = (fun t e -> S.index (S.index h t) e);
  Shared.softmax_inputs = (fun t e -> S.index (S.index h t) e);
  Shared.probability_rows = (fun t e -> S.index (row (S.index h t)) e)
}

inline_for_extraction noextract
let observation_valid (x:Shared.token_matrix) (w:Shared.weight_matrix)
  (h:history 256)
  : Lemma
      (requires valid x w h)
      (ensures Shared.producer_post x w (observation h))
  =
  let cell (t:Shared.token_index) (e:Shared.expert_index)
    : Lemma (
        F.is_finite (S.index (S.index h t) e) /\
        A.v_approximates (S.index (S.index h t) e)
          (C.acc2 (Projection.projection true x w) t e) /\
        Kuiper.Floating.Bounds.unit (S.index (row (S.index h t)) e) /\
        A.v_approximates (S.index (row (S.index h t)) e)
          (C.acc2 (Projection.projection_softmax true x w) t e))
    = Classification.normalized_range (S.index h t)
  in Classical.forall_intro_2 cell

inline_for_extraction noextract
let numerical_execution (x:Shared.token_matrix) (w:Shared.weight_matrix)
  (h:history 256{valid x w h}) : Shared.numerical_execution x w =
  observation_valid x w h;
  observation h

inline_for_extraction noextract
let final_boundary_compatible (x:Shared.token_matrix) (w:Shared.weight_matrix)
  (h:history 256{valid x w h})
  : Lemma (
      compatible h (Shared.selected_boundary x w (numerical_execution x w h)))
  =
  let b = Shared.selected_boundary x w (numerical_execution x w h) in
  let token (t:Shared.token_index)
    : Lemma (
        b.selected_expert t == choice (S.index h t) /\
        (forall (e:Shared.expert_index).
          b.probability_rows t e == S.index (row (S.index h t)) e))
    =
    Capacity.selection_prefix_extensional (b.probability_rows t)
      (native_row (S.index h t)) 8
  in Classical.forall_intro token

inline_for_extraction noextract
let normalization (h:history 256) : Shared.normalization_observation = {
  Shared.selected_experts = (fun t -> choice (S.index h t));
  Shared.gathered_probabilities = (fun t -> gathered (S.index h t));
  Shared.selected_sums = (fun t -> F.add (F.zero #f32) (gathered (S.index h t)));
  Shared.combine_probabilities = (fun t -> combined (S.index h t))
}

inline_for_extraction noextract
let normalization_valid (x:Shared.token_matrix) (w:Shared.weight_matrix)
  (h:history 256{valid x w h})
  : Lemma (Shared.normalization_post
      (Shared.selected_boundary x w (numerical_execution x w h)) (normalization h))
  =
  final_boundary_compatible x w h;
  let b = Shared.selected_boundary x w (numerical_execution x w h) in
  let token (t:Shared.token_index)
    : Lemma (
        (normalization h).selected_experts t == b.selected_expert t /\
        (normalization h).gathered_probabilities t == b.selected_probability t /\
        A.v_approximates ((normalization h).selected_sums t)
          (Capacity.selected_set_sum (b.selected_model t)) /\
        A.v_approximates ((normalization h).combine_probabilities t)
          (Capacity.normalized_probability b t))
    =
    A.a_add (F.zero #f32) (gathered (S.index h t)) 0.0R (b.selected_model t);
    BC.selected_weight_represents (gathered (S.index h t)) (b.selected_model t)
  in Classical.forall_intro token

inline_for_extraction noextract
let output (demand:lseq i32 8) (st:BC.physical_state Capacity.Smoke 256)
  : Shared.public_outputs = {
  Capacity.expert_demand = (fun e -> S.index demand e);
  Capacity.token_indices = (fun e c -> S.index st.indices (e * 32 + c));
  Capacity.probabilities = (fun e c -> S.index st.weights (e * 32 + c));
  Capacity.survival = (fun t e -> S.index st.flags (t * 8 + e))
}

inline_for_extraction noextract
let final_refines (x:Shared.token_matrix) (w:Shared.weight_matrix)
  (h:history 256) (st:BC.physical_state Capacity.Smoke 256)
  (demand:lseq i32 8)
  : Lemma
      (requires Shared.input_domain x w /\ valid x w h /\ prefix_ok h st /\
        (forall (e:Shared.expert_index).
          I32.v (S.index demand e) == I64.v (S.index st.counts e)))
      (ensures Shared.task_relation x w x w (output demand st))
  =
  final_boundary_compatible x w h;
  normalization_valid x w h;
  let actual = numerical_execution x w h in
  let b = Shared.selected_boundary x w actual in
  assert (BC.state_representation Capacity.Smoke 256 st (Capacity.prefix_state b 256));
  assert (Capacity.route_post b (output demand st));
  let slot (e:Shared.expert_index) (c:Shared.slot_index)
    : Lemma (
        let owner = Capacity.slot_owner b 256 e c in
        if owner < 0 then A.v_approximates ((output demand st).probabilities e c) 0.0R
        else (output demand st).probabilities e c ==
          (normalization h).combine_probabilities owner)
    = ()
  in Classical.forall_intro_2 slot;
  Shared.postcondition_instantiates_task x w x w actual (normalization h) (output demand st)
