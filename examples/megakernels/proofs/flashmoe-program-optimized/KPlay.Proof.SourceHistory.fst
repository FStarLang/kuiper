module KPlay.Proof.SourceHistory

open Kuiper

module S = FStar.Seq
module F = Kuiper.Floating
module A = Kuiper.Approximates.Base
module I32 = FStar.Int32
module Layout = KPlay.Proof.SourceLayouts
module Project = KPlay.Proof.SourceProjection
module Routing = KPlay.Proof.SourceRouting
module Steps = KPlay.Proof.RouteSteps
module Shared = KPlay.SharedSpec
module Capacity = KPlay.Proof.Shared.Capacity

inline_for_extraction noextract
let observation (x:Layout.token_values) (w:Layout.weight_values)
  (h:Routing.history 256) : GTot Shared.producer_observation =
  let produced = Project.scores x w in {
  projected_scores = (fun t e -> S.index produced (t * 8 + e));
  softmax_inputs = (fun t e -> S.index produced (t * 8 + e));
  probability_rows = (fun t e -> S.index (S.index h t) e)
}

inline_for_extraction noextract
let observation_valid (x:Layout.token_values) (w:Layout.weight_values)
  (h:Routing.history 256)
  : Lemma
      (requires Shared.input_domain (Layout.logical_tokens x) (Layout.logical_weights w) /\
        Routing.valid x w h)
      (ensures Shared.producer_post (Layout.logical_tokens x) (Layout.logical_weights w)
        (observation x w h))
  =
  let cell (t:natlt 256) (e:natlt 8)
    : Lemma (
        F.is_finite (S.index (Project.scores x w) (t * 8 + e)) /\
        A.v_approximates (S.index (Project.scores x w) (t * 8 + e))
          (Kuiper.Chest.acc2 (KPlay.Proof.Shared.ProjectionSoftmax.projection true
            (Layout.logical_tokens x) (Layout.logical_weights w)) t e))
    = Project.score_certificate x w t e
  in FStar.Classical.forall_intro_2 cell

inline_for_extraction noextract
let compatible (h:Routing.history 256) (b:Capacity.selected_boundary Capacity.Smoke) : prop =
  forall (t:natlt 256) (e:natlt 8).
    b.probability_rows t e == S.index (S.index h t) e

inline_for_extraction noextract
let choice (h:Routing.history 256) (t:natlt 256) : natlt 8 =
  I32.v (Steps.scan_prefix (S.index h t) 8).chosen

inline_for_extraction noextract
let combined (h:Routing.history 256) (t:natlt 256) : f32 =
  Steps.emitted_probability (S.index h t) (choice h t)

inline_for_extraction noextract
let selection (h:Routing.history 256) (b:Capacity.selected_boundary Capacity.Smoke)
  (t:natlt 256)
  : Lemma
      (requires compatible h b)
      (ensures
        choice h t == b.selected_expert t /\
        (Steps.scan_prefix (S.index h t) 8).best == b.selected_probability t /\
        combined h t == F.div (b.selected_probability t)
          (F.add F.zero (b.selected_probability t)) /\
        A.v_approximates (combined h t) (Capacity.normalized_probability b t))
  =
  Steps.same_row_selection (S.index h t);
  Capacity.selection_prefix_extensional (Steps.native_row (S.index h t)) (b.probability_rows t) 8;
  A.a_add F.zero (b.selected_probability t) 0.0R (b.selected_model t);
  Capacity.normalization_approx b t (F.add F.zero (b.selected_probability t))

inline_for_extraction noextract
let normalization (h:Routing.history 256) : Shared.normalization_observation = {
  selected_experts = choice h;
  gathered_probabilities = (fun t -> (Steps.scan_prefix (S.index h t) 8).best);
  selected_sums = (fun t -> F.add F.zero (Steps.scan_prefix (S.index h t) 8).best);
  combine_probabilities = combined h
}

inline_for_extraction noextract
let normalization_valid (h:Routing.history 256) (b:Capacity.selected_boundary Capacity.Smoke)
  : Lemma
      (requires compatible h b)
      (ensures Shared.normalization_post b (normalization h))
  =
  let cell (t:natlt 256)
    : Lemma (
      choice h t == b.selected_expert t /\
      (Steps.scan_prefix (S.index h t) 8).best == b.selected_probability t /\
      A.v_approximates (F.add F.zero (Steps.scan_prefix (S.index h t) 8).best)
        (Capacity.selected_set_sum (b.selected_model t)) /\
      A.v_approximates (combined h t) (Capacity.normalized_probability b t))
    =
    selection h b t;
    A.a_add F.zero (b.selected_probability t) 0.0R (b.selected_model t)
  in FStar.Classical.forall_intro cell

inline_for_extraction noextract
let flag_and_weight (h:Routing.history 256) (b:Capacity.selected_boundary Capacity.Smoke)
  (fs:Routing.flags) (ps:Routing.probabilities) (t:natlt 256) (e:natlt 8)
  : Lemma
      (requires compatible h b /\ Routing.routes_match h fs ps)
      (ensures
        I32.v (S.index fs (t * 8 + e)) == (if b.selected_expert t = e then 1 else 0) /\
        (b.selected_expert t == e ==> S.index ps (t * 8 + e) == combined h t))
  = selection h b t
