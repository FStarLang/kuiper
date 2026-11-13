module KPlay.Proof.Optimized.Numerical

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
module S = FStar.Seq
module C = FStar.Classical
module Approx = Kuiper.Approximates.Base
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module O = KPlay.Proof.Shared.Observations
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module Layout = KPlay.Proof.Optimized.Layout
module State = KPlay.Proof.Optimized.State
module Layer = KPlay.Proof.Optimized.Layer
module Book = KPlay.Proof.Optimized.LayerBook
module Step = KPlay.Proof.Optimized.Step
module Embedding = KPlay.Proof.Optimized.Embedding
module RF = KPlay.Proof.Optimized.RecurrentFamily
module AF = KPlay.Proof.Optimized.AttentionFamily
module FA = KPlay.Proof.Optimized.FamilyAttention
module Final = KPlay.Proof.Optimized.FinalMath

inline_for_extraction noextract
let scope (position:nat{position < 2048}) (book:Book.book position{book.Book.count == 24}) : prop =
  forall (i:A.layer_index).
    match S.index book.Book.entries i with
    | Layer.LinearCall _ _ _ -> True
    | Layer.AttentionCall _ _ produced -> FA.bilateral_live_scores_finite position produced.scores

inline_for_extraction noextract
let dependencies (call:Model.invocation) (w:Layout.packed)
  (position:nat{position < Model.calls call}) (token:D.token 248320)
  (incoming:A.hybrid_cache{Model.cache_shape position 0 incoming})
  (book:Book.book position{book.Book.count == 24}) : prop =
  forall (i:A.layer_index).
    match S.index book.Book.entries i with
    | Layer.LinearCall layer before produced -> RF.dependency w position layer before produced
    | Layer.AttentionCall layer before produced ->
        AF.dependency call w position layer before produced
          (Model.layers_at call (O.model_parameters (Layout.parameter_values w)) position token incoming layer)

let rec stage_refines (call:Model.invocation) (w:Layout.packed)
  (position:nat{position < Model.calls call}) (token:D.token 248320)
  (incoming:A.hybrid_cache{Model.cache_shape position 0 incoming})
  (book:Book.book position{book.Book.count == 24}) (next:Model.cursor)
  : Lemma
      (requires Book.well_formed book /\
        book.Book.initial.State.hidden == Embedding.row w token /\
        O.cache_approximates book.Book.initial.State.cache incoming /\
        dependencies call w position token incoming book /\ scope position book)
      (ensures (
        let expected = Model.layers_at call (O.model_parameters (Layout.parameter_values w))
          position token incoming next in
        (forall (i:P.index 1024).
          Approx.v_approximates (S.index (Book.at book next).State.hidden i) (expected.hidden i)) /\
        O.cache_approximates (Book.at book next).State.cache expected.cache))
      (decreases next) =
  Model.full_profile call;
  if next = 0 then (
    Embedding.row_binding w token;
    O.embedding_models_input (Layout.parameter_values w) token
  ) else (
    stage_refines call w position token incoming book (next - 1);
    let previous = Model.layers_at call (O.model_parameters (Layout.parameter_values w))
      position token incoming (next - 1) in
    let r = S.index book.Book.entries (next - 1) in
    assert (Layer.layer_of r == next - 1 /\ Layer.before_of r == Book.at book (next - 1));
    match r with
    | Layer.LinearCall layer before produced ->
        RF.refines call w position layer before produced previous
    | Layer.AttentionCall layer before produced ->
        AF.refines call w position layer before produced previous
  )

#push-options "--fuel 0 --ifuel 0 --z3rlimit 60"
let model_logits_form (call:Model.invocation) (w:Layout.packed)
  (position:nat{position < Model.calls call}) (token:D.token 248320)
  (incoming:A.hybrid_cache{Model.cache_shape position 0 incoming})
  : Lemma (
      (Model.model_step call (O.model_parameters (Layout.parameter_values w))
        position token incoming).Model.logits ==
      Final.logits w (fun i ->
        (Model.layers_at call (O.model_parameters (Layout.parameter_values w))
          position token incoming 24).A.hidden i)) =
  assert (
    (Model.model_step call (O.model_parameters (Layout.parameter_values w))
      position token incoming).Model.logits ==
    Final.logits w (fun i ->
      (Model.layers_at call (O.model_parameters (Layout.parameter_values w))
        position token incoming 24).A.hidden i))
  by (FStar.Tactics.V2.norm [delta_only [
        `%Model.model_step; `%Final.logits; `%Final.normalized]; zeta; iota];
      FStar.Tactics.V2.trefl ())

let refines (call:Model.invocation) (w:Layout.packed)
  (position:nat{position < Model.calls call})
  (step:Step.data position)
  (incoming:A.hybrid_cache{Model.cache_shape position 0 incoming})
  : Lemma
      (requires Step.well_formed w step /\
        O.cache_approximates step.Step.layers.Book.initial.State.cache incoming /\
        dependencies call w position step.Step.consumed incoming step.Step.layers /\
        scope position step.Step.layers)
      (ensures O.numerical call (O.model_parameters (Layout.parameter_values w))
        position step.Step.consumed incoming (Step.observed step)) =
  Model.full_profile call;
  FStar.Pervasives.norm_spec [delta_only [`%Step.well_formed]; zeta; iota] (Step.well_formed w step);
  FStar.Pervasives.norm_spec [delta_only [`%Book.last]; zeta; iota] (Book.last step.Step.layers);
  let each_stage (next:Model.cursor) : Lemma (
      let expected = Model.layers_at call (O.model_parameters (Layout.parameter_values w))
        position step.Step.consumed incoming next in
      (forall (i:P.index 1024).
        Approx.v_approximates (S.index (Book.at step.Step.layers next).State.hidden i) (expected.hidden i)) /\
      O.cache_approximates (Book.at step.Step.layers next).State.cache expected.cache) =
    stage_refines call w position step.Step.consumed incoming step.Step.layers next
  in
  C.forall_intro each_stage;
  Final.propagated w (Book.last step.Step.layers).State.hidden
    (fun i -> (Model.layers_at call (O.model_parameters (Layout.parameter_values w))
      position step.Step.consumed incoming 24).A.hidden i) step.Step.chosen step.Step.head;
  model_logits_form call w position step.Step.consumed incoming;
  FStar.Pervasives.norm_spec [delta_only [`%Final.normalized]; zeta; iota]
    (Final.normalized w (fun i ->
      (Model.layers_at call (O.model_parameters (Layout.parameter_values w))
        position step.Step.consumed incoming 24).A.hidden i));
  FStar.Pervasives.norm_spec [delta_only [
    `%O.numerical; `%O.vector_approximates; `%Step.observed; `%State.logical_stage]; zeta; iota]
    (O.numerical call (O.model_parameters (Layout.parameter_values w))
      position step.Step.consumed incoming (Step.observed step))
#pop-options
