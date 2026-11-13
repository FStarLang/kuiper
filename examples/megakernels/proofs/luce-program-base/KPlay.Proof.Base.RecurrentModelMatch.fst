module KPlay.Proof.Base.RecurrentModelMatch

#set-options "--fuel 1 --ifuel 0 --z3rlimit 40"

module P = KPlay.Proof.Shared.Projection
module R = KPlay.Proof.Shared.Recurrent
module H = KPlay.Proof.Base.RecurrentHeadModel
module Prior = KPlay.Proof.Base.RecurrentPipeline
module Bridge = KPlay.Proof.Base.RecurrentPipelineBridge

inline_for_extraction noextract
let plan_valid (call:Prior.logical_call) (stages:Prior.decoder_stages) (h:P.index 16)
  : Lemma
      (requires stages == Prior.build_stages call)
      (ensures Prior.valid_head (stages.Prior.head_plans h)) =
  Prior.stage_head_view_lemma call h;
  norm_spec [delta_once ["KPlay.Proof.Base.RecurrentPipeline.stage_head_view"]] (Prior.stage_head_view call h)

inline_for_extraction noextract
let state_matches (call:Prior.logical_call) (stages:Prior.decoder_stages)
  (h:P.index 16) (i j:P.index 128)
  : Lemma
      (requires stages == Prior.build_stages call)
      (ensures H.next_column (Bridge.inputs (stages.Prior.head_plans h)) j i ==
        (R.decoder_values call.Prior.call call.Prior.weight call.Prior.input call.Prior.before).R.recurrent_value h i j) =
  plan_valid call stages h;
  Bridge.updated (stages.Prior.head_plans h) j;
  Prior.models_match call

inline_for_extraction noextract
let gated_matches (call:Prior.logical_call) (stages:Prior.decoder_stages)
  (h:P.index 16) (i:P.index 128)
  : Lemma
      (requires stages == Prior.build_stages call)
      (ensures H.gated (Bridge.inputs (stages.Prior.head_plans h)) i ==
        stages.Prior.gated_values (h * 128 + i)) =
  let canonical = Prior.build_stages call in
  plan_valid call stages h;
  Bridge.gated (stages.Prior.head_plans h) i;
  assert ((h * 128 + i) / 128 == h /\ (h * 128 + i) % 128 == i)
