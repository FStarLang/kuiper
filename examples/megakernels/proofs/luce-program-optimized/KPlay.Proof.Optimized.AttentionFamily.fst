module KPlay.Proof.Optimized.AttentionFamily

#set-options "--fuel 3 --ifuel 1 --z3rlimit 80"

open Kuiper
module S = FStar.Seq
module C = FStar.Classical
module Approx = Kuiper.Approximates.Base
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module FA = KPlay.Proof.Optimized.FamilyAttention
module Layout = KPlay.Proof.Optimized.Layout
module State = KPlay.Proof.Optimized.State
module Cache = KPlay.Proof.Optimized.Cache
module Native = KPlay.Proof.Optimized.Attention
module Math = KPlay.Proof.Optimized.AttentionMath

inline_for_extraction noextract
let weights (w:Layout.packed) (layer:A.full_attention_layer) : lseq bf16 18352640 =
  Layout.layer_partition layer;
  S.slice w (Layout.layer_base layer) (Layout.layer_base layer + 18352640)

inline_for_extraction noextract
let packed_model (w:Layout.packed) (layer:A.full_attention_layer) : GTot (A.vector 18352640) =
  Layout.layer_partition layer;
  fun i -> Approx.to_real (S.index w (Layout.layer_base layer + i))

inline_for_extraction noextract
let old_keys (position:nat{position < 2048}) (before:State.stage) (layer:A.full_attention_layer)
  : lseq bf16 (FA.previous_elements position) =
  if position = 0 then S.empty else State.local_keys before.State.memory layer

inline_for_extraction noextract
let old_values (position:nat{position < 2048}) (before:State.stage) (layer:A.full_attention_layer)
  : lseq bf16 (FA.previous_elements position) =
  if position = 0 then S.empty else State.local_values before.State.memory layer

inline_for_extraction noextract
let padded (position:nat{position < 2048}) (history:option A.kv_history{A.history_length history == position})
  (keys:bool) : A.vector (FA.previous_elements position) =
  fun i ->
    match history with
    | None -> 0.0R
    | Some previous ->
        let h:P.index 2 = i / 524288 in
        let t:P.index 2048 = (i % 524288) / 256 in
        let d:P.index 256 = i % 256 in
        if t < previous.length
        then (if keys then previous.keys h t d else previous.values h t d)
        else 0.0R

inline_for_extraction noextract
let logical_input (call:Model.invocation) (w:Layout.packed)
  (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (ideal:A.layer_result{Model.cache_shape position layer ideal.cache}) : GTot FA.logical_inputs =
  Model.full_profile call;
  Model.attention_entry call position layer ideal.cache;
  {
    FA.call = Model.attention_call call position layer;
    FA.packed_weights = packed_model w layer;
    FA.input_hidden = ideal.hidden;
    FA.old_keys = padded position (ideal.cache.kv layer) true;
    FA.old_values = padded position (ideal.cache.kv layer) false;
    FA.frame = ideal.cache;
  }

let weight_model_agrees (w:Layout.packed) (layer:A.full_attention_layer)
  : Lemma (
      FA.weights_view (packed_model w layer) ==
        (O.model_parameters (Layout.parameter_values w)).Model.full layer) =
  assert (
    FA.weights_view (packed_model w layer) ==
      (O.model_parameters (Layout.parameter_values w)).Model.full layer)
  by (FStar.Tactics.V2.norm [delta_only [
        `%FA.weights_view; `%packed_model; `%O.model_parameters; `%O.model_full;
        `%Layout.parameter_values; `%Layout.parameters; `%Layout.full_weights]; zeta; iota];
      FStar.Tactics.V2.trefl ())

let history_agrees (call:Model.invocation) (w:Layout.packed)
  (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (ideal:A.layer_result{Model.cache_shape position layer ideal.cache})
  : Lemma (Math.equal_history
      ((FA.logical_before (logical_input call w position layer ideal)).A.kv layer)
      (ideal.cache.kv layer)) =
  Model.full_profile call;
  Model.attention_entry call position layer ideal.cache;
  match ideal.cache.kv layer with
  | None -> ()
  | Some history ->
      let cell (h:P.index 2) (t:P.index history.length) (d:P.index 256)
        : Lemma (
            padded position (Some history) true (h * 524288 + t * 256 + d) == history.keys h t d /\
            padded position (Some history) false (h * 524288 + t * 256 + d) == history.values h t d) = ()
      in C.forall_intro_3 cell

let represented_input (call:Model.invocation) (w:Layout.packed)
  (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (before:State.stage) (ideal:A.layer_result{Model.cache_shape position layer ideal.cache})
  : Lemma
      (requires State.represents position layer before.State.cache before.State.memory /\
        O.cache_approximates before.State.cache ideal.cache /\
        (forall (i:P.index 1024). Approx.v_approximates (S.index before.State.hidden i) (ideal.hidden i)))
      (ensures FA.input_representation (logical_input call w position layer ideal)
        (weights w layer) before.State.hidden (old_keys position before layer) (old_values position before layer)) =
  Model.full_profile call;
  Model.attention_entry call position layer ideal.cache;
  let weight (i:P.index 18352640)
    : Lemma (S.index (weights w layer) i %~ packed_model w layer i) =
    S.lemma_index_slice w (Layout.layer_base layer) (Layout.layer_base layer + 18352640) i;
    Approx.to_real_ok (S.index w (Layout.layer_base layer + i))
  in
  C.forall_intro weight;
  if position > 0 then (
    let cell (i:P.index 1048576)
      : Lemma (
          S.index (State.local_keys before.State.memory layer) i %~
            padded position (ideal.cache.kv layer) true i /\
          S.index (State.local_values before.State.memory layer) i %~
            padded position (ideal.cache.kv layer) false i) =
      let h:P.index 2 = i / 524288 in
      let t:P.index 2048 = (i % 524288) / 256 in
      let d:P.index 256 = i % 256 in
      let offset = Cache.kv_offset (layer / 4) h t d in
      assert (i == h * 524288 + t * 256 + d);
      assert (S.index (State.local_keys before.State.memory layer) i ==
        S.index before.State.memory.State.keys offset);
      assert (S.index (State.local_values before.State.memory layer) i ==
        S.index before.State.memory.State.values offset);
      assert (Some? (before.State.cache.kv layer) /\ Some? (ideal.cache.kv layer));
      let actual_slot : option (O.kv_history bf16) = before.State.cache.kv layer in
      let ideal_slot : option A.kv_history = ideal.cache.kv layer in
      match actual_slot, ideal_slot with
      | Some actual, Some expected ->
          assert (actual.O.length == position /\ expected.A.length == position);
          if t < position then (
            assert (actual.O.keys h t d == S.index before.State.memory.State.keys offset /\
              actual.O.values h t d == S.index before.State.memory.State.values offset);
            assert (Approx.v_approximates (actual.O.keys h t d) (expected.A.keys h t d) /\
              Approx.v_approximates (actual.O.values h t d) (expected.A.values h t d))
          )
          else (
            Approx.a0 #bf16;
            assert (S.index before.State.memory.State.keys offset == (zero #bf16) /\
              S.index before.State.memory.State.values offset == (zero #bf16));
            assert (S.index (State.local_keys before.State.memory layer) i == (zero #bf16) /\
              S.index (State.local_values before.State.memory layer) i == (zero #bf16))
          )
      | _ -> ()
    in C.forall_intro cell
  )

(* Only the accepted, source-bound family conclusion is a dependency. Its
   input representation is discharged by represented_input, not assumed. *)
inline_for_extraction noextract
let dependency (call:Model.invocation) (w:Layout.packed)
  (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (before:State.stage) (produced:Native.observation position)
  (ideal:A.layer_result{Model.cache_shape position layer ideal.cache}) : prop =
  let input = logical_input call w position layer ideal in
  FA.input_representation input (weights w layer) before.State.hidden
    (old_keys position before layer) (old_values position before layer) ==>
  FA.source_score_post input (old_keys position before layer) (old_values position before layer)
    produced.Native.hidden produced.Native.keys produced.Native.values produced.Native.scores

let output_representation (input:FA.logical_inputs)
  (previous_keys previous_values:FA.bfseq (FA.previous_elements input.call.position))
  (produced:Native.observation input.call.position)
  : Lemma
      (requires FA.semantic_postcondition input previous_keys previous_values
        produced.hidden produced.keys produced.values)
      (ensures (
        let expected = A.full_attention_decoder input.call (FA.weights_view input.packed_weights)
          input.input_hidden (FA.logical_before input) in
        (forall (i:P.index 1024). Approx.v_approximates (S.index produced.hidden i) (expected.hidden i)) /\
        (match expected.cache.kv input.call.layer with
        | None -> False
        | Some history ->
            history.A.length == input.call.position + 1 /\
            (forall (h:P.index 2) (t:nat{t < history.A.length /\ t <= input.call.position}) (d:P.index 256).
              S.index produced.keys (h * 524288 + t * 256 + d) %~ history.A.keys h t d /\
              S.index produced.values (h * 524288 + t * 256 + d) %~ history.A.values h t d)))) = ()

let refines (call:Model.invocation) (w:Layout.packed)
  (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (before:State.stage) (produced:Native.observation position)
  (ideal:A.layer_result{Model.cache_shape position layer ideal.cache})
  : Lemma
      (requires State.represents position layer before.State.cache before.State.memory /\
        O.cache_approximates before.State.cache ideal.cache /\
        (forall (i:P.index 1024). Approx.v_approximates (S.index before.State.hidden i) (ideal.hidden i)) /\
        dependency call w position layer before produced ideal /\
        FA.bilateral_live_scores_finite position produced.scores)
      (ensures
        (forall (i:P.index 1024). Approx.v_approximates (S.index produced.hidden i)
          ((Model.layer_step call (O.model_parameters (Layout.parameter_values w))
            position layer ideal.hidden ideal.cache).A.hidden i)) /\
        O.cache_approximates (State.attention_after position before layer produced).State.cache
          (Model.layer_step call (O.model_parameters (Layout.parameter_values w))
            position layer ideal.hidden ideal.cache).A.cache) =
  Model.full_profile call;
  Model.attention_entry call position layer ideal.cache;
  represented_input call w position layer before ideal;
  let input = logical_input call w position layer ideal in
  assert (FA.semantic_postcondition input (old_keys position before layer) (old_values position before layer)
    produced.hidden produced.keys produced.values);
  output_representation input (old_keys position before layer) (old_values position before layer) produced;
  weight_model_agrees w layer;
  history_agrees call w position layer ideal;
  Math.decoder_history_ext input.call (FA.weights_view input.packed_weights) ideal.hidden
    (FA.logical_before input) ideal.cache
