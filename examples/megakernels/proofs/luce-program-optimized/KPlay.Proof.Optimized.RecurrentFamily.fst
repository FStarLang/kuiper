module KPlay.Proof.Optimized.RecurrentFamily

#set-options "--fuel 3 --ifuel 1 --z3rlimit 80"

open Kuiper
module S = FStar.Seq
module C = FStar.Classical
module Approx = Kuiper.Approximates.Base
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module R = KPlay.Proof.Shared.Recurrent
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module FR = KPlay.Proof.Optimized.FamilyRecurrent
module Layout = KPlay.Proof.Optimized.Layout
module State = KPlay.Proof.Optimized.State
module Native = KPlay.Proof.Optimized.Recurrent

inline_for_extraction noextract
let weights (w:Layout.packed) (layer:Model.linear_layer) : lseq bf16 21555360 =
  Layout.layer_partition layer;
  S.slice w (Layout.layer_base layer) (Layout.layer_base layer + 21555360)

inline_for_extraction noextract
let old_recurrent (position:nat) (before:State.stage) (layer:Model.linear_layer)
  : lseq f32 (FR.previous_elements position 262144) =
  if position = 0 then S.empty else State.local_recurrent before.State.memory layer

inline_for_extraction noextract
let old_convolution (position:nat) (before:State.stage) (layer:Model.linear_layer)
  : lseq f32 (FR.previous_elements position 24576) =
  if position = 0 then S.empty else State.local_convolution before.State.memory layer

(* The exact accepted postcondition at reduced.cu's l%4<3 source block. *)
inline_for_extraction noextract
let dependency (w:Layout.packed) (position:nat{position < 2048})
  (layer:Model.linear_layer) (before:State.stage) (produced:Native.observation) : prop =
  FR.semantic_postcondition (weights w layer) before.State.hidden position
    (old_recurrent position before layer) (old_convolution position before layer)
    produced.Native.hidden produced.Native.recurrent produced.Native.convolution

let weight_representation (w:Layout.packed) (layer:Model.linear_layer)
  : Lemma (FR.weights_representation (weights w layer)
      ((O.model_parameters (Layout.parameter_values w)).Model.linear layer)) =
  let cell (i:nat{i < 21555360})
    : Lemma (S.index (weights w layer) i %~
        Approx.to_real (S.index w (Layout.layer_base layer + i))) =
    S.lemma_index_slice w (Layout.layer_base layer) (Layout.layer_base layer + 21555360) i;
    Approx.to_real_ok (S.index w (Layout.layer_base layer + i))
  in
  C.forall_intro cell;
  FStar.Pervasives.norm_spec
    [delta_only [`%FR.weights_representation; `%FR.weight_vector_representation;
      `%FR.weight_matrix_representation; `%FR.row_major;
      `%O.model_parameters; `%O.model_linear; `%O.model_vector; `%O.model_matrix;
      `%Layout.parameter_values; `%Layout.parameters; `%Layout.linear_weights]; zeta; iota]
    (FR.weights_representation (weights w layer)
      ((O.model_parameters (Layout.parameter_values w)).Model.linear layer))

let represented_input (call:Model.invocation) (w:Layout.packed)
  (position:nat{position < Model.calls call}) (layer:Model.linear_layer)
  (before:State.stage) (ideal:A.layer_result{Model.cache_shape position layer ideal.cache})
  : Lemma
      (requires State.represents position layer before.State.cache before.State.memory /\
        O.cache_approximates before.State.cache ideal.cache /\
        (forall (i:P.index 1024). Approx.v_approximates (S.index before.State.hidden i) (ideal.hidden i)))
      (ensures FR.input_representation
        (Model.recurrent_call call position layer)
        ((O.model_parameters (Layout.parameter_values w)).Model.linear layer)
        (fun i -> ideal.hidden i) (Model.recurrent_cache ideal.cache)
        (weights w layer) before.State.hidden position
        (old_recurrent position before layer) (old_convolution position before layer)) =
  Model.full_profile call;
  Model.recurrent_entry call position layer ideal.cache;
  weight_representation w layer;
  if position > 0 then (
    let conv (channel:P.index 6144) (tap:P.index 4)
      : Lemma (
          S.index (State.local_convolution before.State.memory layer) (channel * 4 + tap) %~
          FR.old_convolution (Model.recurrent_call call position layer)
            (Model.recurrent_cache ideal.cache) channel tap) =
      assert (Some? (before.State.cache.convolution layer) /\ Some? (ideal.cache.convolution layer));
      let actual_slot : option (O.matrix f32 6144 4) = before.State.cache.convolution layer in
      let ideal_slot : option A.convolution_state = ideal.cache.convolution layer in
      match actual_slot with
      | None -> ()
      | Some actual ->
          (match ideal_slot with
          | None -> ()
          | Some expected ->
              assert (Approx.v_approximates (actual channel tap) (expected channel tap)))
    in
    C.forall_intro_2 conv;
    let rec_cell (head:P.index 16) (key value:P.index 128)
      : Lemma (
          S.index (State.local_recurrent before.State.memory layer)
            (head * 16384 + value * 128 + key) %~
          FR.old_recurrent (Model.recurrent_call call position layer)
            (Model.recurrent_cache ideal.cache) head key value) =
      assert (Some? (before.State.cache.recurrent layer) /\ Some? (ideal.cache.recurrent layer));
      let actual_slot : option (O.tensor f32 16 128 128) = before.State.cache.recurrent layer in
      let ideal_slot : option A.recurrent_state = ideal.cache.recurrent layer in
      match actual_slot with
      | None -> ()
      | Some actual ->
          (match ideal_slot with
          | None -> ()
          | Some expected ->
              assert (Approx.v_approximates (actual head key value) (expected head key value)))
    in C.forall_intro_3 rec_cell
  )

let post_to_expected (call:R.invocation) (w:R.weights) (hidden:R.vector 1024)
  (before:R.model_cache{R.cache_domain call before})
  (produced:Native.observation)
  : Lemma
      (requires FR.call_postcondition call w hidden before
        produced.hidden produced.recurrent produced.convolution)
      (ensures FR.output_representation produced.hidden produced.recurrent produced.convolution
        (R.decoder_values call w hidden before)) = ()

let cache_representation (before:State.stage) (layer:Model.linear_layer)
  (produced:Native.observation) (ideal:A.hybrid_cache) (computed:R.layer_values)
  : Lemma
      (requires O.cache_approximates before.State.cache ideal /\
        FR.output_representation produced.hidden produced.recurrent produced.convolution computed)
      (ensures O.cache_approximates (State.linear_after before layer produced).State.cache
        (Model.install_recurrent ideal layer computed)) =
  FStar.Pervasives.norm_spec [delta_only [
    `%FR.output_representation; `%FR.convolution_representation; `%FR.recurrent_representation;
    `%FR.convolution_index; `%FR.recurrent_index]; zeta; iota]
    (FR.output_representation produced.hidden produced.recurrent produced.convolution computed);
  let after = (State.linear_after before layer produced).State.cache in
  let expected = Model.install_recurrent ideal layer computed in
  let convolution_cell (l:A.layer_index)
    : Lemma (
        match after.convolution l with
        | None -> expected.convolution l == None
        | Some actual ->
            (match expected.convolution l with
            | None -> False
            | Some target ->
                forall (c:P.index 6144) (t:P.index 4).
                  Approx.v_approximates (actual c t) (target c t))) =
    if l <> layer then
      assert (after.convolution l == before.State.cache.convolution l /\
        expected.convolution l == ideal.convolution l)
  in
  C.forall_intro convolution_cell;
  let recurrent_cell (l:A.layer_index)
    : Lemma (
        match after.recurrent l with
        | None -> expected.recurrent l == None
        | Some actual ->
            (match expected.recurrent l with
            | None -> False
            | Some target ->
                forall (h:P.index 16) (key value:P.index 128).
                  Approx.v_approximates (actual h key value) (target h key value))) =
    if l <> layer then
      assert (after.recurrent l == before.State.cache.recurrent l /\
        expected.recurrent l == ideal.recurrent l)
  in
  C.forall_intro recurrent_cell;
  assert (after.kv == before.State.cache.kv /\ expected.kv == ideal.kv)

#push-options "--fuel 0 --ifuel 0 --z3rlimit 80"
let refines (call:Model.invocation) (w:Layout.packed)
  (position:nat{position < Model.calls call}) (layer:Model.linear_layer)
  (before:State.stage) (produced:Native.observation)
  (ideal:A.layer_result{Model.cache_shape position layer ideal.cache})
  : Lemma
      (requires State.represents position layer before.State.cache before.State.memory /\
        O.cache_approximates before.State.cache ideal.cache /\
        (forall (i:P.index 1024). Approx.v_approximates (S.index before.State.hidden i) (ideal.hidden i)) /\
        dependency w position layer before produced)
      (ensures
        (forall (i:P.index 1024). Approx.v_approximates (S.index produced.hidden i)
          ((Model.layer_step call (O.model_parameters (Layout.parameter_values w))
            position layer ideal.hidden ideal.cache).A.hidden i)) /\
        O.cache_approximates (State.linear_after before layer produced).State.cache
          (Model.layer_step call (O.model_parameters (Layout.parameter_values w))
            position layer ideal.hidden ideal.cache).A.cache) =
  Model.full_profile call;
  Model.recurrent_entry call position layer ideal.cache;
  represented_input call w position layer before ideal;
  FStar.Pervasives.norm_spec [delta_only [`%dependency; `%FR.semantic_postcondition]; zeta; iota]
    (dependency w position layer before produced);
  let rc = Model.recurrent_call call position layer in
  let rw = (O.model_parameters (Layout.parameter_values w)).Model.linear layer in
  let prior = Model.recurrent_cache ideal.cache in
  assert (FR.call_postcondition rc rw (fun i -> ideal.hidden i) prior
    produced.hidden produced.recurrent produced.convolution);
  post_to_expected rc rw (fun i -> ideal.hidden i) prior produced;
  let computed = R.decoder_values rc rw (fun i -> ideal.hidden i) prior in
  FStar.Pervasives.norm_spec [delta_only [
    `%FR.output_representation; `%FR.bf16_vector_representation;
    `%FR.convolution_representation; `%FR.recurrent_representation;
    `%FR.convolution_index; `%FR.recurrent_index]; zeta; iota]
    (FR.output_representation produced.hidden produced.recurrent produced.convolution computed);
  let expected = Model.layer_step call (O.model_parameters (Layout.parameter_values w))
    position layer ideal.hidden ideal.cache in
  FStar.Pervasives.norm_spec [delta_only [`%Model.layer_step]; zeta; iota]
    (Model.layer_step call (O.model_parameters (Layout.parameter_values w))
      position layer ideal.hidden ideal.cache);
  assert (expected.cache == Model.install_recurrent ideal.cache layer computed);
  cache_representation before layer produced ideal.cache computed;
  let hidden_cell (i:P.index 1024)
    : Lemma (Approx.v_approximates (S.index produced.hidden i) (expected.hidden i)) =
    assert (expected.hidden i == computed.hidden_value i);
    assert (Approx.v_approximates (S.index produced.hidden i) (computed.hidden_value i))
  in
  C.forall_intro hidden_cell
#pop-options
