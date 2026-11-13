module KPlay.SharedSpec

open Pulse.Lib.Core
module S = FStar.Seq
module C = FStar.Classical
module F = Kuiper.Floating.Base
module Scalar = Kuiper.Scalars.Base
module Approx = Kuiper.Approximates.Base
module Permission = PulseCore.FractionalPermission
module BF16 = Kuiper.BFloat16
module F32 = Kuiper.Float32
module E = KPlay.Proof.Shared.Embedding
module A = KPlay.Proof.Shared.Attention
module P = KPlay.Proof.Shared.Projection
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module O = KPlay.Proof.Shared.Observations
module Storage = KPlay.Proof.Shared.Storage

inline_for_extraction noextract
let caller_state (call:Model.invocation) (m h c r k:Type0) (completed:nat) =
  D.decode_state 248320 1024 h (O.parameters h) (O.cache m c r k)
    call.prompt_length completed

inline_for_extraction noextract
noeq type execution (call:Model.invocation) (m h c r k norm score:Type0) = {
  initial_metadata: m;
  observations: D.vec (O.model_observation m h c r k norm score) (Model.calls call);
  selected: D.vec (D.token 248320) (Model.calls call);
  caller: (t:nat{t <= Model.calls call}) -> GTot (caller_state call m h c r k t);
  prepared: (t:nat{t < Model.calls call}) -> GTot (caller_state call m h c r k t);
}

inline_for_extraction noextract
let event (#m #h #c #r #k #norm #score:Type0)
  (observed:O.model_observation m h c r k norm score)
  : D.observation 248320 1024 h (O.cache m c r k) = {
  D.logits = observed.score_keys;
  D.residual = observed.captured;
  D.cache_after = observed.cache_after;
}

inline_for_extraction noextract
let events (#m #h #c #r #k #norm #score:Type0)
  (call:Model.invocation) (run:execution call m h c r k norm score)
  : D.vec (D.observation 248320 1024 h (O.cache m c r k)) (Model.calls call) =
  S.init (Model.calls call) (fun t -> event (S.index run.observations t))

inline_for_extraction noextract
let events_length (#m #h #c #r #k #norm #score:Type0)
  (call:Model.invocation) (run:execution call m h c r k norm score)
  : Lemma (S.length (events call run) == Model.calls call) = ()

inline_for_extraction noextract
let events_at (#m #h #c #r #k #norm #score:Type0)
  (call:Model.invocation) (run:execution call m h c r k norm score)
  (t:nat{t < Model.calls call})
  : Lemma (
      S.index (events call run) t == event (S.index run.observations t)) = ()

inline_for_extraction noextract
let state_at (#m #h #c #r #k #norm #score:Type0)
  (call:Model.invocation) (w:O.parameters h)
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (run:execution call m h c r k norm score) (completed:nat{completed <= Model.calls call})
  : GTot (caller_state call m h c r k completed) =
  events_length call run;
  let observed = events call run in
  D.state_at #248320 #1024 #(call.prompt_length)
    prompt w (O.empty_cache run.initial_metadata) observed run.selected completed

inline_for_extraction noextract
let state_at_input (#m #h #c #r #k #norm #score:Type0)
  (call:Model.invocation) (w:O.parameters h)
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (run:execution call m h c r k norm score) (t:nat{t < Model.calls call})
  : Lemma (
      (state_at call w prompt run t).next_token ==
      Model.consumed call prompt run.selected t) = ()

inline_for_extraction noextract
let state_at_successor (#m #h #c #r #k #norm #score:Type0)
  (call:Model.invocation) (w:O.parameters h)
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (run:execution call m h c r k norm score) (t:nat{t < Model.calls call})
  : Lemma (
      state_at call w prompt run (t + 1) ==
      D.advance prompt (D.prepare_call (state_at call w prompt run t))
        (event (S.index run.observations t)) (S.index run.selected t)) =
  events_length call run;
  events_at call run t;
  D.state_at_successor #248320 #1024 #(call.prompt_length)
    prompt w (O.empty_cache run.initial_metadata) (events call run) run.selected t

(* This is the common concrete step, not a caller-supplied model predicate. *)
inline_for_extraction noextract
let structural_step (#m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar c, Scalar.scalar r, F.floating score |}
  (call:Model.invocation) (w:O.parameters h)
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (run:execution call m h c r k norm score) (t:nat{t < Model.calls call})
  : GTot prop =
  let observed = S.index run.observations t in
  let before = run.caller t in
  O.structure w t before.next_token before.model_cache observed /\
  D.maximum observed.score_keys (S.index run.selected t) /\
  run.prepared t == D.prepare_call before /\
  run.caller (t + 1) ==
    D.advance prompt (run.prepared t) (event observed) (S.index run.selected t)

inline_for_extraction noextract
let safety_prefix (#m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar c, Scalar.scalar r, F.floating score |}
  (call:Model.invocation) (w:O.parameters h)
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (run:execution call m h c r k norm score) (completed:nat{completed <= Model.calls call})
  : GTot prop =
  run.caller 0 == state_at call w prompt run 0 /\
  (forall (t:nat{t < completed}). structural_step call w prompt run t)

inline_for_extraction noextract
let numerical_step (#m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar h, Approx.real_like h,
     Scalar.scalar c, Approx.real_like c,
     Scalar.scalar r, Approx.real_like r,
     Scalar.scalar k, Approx.real_like k,
     Scalar.scalar norm, Approx.real_like norm,
     F.floating score, Approx.real_like score |}
  (call:Model.invocation) (w:O.parameters h)
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (run:execution call m h c r k norm score) (t:nat{t < Model.calls call})
  : GTot prop =
  let ideal_weights = O.model_parameters w in
  O.numerical call ideal_weights t
    (Model.consumed call prompt run.selected t)
    (Model.cache_at call ideal_weights prompt run.selected t)
    (S.index run.observations t)

inline_for_extraction noextract
let numerical_prefix (#m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar h, Approx.real_like h,
     Scalar.scalar c, Approx.real_like c,
     Scalar.scalar r, Approx.real_like r,
     Scalar.scalar k, Approx.real_like k,
     Scalar.scalar norm, Approx.real_like norm,
     F.floating score, Approx.real_like score |}
  (call:Model.invocation) (w:O.parameters h)
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (run:execution call m h c r k norm score) (completed:nat{completed <= Model.calls call})
  : GTot prop =
  forall (t:nat{t < completed}). numerical_step call w prompt run t

inline_for_extraction noextract
let rec safety_prefix_state (#m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar c, Scalar.scalar r, F.floating score |}
  (call:Model.invocation) (w:O.parameters h)
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (run:execution call m h c r k norm score) (completed:nat{completed <= Model.calls call})
  : Lemma
      (requires safety_prefix call w prompt run completed)
      (ensures run.caller completed == state_at call w prompt run completed)
      (decreases completed) =
  if completed > 0 then (
    safety_prefix_state call w prompt run (completed - 1);
    assert (structural_step call w prompt run (completed - 1));
    state_at_successor call w prompt run (completed - 1);
    assert (run.prepared (completed - 1) ==
      D.prepare_call (state_at call w prompt run (completed - 1)));
    assert (run.caller completed ==
      D.advance prompt (run.prepared (completed - 1))
        (event (S.index run.observations (completed - 1)))
        (S.index run.selected (completed - 1)))
  )

inline_for_extraction noextract
let safety_invariant (#m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar c, Scalar.scalar r, F.floating score |}
  (call:Model.invocation) (w:O.parameters h)
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (run:execution call m h c r k norm score) (completed:nat{completed <= Model.calls call})
  : GTot prop =
  safety_prefix call w prompt run completed /\
  run.caller completed == state_at call w prompt run completed /\
  O.cache_shape completed 0 (run.caller completed).model_cache /\
  (run.caller completed).model_cache.metadata == run.initial_metadata

inline_for_extraction noextract
let rec safety_invariant_from_steps (#m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar c, Scalar.scalar r, F.floating score |}
  (call:Model.invocation) (w:O.parameters h)
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (run:execution call m h c r k norm score) (completed:nat{completed <= Model.calls call})
  : Lemma
      (requires safety_prefix call w prompt run completed)
      (ensures safety_invariant call w prompt run completed)
      (decreases completed) =
  safety_prefix_state call w prompt run completed;
  if completed > 0 then (
    safety_invariant_from_steps call w prompt run (completed - 1);
    let observed = S.index run.observations (completed - 1) in
    let before = run.caller (completed - 1) in
    assert (structural_step call w prompt run (completed - 1));
    assert (O.structure w (completed - 1) before.next_token before.model_cache observed);
    O.structure_boundary w (completed - 1) before.next_token before.model_cache observed;
    assert ((run.caller completed).model_cache == observed.cache_after)
  )

inline_for_extraction noextract
let numerical_invariant (#m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar h, Approx.real_like h,
     Scalar.scalar c, Approx.real_like c,
     Scalar.scalar r, Approx.real_like r,
     Scalar.scalar k, Approx.real_like k,
     Scalar.scalar norm, Approx.real_like norm,
     F.floating score, Approx.real_like score |}
  (call:Model.invocation) (w:O.parameters h)
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (run:execution call m h c r k norm score) (completed:nat{completed <= Model.calls call})
  : GTot prop =
  safety_invariant call w prompt run completed /\
  numerical_prefix call w prompt run completed /\
  O.cache_approximates (run.caller completed).model_cache
    (Model.cache_at call (O.model_parameters w) prompt run.selected completed)

inline_for_extraction noextract
let invariant_from_steps (#m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar h, Approx.real_like h,
     Scalar.scalar c, Approx.real_like c,
     Scalar.scalar r, Approx.real_like r,
     Scalar.scalar k, Approx.real_like k,
     Scalar.scalar norm, Approx.real_like norm,
     F.floating score, Approx.real_like score |}
  (call:Model.invocation) (w:O.parameters h)
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (run:execution call m h c r k norm score) (completed:nat{completed <= Model.calls call})
  : Lemma
      (requires safety_prefix call w prompt run completed /\
                numerical_prefix call w prompt run completed)
      (ensures numerical_invariant call w prompt run completed) =
  safety_invariant_from_steps call w prompt run completed;
  if completed > 0 then (
    let t = completed - 1 in
    let observed = S.index run.observations t in
    let ideal_weights = O.model_parameters w in
    let ideal_cache = Model.cache_at call ideal_weights prompt run.selected t in
    let token = Model.consumed call prompt run.selected t in
    assert (numerical_step call w prompt run t);
    assert (structural_step call w prompt run t);
    assert (O.numerical call ideal_weights t token ideal_cache observed);
    assert (observed.cache_after == (observed.stages 24).cache);
    O.numerical_cache_boundary call ideal_weights t token ideal_cache observed;
    Model.cache_at_successor call ideal_weights prompt run.selected t;
    assert ((run.caller completed).model_cache == observed.cache_after)
  )

inline_for_extraction noextract
let invariant_successor (#m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar h, Approx.real_like h,
     Scalar.scalar c, Approx.real_like c,
     Scalar.scalar r, Approx.real_like r,
     Scalar.scalar k, Approx.real_like k,
     Scalar.scalar norm, Approx.real_like norm,
     F.floating score, Approx.real_like score |}
  (call:Model.invocation) (w:O.parameters h)
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (run:execution call m h c r k norm score) (t:nat{t < Model.calls call})
  : Lemma
      (requires numerical_invariant call w prompt run t /\
                structural_step call w prompt run t /\
                numerical_step call w prompt run t)
      (ensures numerical_invariant call w prompt run (t + 1)) =
  invariant_from_steps call w prompt run (t + 1)

(* Always required, including outside the authorized attention numerical scope. *)
inline_for_extraction noextract
let safety_postcondition (#m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar c, Scalar.scalar r, F.floating score |}
  (call:Model.invocation) (decode_metadata:m -> GTot O.cache_configuration)
  (w:O.parameters h) (prompt:D.vec (D.token 248320) call.prompt_length)
  (run:execution call m h c r k norm score)
  (output:D.histories 248320 1024 h call.decode_steps)
  (final_state:caller_state call m h c r k (Model.calls call))
  : GTot prop =
  O.configuration_ok (decode_metadata run.initial_metadata) /\
  safety_prefix call w prompt run (Model.calls call) /\
  output == (run.caller (Model.calls call)).returned /\
  final_state == run.caller (Model.calls call)

(* No numerical-domain Boolean, arbitrary transition, or assumed score vector
   is an input. Role numerical theorems establish this fixed relation under the
   source-bound bilateral attention premises recorded in M.json. *)
inline_for_extraction noextract
let postcondition (#m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar h, Approx.real_like h,
     Scalar.scalar c, Approx.real_like c,
     Scalar.scalar r, Approx.real_like r,
     Scalar.scalar k, Approx.real_like k,
     Scalar.scalar norm, Approx.real_like norm,
     F.floating score, Approx.real_like score |}
  (call:Model.invocation) (decode_metadata:m -> GTot O.cache_configuration)
  (w:O.parameters h) (prompt:D.vec (D.token 248320) call.prompt_length)
  (run:execution call m h c r k norm score)
  (output:D.histories 248320 1024 h call.decode_steps)
  (final_state:caller_state call m h c r k (Model.calls call))
  : GTot prop =
  safety_postcondition call decode_metadata w prompt run output final_state /\
  numerical_prefix call w prompt run (Model.calls call)

inline_for_extraction noextract
let complete_histories (#m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar c, Scalar.scalar r, F.floating score |}
  (call:Model.invocation) (decode_metadata:m -> GTot O.cache_configuration)
  (w:O.parameters h) (prompt:D.vec (D.token 248320) call.prompt_length)
  (run:execution call m h c r k norm score)
  (output:D.histories 248320 1024 h call.decode_steps)
  (final_state:caller_state call m h c r k (Model.calls call))
  : Lemma
      (requires safety_postcondition call decode_metadata w prompt run output final_state)
      (ensures
        final_state.parameters == w /\
        final_state.position == Model.calls call /\
        final_state.next_token == S.index run.selected (Model.calls call - 1) /\
        O.cache_shape (Model.calls call) 0 final_state.model_cache /\
        final_state.model_cache.metadata == run.initial_metadata /\
        (forall (g:P.index call.decode_steps).
          let t = call.prompt_length - 1 + g in
          let observed = S.index run.observations t in
          S.index output.generated_tokens g == S.index run.selected t /\
          (forall (i:P.index 1024).
            S.index (S.index output.hidden_history g) i ==
              (observed.stages 24).hidden i) /\
          O.native_maximum observed.produced_scores (S.index output.generated_tokens g))) =
  safety_invariant_from_steps call w prompt run (Model.calls call);
  events_length call run;
  D.decode_fields #248320 #1024 #(call.prompt_length) #(call.decode_steps)
    prompt w (O.empty_cache run.initial_metadata)
    (events call run) run.selected;
  let each (g:P.index call.decode_steps)
    : Lemma (
        let t = call.prompt_length - 1 + g in
        let observed = S.index run.observations t in
        S.index output.generated_tokens g == S.index run.selected t /\
        (forall (i:P.index 1024).
          S.index (S.index output.hidden_history g) i ==
            (observed.stages 24).hidden i) /\
        O.native_maximum observed.produced_scores (S.index output.generated_tokens g)) =
    let t = call.prompt_length - 1 + g in
    let observed = S.index run.observations t in
    let before = run.caller t in
    assert (structural_step call w prompt run t);
    assert (O.structure w t before.next_token before.model_cache observed);
    events_at call run t;
    O.maximum_is_same_actual_scores observed.produced_scores observed.score_keys
      (S.index run.selected t)
  in
  C.forall_intro each

inline_for_extraction noextract
let generated_token_feedback (#m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar c, Scalar.scalar r, F.floating score |}
  (call:Model.invocation) (w:O.parameters h)
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (run:execution call m h c r k norm score)
  (t:nat{call.prompt_length <= t /\ t < Model.calls call})
  : Lemma
      (requires safety_prefix call w prompt run (t + 1))
      (ensures
        (S.index run.observations t).consumed == S.index run.selected (t - 1) /\
        (forall (i:P.index 1024).
          ((S.index run.observations t).stages 0).hidden i ==
            S.index (E.embedding w.embedding (S.index run.selected (t - 1))) i)) =
  safety_prefix_state call w prompt run t;
  state_at_input call w prompt run t;
  assert (structural_step call w prompt run t);
  let before = run.caller t in
  let observed = S.index run.observations t in
  assert (O.structure w t before.next_token before.model_cache observed);
  assert (before.next_token == S.index run.selected (t - 1))

inline_for_extraction noextract
let complete_numerical_result (#m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar h, Approx.real_like h,
     Scalar.scalar c, Approx.real_like c,
     Scalar.scalar r, Approx.real_like r,
     Scalar.scalar k, Approx.real_like k,
     Scalar.scalar norm, Approx.real_like norm,
     F.floating score, Approx.real_like score |}
  (call:Model.invocation) (decode_metadata:m -> GTot O.cache_configuration)
  (w:O.parameters h) (prompt:D.vec (D.token 248320) call.prompt_length)
  (run:execution call m h c r k norm score)
  (output:D.histories 248320 1024 h call.decode_steps)
  (final_state:caller_state call m h c r k (Model.calls call))
  : Lemma
      (requires postcondition call decode_metadata w prompt run output final_state)
      (ensures
        O.cache_approximates final_state.model_cache
          (Model.cache_at call (O.model_parameters w) prompt run.selected (Model.calls call)) /\
        (forall (g:P.index call.decode_steps).
          let t = call.prompt_length - 1 + g in
          let expected = Model.computation_at call (O.model_parameters w) prompt run.selected t in
          (forall (i:P.index 1024).
            Approx.v_approximates (S.index (S.index output.hidden_history g) i)
              (expected.residual i)) /\
          (forall (v:(D.token 248320)).
            Approx.v_approximates ((S.index run.observations t).produced_scores v)
              (expected.logits v)))) =
  invariant_from_steps call w prompt run (Model.calls call);
  complete_histories call decode_metadata w prompt run output final_state;
  let each (g:P.index call.decode_steps)
    : Lemma (
        let t = call.prompt_length - 1 + g in
        let expected = Model.computation_at call (O.model_parameters w) prompt run.selected t in
        (forall (i:P.index 1024).
          Approx.v_approximates (S.index (S.index output.hidden_history g) i)
            (expected.residual i)) /\
        (forall (v:(D.token 248320)).
          Approx.v_approximates ((S.index run.observations t).produced_scores v)
            (expected.logits v))) =
    let t = call.prompt_length - 1 + g in
    let ideal_weights = O.model_parameters w in
    let ideal_cache = Model.cache_at call ideal_weights prompt run.selected t in
    let token = Model.consumed call prompt run.selected t in
    let observed = S.index run.observations t in
    assert (numerical_step call w prompt run t);
    assert (O.numerical call ideal_weights t token ideal_cache observed);
    O.numerical_residual_boundary call ideal_weights t token ideal_cache observed
  in
  C.forall_intro each

(* Spatial obligations are not premises disguised as pure correctness flags.
   These are actual library points-to resources at the caller's native handles. *)
inline_for_extraction noextract
let safety_loop_invariant (#token_cell #m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar c, Scalar.scalar r, F.floating score |}
  (call:Model.invocation) (decode_token:token_cell -> GTot int)
  (permission:Permission.perm)
  (w:O.parameters h) (weights_at:O.parameters (Storage.cell h))
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (prompt_at:Storage.token_storage token_cell call.prompt_length)
  (run:execution call m h c r k norm score) (completed:nat{completed <= Model.calls call})
  (cache_at:Storage.cache_storage (run.caller completed).model_cache)
  (history_at:Storage.history_storage token_cell h
    (D.retained_count call.prompt_length completed))
  : slprop =
  Storage.loop_owns call decode_token permission w weights_at prompt prompt_at completed
    (run.caller completed).model_cache cache_at (run.caller completed).returned history_at **
  pure (safety_invariant call w prompt run completed)

inline_for_extraction noextract
let caller_precondition (#token_cell #h:Type0)
  (call:Model.invocation) (decode_token:token_cell -> GTot int)
  (permission:Permission.perm)
  (w:O.parameters h) (weights_at:O.parameters (Storage.cell h))
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (prompt_at:Storage.token_storage token_cell call.prompt_length)
  : slprop =
  Storage.inputs_owns call decode_token permission w weights_at prompt prompt_at

inline_for_extraction noextract
let safety_caller_postcondition (#token_cell #m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar c, Scalar.scalar r, F.floating score |}
  (call:Model.invocation) (decode_token:token_cell -> GTot int)
  (decode_metadata:m -> GTot O.cache_configuration) (permission:Permission.perm)
  (w:O.parameters h) (weights_at:O.parameters (Storage.cell h))
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (prompt_at:Storage.token_storage token_cell call.prompt_length)
  (run:execution call m h c r k norm score)
  (output:D.histories 248320 1024 h call.decode_steps)
  (output_at:Storage.history_storage token_cell h call.decode_steps)
  (final_state:caller_state call m h c r k (Model.calls call))
  : slprop =
  caller_precondition call decode_token permission w weights_at prompt prompt_at **
  Storage.history_owns decode_token output_at output **
  pure (safety_postcondition call decode_metadata w prompt run output final_state)

inline_for_extraction noextract
let caller_postcondition (#token_cell #m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar h, Approx.real_like h,
     Scalar.scalar c, Approx.real_like c,
     Scalar.scalar r, Approx.real_like r,
     Scalar.scalar k, Approx.real_like k,
     Scalar.scalar norm, Approx.real_like norm,
     F.floating score, Approx.real_like score |}
  (call:Model.invocation) (decode_token:token_cell -> GTot int)
  (decode_metadata:m -> GTot O.cache_configuration) (permission:Permission.perm)
  (w:O.parameters h) (weights_at:O.parameters (Storage.cell h))
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (prompt_at:Storage.token_storage token_cell call.prompt_length)
  (run:execution call m h c r k norm score)
  (output:D.histories 248320 1024 h call.decode_steps)
  (output_at:Storage.history_storage token_cell h call.decode_steps)
  (final_state:caller_state call m h c r k (Model.calls call))
  : slprop =
  safety_caller_postcondition call decode_token decode_metadata permission
    w weights_at prompt prompt_at run output output_at final_state **
  pure (numerical_prefix call w prompt run (Model.calls call))

inline_for_extraction noextract
let mixed_precision_postcondition (#m #score:Type0)
  {| F.floating score, Approx.real_like score |}
  (call:Model.invocation) (decode_metadata:m -> GTot O.cache_configuration)
  (w:O.parameters BF16.t) (prompt:D.vec (D.token 248320) call.prompt_length)
  (run:execution call m BF16.t BF16.t F32.t BF16.t BF16.t score)
  (output:D.histories 248320 1024 BF16.t call.decode_steps)
  (final_state:caller_state call m BF16.t BF16.t F32.t BF16.t (Model.calls call))
  : GTot prop =
  postcondition call decode_metadata w prompt run output final_state
