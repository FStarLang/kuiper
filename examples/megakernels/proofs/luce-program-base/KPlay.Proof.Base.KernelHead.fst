module KPlay.Proof.Base.KernelHead

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open FStar.Real

module Seq = FStar.Seq
module BF = Kuiper.BFloat16
module I64 = FStar.Int64
module SZ = Kuiper.SizeT
module Approx = Kuiper.Approximates.Base
module P = KPlay.Proof.Shared.Projection
module D = KPlay.Proof.Shared.Selection
module O = KPlay.Proof.Shared.Observations
module Attention = KPlay.Proof.Shared.Attention
module Model = KPlay.Proof.Shared.Model
module K = KPlay.Proof.Base.KernelState
module Trace = KPlay.Proof.Base.Trace
module Layers = KPlay.Proof.Base.LayerTrace
module Transfer = KPlay.Proof.Base.Transfer
module Inputs = KPlay.Proof.Base.Inputs
module Native = KPlay.Proof.Base.NativeProjection
module Hidden = KPlay.Proof.Base.WarpHidden
module Cache = KPlay.Proof.Base.CacheNative
module History = KPlay.Proof.Base.HistoryNative
module RMS = KPlay.Proof.Base.RMS
module Config = KPlay.Proof.Base.RMSConfigs
module Scratch = KPlay.Proof.Base.ScratchMath
module Linear = KPlay.Proof.Base.LinearBook
module Vocabulary = KPlay.Proof.Base.Vocabulary
module Order = KPlay.Proof.Base.Order
module Scores = KPlay.Proof.Base.ScoreObservation
module Algebra = KPlay.Proof.Base.Algebra
module Norm = KPlay.Proof.Base.Norm

inline_for_extraction noextract
let geometry ()
  : Lemma (
      SZ.v (Config.model_norm 24 false).RMS.width == 1024 /\
      SZ.v (Config.model_norm 24 false).RMS.rows == 1 /\
      SZ.v (Config.model_norm 24 false).RMS.source == 0 /\
      SZ.v (Config.model_norm 24 false).RMS.destination == 1024 /\
      SZ.v (Config.model_norm 24 false).RMS.weight_offset == 254279680) = ()

inline_for_extraction noextract
let normalized (weights:Transfer.packed) (before:Transfer.scratch) : GTot Transfer.scratch =
  RMS.after (Config.model_norm 24 false) before weights 1

inline_for_extraction noextract
let capture (call:Model.invocation) (position:nat{position < Model.calls call})
  (memory:K.memory call) (before:Transfer.scratch) : GTot (K.memory call) =
  {memory with
    K.hidden = (if position < call.Model.prompt_length - 1 then memory.K.hidden
      else History.hidden_writes call memory.K.hidden before (History.retained_row call position) 1024);
    K.best = Vocabulary.negative_infinity;
    K.next = 0L}

inline_for_extraction noextract
let start (call:Model.invocation) (weights:Transfer.packed) (m:K.metadata call)
  (c:K.current call) (before:Transfer.scratch) : GTot (K.metadata call) =
  {m with K.memory = capture call c.K.position m.K.memory before;
    K.phase = K.Normalized c (normalized weights before)}

inline_for_extraction noextract
let same_history (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (old fresh:K.metadata call)
  : Lemma
      (requires old.K.completed == fresh.K.completed /\ old.K.history == fresh.K.history)
      (ensures
        K.caller call weights prompt old == K.caller call weights prompt fresh /\
        K.reference call weights prompt old == K.reference call weights prompt fresh) = ()

inline_for_extraction noextract
let capture_previous (call:Model.invocation) (position:nat{position < Model.calls call})
  (memory:K.memory call) (before:Transfer.scratch)
  (history:D.histories 248320 1024 BF.t (D.retained_count call.Model.prompt_length position))
  : Lemma
      (requires K.history_cells call (D.retained_count call.Model.prompt_length position) memory history)
      (ensures K.history_cells call (D.retained_count call.Model.prompt_length position)
        (capture call position memory before) history) =
  let cell (row:P.index (D.retained_count call.Model.prompt_length position)) (i:P.index 1024)
    : Lemma (Seq.index (capture call position memory before).K.hidden (row * 1024 + i) ==
        Seq.index memory.K.hidden (row * 1024 + i)) =
    if position >= call.Model.prompt_length - 1 then
      History.capture_frame call memory.K.hidden before (History.retained_row call position) (row * 1024 + i)
  in
  FStar.Classical.forall_intro_2 cell

inline_for_extraction noextract
let norm_approx (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (m:K.metadata call)
  (c:K.current call{c.K.layer == 24}) (h:Hidden.snapshot) (kv:Cache.snapshot)
  : Lemma
      (requires K.current_valid call weights prompt m c h kv)
      (ensures K.normalized_input call weights prompt m c (normalized weights (Native.hidden_values h))) =
  let input : P.vector FStar.Real.real 1024 = fun i ->
    (Model.layers_at call (O.model_parameters (Inputs.parameter_values weights))
      c.K.position c.K.token (K.reference call weights prompt m) 24).Attention.hidden i in
  let each_weight (i:P.index 1024)
    : Lemma (Approx.v_approximates (Seq.index weights (254279680 + i))
        ((O.model_parameters (Inputs.parameter_values weights)).Model.final_norm i)) =
    Approx.to_real_ok (Seq.index weights (254279680 + i))
  in
  FStar.Classical.forall_intro each_weight;
  geometry ();
  Scratch.rms_approx True (Config.model_norm 24 false) weights (Native.hidden_values h) 0
    input
    (O.model_parameters (Inputs.parameter_values weights)).Model.final_norm;
  let result = P.normalized_for_logits #1024 P.reference_epsilon
    input
    (O.model_parameters (Inputs.parameter_values weights)).Model.final_norm in
  assert (Scratch.represents (normalized weights (Native.hidden_values h)) 1024 result);
  let cell (i:P.index 1024)
    : Lemma (Approx.v_approximates
        (Seq.index (normalized weights (Native.hidden_values h)) (1024 + i)) (result i)) = ()
  in
  FStar.Classical.forall_intro cell;
  norm_spec [delta_once ["KPlay.Proof.Base.KernelState.ideal_normalized"]]
    (K.ideal_normalized call (O.model_parameters (Inputs.parameter_values weights))
      c.K.position c.K.token (K.reference call weights prompt m));
  norm_spec [delta_once ["KPlay.Proof.Base.KernelState.normalized_input"]]
    (K.normalized_input call weights prompt m c (normalized weights (Native.hidden_values h)))

inline_for_extraction noextract
let start_valid (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call})
  (m:K.metadata call) (h:Hidden.snapshot) (kv:Cache.snapshot)
  : Lemma
      (requires K.layer call weights prompt position 24 m h kv)
      (ensures
        K.Decoding? m.K.phase /\
        K.head_ready call weights prompt position
          (start call weights m (K.decoding call m) (Native.hidden_values h))
          (Linear.tensor (normalized weights (Native.hidden_values h))) kv) =
  K.layer_facts call weights prompt position 24 m h kv;
  let c = K.decoding call m in
  let fresh = start call weights m c (Native.hidden_values h) in
  same_history call weights prompt m fresh;
  capture_previous call position m.K.memory (Native.hidden_values h) (K.caller call weights prompt m).D.returned;
  norm_approx call weights prompt m c h kv;
  geometry ();
  Scratch.rms_frame (Config.model_norm 24 false) weights (Native.hidden_values h);
  Linear.tensor_values (normalized weights (Native.hidden_values h));
  let out = Linear.tensor (normalized weights (Native.hidden_values h)) in
  let residual (i:P.index 1024)
    : Lemma (Seq.index (Native.hidden_values out) i == (Layers.last c.K.layers).O.hidden i) = ()
  in
  FStar.Classical.forall_intro residual;
  let captured (i:P.index 1024)
    : Lemma (position >= call.Model.prompt_length - 1 ==>
        Seq.index fresh.K.memory.K.hidden ((position - call.Model.prompt_length + 1) * 1024 + i) ==
          Some ((Layers.last c.K.layers).O.hidden i)) =
    if position >= call.Model.prompt_length - 1 then
      History.captured_cell call m.K.memory.K.hidden (Native.hidden_values h) (History.retained_row call position) i
  in
  FStar.Classical.forall_intro captured;
  assert (K.common call weights prompt fresh);
  assert (K.current_valid call weights prompt fresh c out kv);
  assert (K.normalized_input call weights prompt fresh c (normalized weights (Native.hidden_values h)));
  assert (K.capture_cells call position fresh.K.memory (Layers.last c.K.layers).O.hidden);
  norm_spec [delta_once ["KPlay.Proof.Base.KernelState.head_ready"]]
    (K.head_ready call weights prompt position fresh (Linear.tensor (normalized weights (Native.hidden_values h))) kv)

inline_for_extraction noextract
let current (call:Model.invocation) (m:K.metadata call{K.Normalized? m.K.phase})
  : GTot (K.current call) =
  let K.Normalized c _ = m.K.phase in c

inline_for_extraction noextract
let values (call:Model.invocation) (m:K.metadata call{K.Normalized? m.K.phase})
  : GTot Transfer.scratch =
  let K.Normalized _ n = m.K.phase in n

inline_for_extraction noextract
let ready_facts (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call})
  (m:K.metadata call) (h:Hidden.snapshot) (kv:Cache.snapshot)
  : Lemma
      (requires K.head_ready call weights prompt position m h kv)
      (ensures
        K.Normalized? m.K.phase /\ m.K.completed == position /\
        (current call m).K.position == position /\ (current call m).K.layer == 24 /\
        K.current_valid call weights prompt m (current call m) h kv /\
        h == Linear.tensor (values call m) /\
        Native.hidden_values h == values call m /\
        K.normalized_input call weights prompt m (current call m) (values call m) /\
        K.capture_cells call position m.K.memory (Layers.last (current call m).K.layers).O.hidden /\
        K.common call weights prompt m /\
        m.K.memory.K.best == Vocabulary.negative_infinity /\ I64.v m.K.memory.K.next == 0) =
  norm_spec [delta_once ["KPlay.Proof.Base.KernelState.head_ready"]]
    (K.head_ready call weights prompt position m h kv);
  Linear.tensor_roundtrip h

inline_for_extraction noextract
let observation (call:Model.invocation) (weights:Transfer.packed)
  (c:K.current call{c.K.layer == 24}) (normalized:Transfer.scratch) : GTot Trace.observation = {
  O.position = c.K.position;
  O.consumed = c.K.token;
  O.stages = (fun next -> Seq.index c.K.layers.Layers.stages next);
  O.updates = (fun layer -> Seq.index c.K.layers.Layers.updates layer);
  O.norm_input = (Layers.last c.K.layers).O.hidden;
  O.normalized = (fun i -> Seq.index normalized (1024 + i));
  O.produced_scores = (fun i -> Seq.index (Vocabulary.score_values weights normalized) i);
  O.score_keys = Scores.keys (Vocabulary.score_values weights normalized);
  O.captured = Seq.init_ghost 1024 (fun i -> (Layers.last c.K.layers).O.hidden i);
  O.cache_after = (Layers.last c.K.layers).O.cache;
}

inline_for_extraction noextract
let normalization_equal (left right weight:P.vector real 1024)
  : Lemma
      (requires forall (i:P.index 1024). left i == right i)
      (ensures forall (i:P.index 1024).
        P.normalized_for_logits P.reference_epsilon left weight i ==
        P.normalized_for_logits P.reference_epsilon right weight i) =
  Algebra.sum_ext #1024 (Norm.squares left) (Norm.squares right) 1024;
  Norm.argument_formula 1024 left;
  Norm.argument_formula 1024 right;
  assert (P.rms_argument P.reference_epsilon left == P.rms_argument P.reference_epsilon right)

inline_for_extraction noextract
let reference_logits (call:Model.invocation) (w:Model.parameters)
  (position:nat{position < Model.calls call}) (token:D.token 248320)
  (incoming:Attention.hybrid_cache{Model.cache_shape position 0 incoming}) (index:D.token 248320)
  : Lemma (
      (Model.model_step call w position token incoming).Model.logits index ==
      P.project #1024 #248320 (K.ideal_normalized call w position token incoming)
        (fun v i -> Seq.index (Seq.index w.Model.embedding v) i) index) =
  let final = Model.layers_at call w position token incoming 24 in
  let left : P.vector real 1024 = fun i -> final.Attention.hidden i in
  let right : P.vector real 1024 = fun i -> (Model.layers_at call w position token incoming 24).Attention.hidden i in
  normalization_equal left right w.Model.final_norm;
  let a : P.vector real 1024 = fun i ->
    P.normalized_for_logits P.reference_epsilon left w.Model.final_norm i *.
      Seq.index (Seq.index w.Model.embedding index) i in
  let b : P.vector real 1024 = fun i ->
    P.normalized_for_logits P.reference_epsilon right w.Model.final_norm i *.
      Seq.index (Seq.index w.Model.embedding index) i in
  Algebra.sum_ext a b 1024;
  norm_spec [delta_once ["KPlay.Proof.Shared.Model.model_step"]]
    (Model.model_step call w position token incoming);
  norm_spec [delta_once ["KPlay.Proof.Base.KernelState.ideal_normalized"]]
    (K.ideal_normalized call w position token incoming)

inline_for_extraction noextract
let embedding_cell (weights:Transfer.packed) (token:D.token 248320) (i:P.index 1024)
  : Lemma (Seq.index (Seq.index (O.model_parameters (Inputs.parameter_values weights)).Model.embedding token) i ==
      Transfer.embedding_model weights token i) =
  assert (Seq.index (Seq.index (Inputs.parameter_values weights).O.embedding token) i ==
    Seq.index weights (token * 1024 + i));
  assert (Seq.index (Seq.index (O.model_parameters (Inputs.parameter_values weights)).Model.embedding token) i ==
    Approx.to_real (Seq.index (Seq.index (Inputs.parameter_values weights).O.embedding token) i))

inline_for_extraction noextract
let projection_terms (normalized:P.vector real 1024) (matrix:P.matrix real 248320 1024)
  (token:D.token 248320) : P.vector real 1024 =
  fun i -> normalized i *. matrix token i

inline_for_extraction noextract
let projection_formula (normalized:P.vector real 1024) (matrix:P.matrix real 248320 1024)
  (token:D.token 248320)
  : Lemma (P.project normalized matrix token == P.real_sum (projection_terms normalized matrix token)) =
  assert (P.project normalized matrix token == P.real_sum (projection_terms normalized matrix token)) by (
    FStar.Tactics.V2.norm [zeta; iota; delta_only [
      "KPlay.Proof.Shared.Projection.project"; "KPlay.Proof.Base.KernelHead.projection_terms"]];
    FStar.Tactics.V2.trefl ())

inline_for_extraction noextract
let projection_equal (normalized:P.vector real 1024) (left right:P.matrix real 248320 1024)
  (token:D.token 248320)
  : Lemma
      (requires forall (i:P.index 1024). left token i == right token i)
      (ensures P.project normalized left token == P.project normalized right token) =
  Algebra.sum_ext #1024 (projection_terms normalized left token) (projection_terms normalized right token) 1024;
  projection_formula normalized left token;
  projection_formula normalized right token

inline_for_extraction noextract
let logits_equation (call:Model.invocation) (weights:Transfer.packed)
  (position:nat{position < Model.calls call}) (token:D.token 248320)
  (incoming:Attention.hybrid_cache{Model.cache_shape position 0 incoming}) (index:D.token 248320)
  : Lemma (
      (Model.model_step call (O.model_parameters (Inputs.parameter_values weights)) position token incoming).Model.logits index ==
      P.project (K.ideal_normalized call (O.model_parameters (Inputs.parameter_values weights)) position token incoming)
        (Transfer.embedding_model weights) index) =
  let w = O.model_parameters (Inputs.parameter_values weights) in
  reference_logits call w position token incoming index;
  FStar.Classical.forall_intro (embedding_cell weights index);
  projection_equal (K.ideal_normalized call w position token incoming)
    (fun v i -> Seq.index (Seq.index w.Model.embedding v) i) (Transfer.embedding_model weights) index

inline_for_extraction noextract
let observation_valid (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call})
  (m:K.metadata call) (h:Hidden.snapshot) (kv:Cache.snapshot)
  : Lemma
      (requires K.head_ready call weights prompt position m h kv)
      (ensures K.Normalized? m.K.phase /\ m.K.completed == position /\
        (current call m).K.position == position /\ (current call m).K.layer == 24 /\
        (let observed = observation call weights (current call m) (values call m) in
        let winner = (Vocabulary.scan_blocks weights (values call m) 15520).Order.token in
        winner < 248320 /\
        O.structure (Inputs.parameter_values weights) position (K.caller call weights prompt m).D.next_token
          (K.caller call weights prompt m).D.model_cache observed /\
        O.numerical call (O.model_parameters (Inputs.parameter_values weights)) position
          (current call m).K.token (K.reference call weights prompt m) observed /\
        D.maximum observed.O.score_keys winner)) =
  ready_facts call weights prompt position m h kv;
  let c = current call m in
  let n = values call m in
  Scores.selected weights n;
  Vocabulary.complete_numerical_scores weights n
    (K.ideal_normalized call (O.model_parameters (Inputs.parameter_values weights))
      position c.K.token (K.reference call weights prompt m));
  FStar.Classical.forall_intro
    (logits_equation call weights position c.K.token (K.reference call weights prompt m));
  norm_spec [delta_once ["KPlay.Proof.Base.KernelState.ideal_normalized"]]
    (K.ideal_normalized call (O.model_parameters (Inputs.parameter_values weights))
      position c.K.token (K.reference call weights prompt m))
