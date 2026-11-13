module KPlay.Proof.Base.KernelPosition

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

module Seq = FStar.Seq
module I64 = FStar.Int64
module BF = Kuiper.BFloat16
module P = KPlay.Proof.Shared.Projection
module D = KPlay.Proof.Shared.Selection
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Shared = KPlay.SharedSpec
module K = KPlay.Proof.Base.KernelState
module Head = KPlay.Proof.Base.KernelHead
module Trace = KPlay.Proof.Base.Trace
module Layers = KPlay.Proof.Base.LayerTrace
module Transfer = KPlay.Proof.Base.Transfer
module Inputs = KPlay.Proof.Base.Inputs
module Embedding = KPlay.Proof.Base.Embedding
module Hidden = KPlay.Proof.Base.WarpHidden
module Native = KPlay.Proof.Base.NativeProjection
module Cache = KPlay.Proof.Base.CacheNative
module Global = KPlay.Proof.Base.GlobalCache
module Linear = KPlay.Proof.Base.LinearBook
module Stages = KPlay.Proof.Base.AttentionStages
module History = KPlay.Proof.Base.HistoryNative

inline_for_extraction noextract
let ready_facts (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position <= Model.calls call})
  (m:K.metadata call) (h:Hidden.snapshot) (kv:Cache.snapshot)
  : Lemma
      (requires K.ready call weights prompt position m h kv)
      (ensures m.K.completed == position /\ m.K.phase == K.Between /\
        K.common call weights prompt m /\ K.previous_selection call m /\
        Global.represents (K.caller call weights prompt m).D.model_cache m.K.memory.K.state m.K.memory.K.convolution kv) =
  norm_spec [delta_once ["KPlay.Proof.Base.KernelState.ready"]]
    (K.ready call weights prompt position m h kv)

inline_for_extraction noextract
let token_binding (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length)
  (physical:D.vec I64.t call.Model.prompt_length) (position:nat{position < Model.calls call})
  (m:K.metadata call) (h:Hidden.snapshot) (kv:Cache.snapshot)
  : Lemma
      (requires K.ready call weights prompt position m h kv /\
        (forall (i:P.index call.Model.prompt_length). I64.v (Seq.index physical i) == Seq.index prompt i))
      (ensures
        Embedding.prompt_valid call physical /\
        (position < call.Model.prompt_length \/ (0 <= I64.v m.K.memory.K.next /\ I64.v m.K.memory.K.next < 248320)) /\
        I64.v (Embedding.source_token call position physical m.K.memory.K.next) == (K.caller call weights prompt m).D.next_token) =
  ready_facts call weights prompt position m h kv;
  Trace.consumed call (Inputs.parameter_values weights) prompt K.configuration m.K.history position

inline_for_extraction noextract
let enter (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length)
  (m:K.metadata call{m.K.completed < Model.calls call}) (token:D.token 248320) (after:Transfer.scratch)
  : GTot (K.metadata call) =
  {m with K.phase = K.Decoding {
    K.position = m.K.completed; K.token = token; K.layer = 0;
    K.layers = Layers.start after (K.caller call weights prompt m).D.model_cache}}

inline_for_extraction noextract
let enter_valid (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call})
  (m:K.metadata call) (before:Hidden.snapshot) (kv:Cache.snapshot) (token:D.token 248320) (after:Transfer.scratch)
  : Lemma
      (requires K.ready call weights prompt position m before kv /\
        token == (K.caller call weights prompt m).D.next_token /\
        Stages.rope position after /\
        (forall (i:P.index 1024). Seq.index after i == Seq.index (Embedding.row weights token) i))
      (ensures m.K.completed == position /\
        K.layer call weights prompt position 0 (enter call weights prompt m token after) (Linear.tensor after) kv) =
  ready_facts call weights prompt position m before kv;
  Trace.cache_shape call (Inputs.parameter_values weights) prompt K.configuration m.K.history position;
  Trace.cache_approx call (Inputs.parameter_values weights) prompt K.configuration m.K.history;
  Layers.start_safety weights position token (K.caller call weights prompt m).D.model_cache after;
  Layers.start_numerical call weights position token (K.caller call weights prompt m).D.model_cache
    (K.reference call weights prompt m) after;
  let fresh = enter call weights prompt m token after in
  Head.same_history call weights prompt m fresh;
  Linear.tensor_values after;
  norm_spec [delta_once ["KPlay.Proof.Base.KernelState.layer"]]
    (K.layer call weights prompt position 0 fresh (Linear.tensor after) kv)

inline_for_extraction noextract
let observation (call:Model.invocation) (m:K.metadata call{K.Emitting? m.K.phase}) : GTot Trace.observation =
  let K.Emitting observed _ = m.K.phase in observed

inline_for_extraction noextract
let winner (call:Model.invocation) (m:K.metadata call{K.Emitting? m.K.phase}) : GTot (D.token 248320) =
  let K.Emitting _ selected = m.K.phase in selected

inline_for_extraction noextract
let emitting_facts (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call})
  (m:K.metadata call) (h:Hidden.snapshot) (kv:Cache.snapshot)
  : Lemma
      (requires K.emitting call weights prompt position m h kv)
      (ensures m.K.completed == position /\ K.Emitting? m.K.phase /\
        K.common call weights prompt m /\ I64.v m.K.memory.K.next == winner call m /\
        O.structure (Inputs.parameter_values weights) position (K.caller call weights prompt m).D.next_token
          (K.caller call weights prompt m).D.model_cache (observation call m) /\
        O.numerical call (O.model_parameters (Inputs.parameter_values weights)) position
          (Model.consumed call prompt (Trace.selected call m.K.history) position)
          (K.reference call weights prompt m) (observation call m) /\
        D.maximum (observation call m).O.score_keys (winner call m) /\
        Global.represents (observation call m).O.cache_after m.K.memory.K.state m.K.memory.K.convolution kv /\
        K.capture_cells call position m.K.memory (observation call m).O.norm_input) =
  norm_spec [delta_once ["KPlay.Proof.Base.KernelState.emitting"]]
    (K.emitting call weights prompt position m h kv)

inline_for_extraction noextract
let emitted (call:Model.invocation) (position:nat{position < Model.calls call}) (memory:K.memory call)
  : GTot (K.memory call) =
  {memory with K.tokens =
    (if position < call.Model.prompt_length - 1 then memory.K.tokens
      else Seq.upd memory.K.tokens (History.retained_row call position) (Some memory.K.next))}

inline_for_extraction noextract
let advance (call:Model.invocation)
  (m:K.metadata call{m.K.completed < Model.calls call /\ K.Emitting? m.K.phase})
  : GTot (K.metadata call) = {
  K.completed = m.K.completed + 1;
  K.history = Trace.append m.K.history (observation call m) (winner call m);
  K.memory = emitted call m.K.completed m.K.memory;
  K.phase = K.Between;
}

inline_for_extraction noextract
let history_advance (call:Model.invocation) (position:nat{position < Model.calls call})
  (prompt:D.vec (D.token 248320) call.Model.prompt_length)
  (before:Shared.caller_state call O.cache_configuration BF.t BF.t Kuiper.Float32.t BF.t position)
  (memory:K.memory call) (observed:Trace.observation) (selected:D.token 248320)
  : Lemma
      (requires
        K.history_cells call (D.retained_count call.Model.prompt_length position) memory before.D.returned /\
        K.capture_cells call position memory observed.O.norm_input /\
        (forall (i:P.index 1024). Seq.index observed.O.captured i == observed.O.norm_input i) /\
        I64.v memory.K.next == selected)
      (ensures
        K.history_cells call (D.retained_count call.Model.prompt_length (position + 1)) (emitted call position memory)
          (D.advance prompt (D.prepare_call before) (Shared.event observed) selected).D.returned) =
  let after = D.advance prompt (D.prepare_call before) (Shared.event observed) selected in
  let tokens (row:P.index (D.retained_count call.Model.prompt_length (position + 1)))
    : Lemma (exists (word:I64.t).
        Seq.index (emitted call position memory).K.tokens row == Some word /\
        I64.v word == Seq.index after.D.returned.D.generated_tokens row) =
    if position >= call.Model.prompt_length - 1 then (
      if row == position - call.Model.prompt_length + 1 then
        assert (Seq.index (emitted call position memory).K.tokens row == Some memory.K.next)
      else
        assert (row < D.retained_count call.Model.prompt_length position)
    )
  in
  FStar.Classical.forall_intro tokens;
  let hidden (row:P.index (D.retained_count call.Model.prompt_length (position + 1))) (i:P.index 1024)
    : Lemma (Seq.index (emitted call position memory).K.hidden (row * 1024 + i) ==
        Some (Seq.index (Seq.index after.D.returned.D.hidden_history row) i)) =
    if position >= call.Model.prompt_length - 1 then (
      if row == position - call.Model.prompt_length + 1 then ()
      else assert (row < D.retained_count call.Model.prompt_length position)
    )
  in
  FStar.Classical.forall_intro_2 hidden

inline_for_extraction noextract
let advance_valid (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call})
  (m:K.metadata call) (h:Hidden.snapshot) (kv:Cache.snapshot)
  : Lemma
      (requires K.emitting call weights prompt position m h kv)
      (ensures m.K.completed == position /\ K.Emitting? m.K.phase /\
        K.ready call weights prompt (position + 1) (advance call m) h kv) =
  emitting_facts call weights prompt position m h kv;
  let observed = observation call m in
  let selected = winner call m in
  Trace.append_safety call (Inputs.parameter_values weights) prompt K.configuration m.K.history observed selected;
  Trace.append_numerical call (Inputs.parameter_values weights) prompt m.K.history observed selected;
  Trace.append_state call (Inputs.parameter_values weights) prompt K.configuration m.K.history observed selected;
  history_advance call position prompt (K.caller call weights prompt m) m.K.memory observed selected;
  norm_spec [delta_once ["KPlay.Proof.Base.KernelState.ready"]]
    (K.ready call weights prompt (position + 1) (advance call m) h kv)
