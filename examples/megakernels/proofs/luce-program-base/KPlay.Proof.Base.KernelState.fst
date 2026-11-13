module KPlay.Proof.Base.KernelState

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
module Seq = FStar.Seq
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module I64 = FStar.Int64
module P = KPlay.Proof.Shared.Projection
module D = KPlay.Proof.Shared.Selection
module A = KPlay.Proof.Shared.Attention
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Trace = KPlay.Proof.Base.Trace
module Layers = KPlay.Proof.Base.LayerTrace
module Inputs = KPlay.Proof.Base.Inputs
module Transfer = KPlay.Proof.Base.Transfer
module Native = KPlay.Proof.Base.NativeProjection
module Hidden = KPlay.Proof.Base.WarpHidden
module Global = KPlay.Proof.Base.GlobalCache
module Cache = KPlay.Proof.Base.Cache
module NativeCache = KPlay.Proof.Base.CacheNative
module Conv = KPlay.Proof.Base.Convolution
module History = KPlay.Proof.Base.HistoryNative
module Vocabulary = KPlay.Proof.Base.Vocabulary
module AttentionStages = KPlay.Proof.Base.AttentionStages

inline_for_extraction noextract
let configuration : O.cache_configuration = {
  O.linear_layers = (fun l -> l % 4 < 3);
  O.transformer_layers = Seq.init #nat 6 (fun i -> 4 * i + 3);
  O.last_linear_layer = 22;
}

inline_for_extraction noextract
let configuration_ok () : Lemma (O.configuration_ok configuration) = ()

noeq type memory (call:Model.invocation) = {
  state:lseq FP.t 4718592;
  convolution:Conv.cells;
  tokens:lseq (option I64.t) call.Model.decode_steps;
  hidden:lseq (option BF.t) (call.Model.decode_steps * 1024);
  next:I64.t;
  best:FP.t;
}

noeq type current (call:Model.invocation) = {
  position:p:nat{p < Model.calls call};
  token:D.token 248320;
  layer:Model.cursor;
  layers:Layers.prefix layer;
}

noeq type phase (call:Model.invocation) =
  | Between : phase call
  | Decoding : current call -> phase call
  | Normalized : current call -> Transfer.scratch -> phase call
  | Emitting : Trace.observation -> D.token 248320 -> phase call

noeq type metadata (call:Model.invocation) = {
  completed:n:nat{n <= Model.calls call};
  history:Trace.prefix completed;
  memory:memory call;
  phase:phase call;
}

inline_for_extraction noextract
let caller (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (m:metadata call) =
  Trace.state call (Inputs.parameter_values weights) prompt configuration m.history m.completed

inline_for_extraction noextract
let reference (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (m:metadata call)
  : GTot (c:A.hybrid_cache{Model.cache_shape m.completed 0 c}) =
  Model.cache_at call (O.model_parameters (Inputs.parameter_values weights)) prompt (Trace.selected call m.history) m.completed

inline_for_extraction noextract
let history_cells (call:Model.invocation) (count:nat{count <= call.Model.decode_steps})
  (mem:memory call) (history:D.histories 248320 1024 BF.t count) : prop =
  (forall (row:P.index count). exists (word:I64.t).
    Seq.index mem.tokens row == Some word /\ I64.v word == Seq.index history.D.generated_tokens row) /\
  (forall (row:P.index count) (i:P.index 1024).
    Seq.index mem.hidden (row * 1024 + i) == Some (Seq.index (Seq.index history.D.hidden_history row) i))

inline_for_extraction noextract
let common (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (m:metadata call) : prop =
  Trace.safety call (Inputs.parameter_values weights) prompt configuration m.history /\
  Trace.numerical call (Inputs.parameter_values weights) prompt m.history /\
  history_cells call (D.retained_count call.Model.prompt_length m.completed) m.memory
    (caller call weights prompt m).D.returned

inline_for_extraction noextract
let previous_selection (call:Model.invocation) (m:metadata call) : prop =
  I64.v m.memory.next == (if m.completed = 0 then 0 else Seq.index m.history.Trace.selected (m.completed - 1))

[@@"opaque_to_smt"]
inline_for_extraction noextract
let ready (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (completed:nat{completed <= Model.calls call})
  (m:metadata call) (_h:Hidden.snapshot) (kv:NativeCache.snapshot) : prop =
  m.completed == completed /\ m.phase == Between /\
  common call weights prompt m /\ previous_selection call m /\
  Global.represents (caller call weights prompt m).D.model_cache m.memory.state m.memory.convolution kv

inline_for_extraction noextract
let current_valid (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (m:metadata call) (c:current call)
  (h:Hidden.snapshot) (kv:NativeCache.snapshot) : prop =
  c.position == m.completed /\
  c.token == (caller call weights prompt m).D.next_token /\
  Layers.safety (Inputs.parameter_values weights) c.position c.token
    (caller call weights prompt m).D.model_cache c.layers /\
  Layers.numerical call (O.model_parameters (Inputs.parameter_values weights)) c.position c.token
    (reference call weights prompt m) c.layers /\
  Global.represents (Layers.last c.layers).O.cache m.memory.state m.memory.convolution kv /\
  (forall (i:P.index 1024). Seq.index (Native.hidden_values h) i == (Layers.last c.layers).O.hidden i)

[@@"opaque_to_smt"]
inline_for_extraction noextract
let layer (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call}) (next:Model.cursor)
  (m:metadata call) (h:Hidden.snapshot) (kv:NativeCache.snapshot) : prop =
  m.completed == position /\ common call weights prompt m /\ previous_selection call m /\
  AttentionStages.rope position (Native.hidden_values h) /\
  exists (c:current call).
    m.phase == Decoding c /\ c.position == position /\ c.layer == next /\ current_valid call weights prompt m c h kv

inline_for_extraction noextract
let decoding (call:Model.invocation) (m:metadata call{Decoding? m.phase}) : GTot (current call) =
  let Decoding c = m.phase in c

inline_for_extraction noextract
let layer_facts (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call}) (next:Model.cursor)
  (m:metadata call) (h:Hidden.snapshot) (kv:NativeCache.snapshot)
  : Lemma
      (requires layer call weights prompt position next m h kv)
      (ensures
        m.completed == position /\ Decoding? m.phase /\
        (decoding call m).position == position /\ (decoding call m).layer == next /\
        current_valid call weights prompt m (decoding call m) h kv /\
        common call weights prompt m /\ previous_selection call m /\
        AttentionStages.rope position (Native.hidden_values h)) =
  norm_spec [delta_once ["KPlay.Proof.Base.KernelState.layer"]]
    (layer call weights prompt position next m h kv)

inline_for_extraction noextract
let capture_cells (call:Model.invocation) (position:nat{position < Model.calls call})
  (mem:memory call) (captured:O.vector BF.t 1024) : prop =
  if position < call.Model.prompt_length - 1 then True
  else forall (i:P.index 1024).
    Seq.index mem.hidden ((position - call.Model.prompt_length + 1) * 1024 + i) == Some (captured i)

inline_for_extraction noextract
let ideal_normalized (call:Model.invocation) (weights:Model.parameters)
  (position:nat{position < Model.calls call}) (token:D.token 248320)
  (incoming:A.hybrid_cache{Model.cache_shape position 0 incoming})
  : GTot (P.vector FStar.Real.real 1024) =
  P.normalized_for_logits P.reference_epsilon
    (fun i -> (Model.layers_at call weights position token incoming 24).A.hidden i)
    weights.Model.final_norm

inline_for_extraction noextract
let normalized_input (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (m:metadata call) (c:current call) (values:Transfer.scratch) : prop =
  c.position == m.completed /\
  Transfer.normalized_input values
    (ideal_normalized call (O.model_parameters (Inputs.parameter_values weights))
      c.position c.token (reference call weights prompt m))

[@@"opaque_to_smt"]
inline_for_extraction noextract
let head_ready (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call})
  (m:metadata call) (h:Hidden.snapshot) (kv:NativeCache.snapshot) : prop =
  common call weights prompt m /\ m.memory.best == Vocabulary.negative_infinity /\ I64.v m.memory.next == 0 /\
  exists (c:current call) (normalized:Transfer.scratch).
    m.phase == Normalized c normalized /\ c.position == position /\ c.layer == 24 /\
    current_valid call weights prompt m c h kv /\
    Native.hidden_values h == normalized /\
    normalized_input call weights prompt m c normalized /\
    capture_cells call position m.memory (Layers.last c.layers).O.hidden

[@@"opaque_to_smt"]
inline_for_extraction noextract
let emitting (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call})
  (m:metadata call) (_h:Hidden.snapshot) (kv:NativeCache.snapshot) : prop =
  m.completed == position /\ common call weights prompt m /\
  exists (observed:Trace.observation) (winner:D.token 248320).
    m.phase == Emitting observed winner /\ I64.v m.memory.next == winner /\
    O.structure (Inputs.parameter_values weights) position (caller call weights prompt m).D.next_token
      (caller call weights prompt m).D.model_cache observed /\
    O.numerical call (O.model_parameters (Inputs.parameter_values weights)) position
      (Model.consumed call prompt (Trace.selected call m.history) position)
      (reference call weights prompt m) observed /\
    D.maximum observed.O.score_keys winner /\
    Global.represents observed.O.cache_after m.memory.state m.memory.convolution kv /\
    capture_cells call position m.memory observed.O.norm_input
