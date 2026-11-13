module KPlay.Proof.Base.KernelLayers

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

module Seq = FStar.Seq
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module D = KPlay.Proof.Shared.Selection
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module K = KPlay.Proof.Base.KernelState
module Layers = KPlay.Proof.Base.LayerTrace
module Global = KPlay.Proof.Base.GlobalCache
module Numeric = KPlay.Proof.Base.LayerNumerics
module Transfer = KPlay.Proof.Base.Transfer
module Inputs = KPlay.Proof.Base.Inputs
module Native = KPlay.Proof.Base.NativeProjection
module Hidden = KPlay.Proof.Base.WarpHidden
module Cache = KPlay.Proof.Base.CacheNative
module Linear = KPlay.Proof.Base.LinearBook
module Prior = KPlay.Proof.Base.RecurrentPipeline
module Attention = KPlay.Proof.Base.AttentionBook
module Pipeline = KPlay.Proof.Base.AttentionPipeline
module AttentionStages = KPlay.Proof.Base.AttentionStages
module Conv = KPlay.Proof.Base.Convolution
module LinearStages = KPlay.Proof.Base.LinearStages
module Binding = KPlay.Proof.Base.LayerBinding
module CacheModel = KPlay.Proof.Base.Cache

inline_for_extraction noextract
let linear_entry (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call}) (layer:Model.linear_layer)
  (r:Linear.root (K.metadata call)) : prop =
  r.Linear.payload.K.completed == position /\
  K.layer call weights prompt position layer r.Linear.payload (Linear.tensor r.Linear.hidden) r.Linear.cache /\
  r.Linear.state == r.Linear.payload.K.memory.K.state /\
  r.Linear.convolution == r.Linear.payload.K.memory.K.convolution /\
  r.Linear.logical == Layers.recurrent_plan call (O.model_parameters (Inputs.parameter_values weights)) position
    (K.caller call weights prompt r.Linear.payload).D.next_token (K.reference call weights prompt r.Linear.payload) layer

inline_for_extraction noextract
let linear_start (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call}) (layer:Model.linear_layer)
  (m:K.metadata call{m.K.completed == position}) (h:Hidden.snapshot) (kv:Cache.snapshot)
  : GTot (Linear.metadata (K.metadata call)) =
  let logical = Layers.recurrent_plan call (O.model_parameters (Inputs.parameter_values weights)) position
    (K.caller call weights prompt m).D.next_token (K.reference call weights prompt m) layer in
  let r:Linear.root (K.metadata call) = {
    Linear.payload = m; Linear.logical = logical; Linear.stages = Prior.build_stages logical;
    Linear.hidden = Native.hidden_values h; Linear.state = m.K.memory.K.state;
    Linear.convolution = m.K.memory.K.convolution; Linear.cache = kv;
  } in
  {Linear.root = r; Linear.state = m.K.memory.K.state; Linear.convolution = m.K.memory.K.convolution;
   Linear.mixer_input = Native.hidden_values h; Linear.mixer_result = Native.hidden_values h}

inline_for_extraction noextract
let linear_start_valid (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call}) (layer:Model.linear_layer)
  (m:K.metadata call{m.K.completed == position}) (h:Hidden.snapshot) (kv:Cache.snapshot)
  : Lemma
      (requires K.layer call weights prompt position layer m h kv)
      (ensures Linear.valid call position layer weights True (linear_entry call weights prompt position layer) 0
        (linear_start call weights prompt position layer m h kv) h kv) =
  K.layer_facts call weights prompt position layer m h kv;
  let current = K.decoding call m in
  let result = linear_start call weights prompt position layer m h kv in
  let logical = result.Linear.root.Linear.logical in
  let expected = Model.layers_at call (O.model_parameters (Inputs.parameter_values weights)) position current.K.token
    (K.reference call weights prompt m) layer in
  Layers.shape (Inputs.parameter_values weights) position current.K.token (K.caller call weights prompt m).D.model_cache current.K.layers;
  Numeric.recurrent_inputs logical (Layers.last current.K.layers).O.cache expected.A.cache
    m.K.memory.K.state m.K.memory.K.convolution kv;
  Linear.tensor_roundtrip h;
  assert (linear_entry call weights prompt position layer result.Linear.root);
  assert (result.Linear.root.Linear.stages == Prior.build_stages logical);
  assert (LinearStages.live logical result.Linear.root.Linear.stages (Native.hidden_values h) 0);
  Linear.initial_numeric_views result.Linear.root;
  norm_spec [delta_once ["KPlay.Proof.Base.LinearBook.valid"]]
    (Linear.valid call position layer weights True (linear_entry call weights prompt position layer) 0 result h kv)

inline_for_extraction noextract
let attention_entry (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (r:Attention.root (K.metadata call)) : prop =
  r.Attention.payload.K.completed == position /\
  K.layer call weights prompt position layer r.Attention.payload (Linear.tensor r.Attention.hidden) r.Attention.cache /\
  r.Attention.logical == Layers.attention_plan call (O.model_parameters (Inputs.parameter_values weights)) position
    (K.caller call weights prompt r.Attention.payload).D.next_token (K.reference call weights prompt r.Attention.payload) layer

inline_for_extraction noextract
let attention_start (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (m:K.metadata call{m.K.completed == position}) (h:Hidden.snapshot) (kv:Cache.snapshot)
  : GTot (Attention.metadata (K.metadata call)) =
  let logical = Layers.attention_plan call (O.model_parameters (Inputs.parameter_values weights)) position
    (K.caller call weights prompt m).D.next_token (K.reference call weights prompt m) layer in
  let r:Attention.root (K.metadata call) = {
    Attention.payload = m; Attention.logical = logical; Attention.stages = Pipeline.build logical;
    Attention.hidden = Native.hidden_values h; Attention.cache = kv;
  } in
  {Attention.root = r; Attention.prepared = Native.hidden_values h}

inline_for_extraction noextract
let attention_start_valid (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (m:K.metadata call{m.K.completed == position}) (h:Hidden.snapshot) (kv:Cache.snapshot)
  : Lemma
      (requires K.layer call weights prompt position layer m h kv)
      (ensures Attention.valid call position layer weights True (attention_entry call weights prompt position layer) 0
        (attention_start call weights prompt position layer m h kv) h kv) =
  K.layer_facts call weights prompt position layer m h kv;
  let current = K.decoding call m in
  let result = attention_start call weights prompt position layer m h kv in
  let expected = Model.layers_at call (O.model_parameters (Inputs.parameter_values weights)) position current.K.token
    (K.reference call weights prompt m) layer in
  Layers.shape (Inputs.parameter_values weights) position current.K.token (K.caller call weights prompt m).D.model_cache current.K.layers;
  Global.initialized (Layers.last current.K.layers).O.cache m.K.memory.K.state m.K.memory.K.convolution kv position layer layer;
  Global.history_approx (Layers.last current.K.layers).O.cache expected.A.cache m.K.memory.K.state m.K.memory.K.convolution kv layer;
  Linear.tensor_roundtrip h;
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionBook.valid"]]
    (Attention.valid call position layer weights True (attention_entry call weights prompt position layer) 0 result h kv)

inline_for_extraction noextract
let advance (call:Model.invocation) (m:K.metadata call) (c:K.current call{c.K.layer < 24})
  (h:Hidden.snapshot) (cache:KPlay.Proof.Base.Cache.native_cache) (update:Layers.update)
  (state:FStar.Seq.lseq FP.t 4718592) (conv:Conv.cells) : GTot (K.metadata call) =
  let current:K.current call = {
    c with K.layer = c.K.layer + 1;
    K.layers = Layers.append c.K.layers (Native.hidden_values h) cache update;
  } in
  {m with K.memory = {m.K.memory with K.state = state; K.convolution = conv}; K.phase = K.Decoding current}

inline_for_extraction noextract
let advance_valid (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call}) (layer:Model.cursor{layer < 24})
  (m:K.metadata call{m.K.completed == position}) (c:K.current call{c.K.layer == layer})
  (before after:Hidden.snapshot) (old_kv new_kv:Cache.snapshot)
  (cache:KPlay.Proof.Base.Cache.native_cache) (update:Layers.update)
  (state:FStar.Seq.lseq FP.t 4718592) (conv:Conv.cells)
  : Lemma
      (requires K.layer call weights prompt position layer m before old_kv /\ m.K.phase == K.Decoding c /\
        O.layer_structure position layer (Layers.last c.K.layers).O.cache cache update /\
        Global.represents cache state conv new_kv /\
        Linear.same_rope (Native.hidden_values before) (Native.hidden_values after) /\
        O.vector_approximates (Layers.hidden (Native.hidden_values after))
          (fun i -> (Model.layers_at call (O.model_parameters (Inputs.parameter_values weights)) position c.K.token
            (K.reference call weights prompt m) (layer + 1)).A.hidden i) /\
        O.cache_approximates cache
          (Model.layers_at call (O.model_parameters (Inputs.parameter_values weights)) position c.K.token
            (K.reference call weights prompt m) (layer + 1)).A.cache)
      (ensures K.layer call weights prompt position (layer + 1) (advance call m c after cache update state conv) after new_kv) =
  K.layer_facts call weights prompt position layer m before old_kv;
  Layers.append_safety (Inputs.parameter_values weights) position c.K.token
    (K.caller call weights prompt m).D.model_cache c.K.layers (Native.hidden_values after) cache update;
  Layers.append_numerical call (O.model_parameters (Inputs.parameter_values weights)) position c.K.token
    (K.reference call weights prompt m) c.K.layers (Native.hidden_values after) cache update;
  Attention.rope_preserved True position (Native.hidden_values before) (Native.hidden_values after);
  norm_spec [delta_once ["KPlay.Proof.Base.KernelState.layer"]]
    (K.layer call weights prompt position (layer + 1) (advance call m c after cache update state conv) after new_kv)

inline_for_extraction noextract
let linear_end_valid (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call}) (layer:Model.linear_layer)
  (old:K.metadata call{old.K.completed == position}) (current:K.current call{current.K.layer == layer})
  (m:Linear.metadata (K.metadata call){m.Linear.root.Linear.payload == old}) (h:Hidden.snapshot) (kv:Cache.snapshot)
  : Lemma
      (requires old.K.phase == K.Decoding current /\
        Linear.valid call position layer weights True (linear_entry call weights prompt position layer) 13 m h kv)
      (ensures K.layer call weights prompt position (layer + 1)
        (advance call old current h (Global.linear_after (Layers.last current.K.layers).O.cache layer m.Linear.state m.Linear.convolution)
          (Global.linear_update layer m.Linear.root.Linear.state m.Linear.state m.Linear.root.Linear.convolution m.Linear.mixer_input)
          m.Linear.state m.Linear.convolution) h kv) =
  norm_spec [delta_once ["KPlay.Proof.Base.LinearBook.valid"]]
    (Linear.valid call position layer weights True (linear_entry call weights prompt position layer) 13 m h kv);
  norm_spec [delta_once ["KPlay.Proof.Base.KernelLayers.linear_entry"]]
    (linear_entry call weights prompt position layer m.Linear.root);
  K.layer_facts call weights prompt position layer old (Linear.tensor m.Linear.root.Linear.hidden) m.Linear.root.Linear.cache;
  Layers.shape (Inputs.parameter_values weights) position current.K.token
    (K.caller call weights prompt old).D.model_cache current.K.layers;
  Binding.linear_completed call weights position current.K.token (K.reference call weights prompt old) layer
    (linear_entry call weights prompt position layer) m h kv (Layers.last current.K.layers).O.cache;
  Linear.tensor_values m.Linear.root.Linear.hidden;
  advance_valid call weights prompt position layer old current (Linear.tensor m.Linear.root.Linear.hidden) h m.Linear.root.Linear.cache kv
    (Global.linear_after (Layers.last current.K.layers).O.cache layer m.Linear.state m.Linear.convolution)
    (Global.linear_update layer m.Linear.root.Linear.state m.Linear.state m.Linear.root.Linear.convolution m.Linear.mixer_input)
    m.Linear.state m.Linear.convolution

inline_for_extraction noextract
let attention_end_valid (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (old:K.metadata call{old.K.completed == position}) (current:K.current call{current.K.layer == layer})
  (m:Attention.metadata (K.metadata call){m.Attention.root.Attention.payload == old})
  (h:Hidden.snapshot) (kv:Cache.snapshot) (next_cache:CacheModel.native_cache)
  : Lemma
      (requires old.K.phase == K.Decoding current /\
        Attention.valid call position layer weights True (attention_entry call weights prompt position layer) 37 m h kv /\
        O.cache_shape position layer (Layers.last current.K.layers).O.cache /\
        next_cache == CacheModel.source_after (Layers.last current.K.layers).O.cache m.Attention.prepared layer position)
      (ensures K.layer call weights prompt position (layer + 1)
        (advance call old current h next_cache
          (O.AttentionUpdate (CacheModel.produced_key m.Attention.prepared) (CacheModel.produced_value m.Attention.prepared))
          old.K.memory.K.state old.K.memory.K.convolution) h kv) =
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionBook.valid"]]
    (Attention.valid call position layer weights True (attention_entry call weights prompt position layer) 37 m h kv);
  norm_spec [delta_once ["KPlay.Proof.Base.KernelLayers.attention_entry"]]
    (attention_entry call weights prompt position layer m.Attention.root);
  K.layer_facts call weights prompt position layer old (Linear.tensor m.Attention.root.Attention.hidden) m.Attention.root.Attention.cache;
  Binding.attention_completed call weights position current.K.token (K.reference call weights prompt old) layer
    (attention_entry call weights prompt position layer) m h kv (Layers.last current.K.layers).O.cache
    old.K.memory.K.state old.K.memory.K.convolution;
  Linear.tensor_values m.Attention.root.Attention.hidden;
  advance_valid call weights prompt position layer old current (Linear.tensor m.Attention.root.Attention.hidden) h m.Attention.root.Attention.cache kv
    next_cache (O.AttentionUpdate (CacheModel.produced_key m.Attention.prepared) (CacheModel.produced_value m.Attention.prepared))
    old.K.memory.K.state old.K.memory.K.convolution
