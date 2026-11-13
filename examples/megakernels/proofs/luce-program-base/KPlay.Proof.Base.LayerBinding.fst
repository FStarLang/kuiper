module KPlay.Proof.Base.LayerBinding

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
module Seq = FStar.Seq
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module P = KPlay.Proof.Shared.Projection
module D = KPlay.Proof.Shared.Selection
module A = KPlay.Proof.Shared.Attention
module R = KPlay.Proof.Shared.Recurrent
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Layers = KPlay.Proof.Base.LayerTrace
module Inputs = KPlay.Proof.Base.Inputs
module Transfer = KPlay.Proof.Base.Transfer
module Native = KPlay.Proof.Base.NativeProjection
module Hidden = KPlay.Proof.Base.WarpHidden
module Init = KPlay.Proof.Base.Initialization
module Cache = KPlay.Proof.Base.Cache
module NativeCache = KPlay.Proof.Base.CacheNative
module Global = KPlay.Proof.Base.GlobalCache
module Numeric = KPlay.Proof.Base.LayerNumerics
module Linear = KPlay.Proof.Base.LinearBook
module Prior = KPlay.Proof.Base.RecurrentPipeline
module Mixer = KPlay.Proof.Base.RecurrentMixer
module Attention = KPlay.Proof.Base.AttentionBook
module AttentionPipeline = KPlay.Proof.Base.AttentionPipeline

inline_for_extraction noextract
let linear_completed (#extra:Type0)
  (call:Model.invocation) (weights:Transfer.packed) (position:nat{position < Model.calls call})
  (token:D.token 248320) (incoming:A.hybrid_cache{Model.cache_shape position 0 incoming}) (layer:Model.linear_layer)
  (anchor:Linear.root extra -> prop) (m:Linear.metadata extra) (h:Hidden.snapshot) (kv:NativeCache.snapshot)
  (before:Cache.native_cache)
  : Lemma
      (requires Linear.valid call position layer weights True anchor 13 m h kv /\
        m.Linear.root.Linear.logical ==
          Layers.recurrent_plan call (O.model_parameters (Inputs.parameter_values weights)) position token incoming layer /\
        Global.represents before m.Linear.root.Linear.state m.Linear.root.Linear.convolution m.Linear.root.Linear.cache /\
        O.cache_shape position layer before /\
        O.cache_approximates before
          (Model.layers_at call (O.model_parameters (Inputs.parameter_values weights)) position token incoming layer).A.cache)
      (ensures
        O.layer_structure position layer before (Global.linear_after before layer m.Linear.state m.Linear.convolution)
          (Global.linear_update layer m.Linear.root.Linear.state m.Linear.state m.Linear.root.Linear.convolution m.Linear.mixer_input) /\
        Global.represents (Global.linear_after before layer m.Linear.state m.Linear.convolution) m.Linear.state m.Linear.convolution kv /\
        O.vector_approximates (Layers.hidden (Native.hidden_values h))
          (fun i -> (Model.layers_at call (O.model_parameters (Inputs.parameter_values weights)) position token incoming (layer + 1)).A.hidden i) /\
        O.cache_approximates (Global.linear_after before layer m.Linear.state m.Linear.convolution)
          (Model.layers_at call (O.model_parameters (Inputs.parameter_values weights)) position token incoming (layer + 1)).A.cache) =
  norm_spec [delta_once ["KPlay.Proof.Base.LinearBook.valid"]]
    (Linear.valid call position layer weights True anchor 13 m h kv);
  let c = m.Linear.root.Linear.logical in
  let s = m.Linear.root.Linear.stages in
  let values = R.decoder_values c.Prior.call c.Prior.weight c.Prior.input c.Prior.before in
  Global.linear_structure before position layer m.Linear.root.Linear.state m.Linear.state
    m.Linear.root.Linear.convolution m.Linear.mixer_input kv;
  Global.linear_represents before layer m.Linear.root.Linear.state m.Linear.state
    m.Linear.root.Linear.convolution m.Linear.mixer_input kv;
  Numeric.recurrent_result c s layer m.Linear.state m.Linear.mixer_result;
  Numeric.convolution_result layer m.Linear.root.Linear.convolution m.Linear.mixer_input
    (Prior.old_convolution c.Prior.call c.Prior.before) s.Prior.raw;
  let canonical = Prior.build_stages c in
  Prior.models_match c;
  Numeric.install_linear before
    (Model.layers_at call (O.model_parameters (Inputs.parameter_values weights)) position token incoming layer).A.cache
    layer m.Linear.state m.Linear.convolution values;
  Layers.recurrent_reference call (O.model_parameters (Inputs.parameter_values weights)) position token incoming layer

inline_for_extraction noextract
let attention_completed (#extra:Type0)
  (call:Model.invocation) (weights:Transfer.packed) (position:nat{position < Model.calls call})
  (token:D.token 248320) (incoming:A.hybrid_cache{Model.cache_shape position 0 incoming}) (layer:A.full_attention_layer)
  (anchor:Attention.root extra -> prop) (m:Attention.metadata extra) (h:Hidden.snapshot) (kv:NativeCache.snapshot)
  (before:Cache.native_cache) (state:lseq FP.t 4718592) (conv:KPlay.Proof.Base.Convolution.cells)
  : Lemma
      (requires Attention.valid call position layer weights True anchor 37 m h kv /\
        m.Attention.root.Attention.logical ==
          Layers.attention_plan call (O.model_parameters (Inputs.parameter_values weights)) position token incoming layer /\
        Global.represents before state conv m.Attention.root.Attention.cache /\
        O.cache_shape position layer before /\
        O.cache_approximates before
          (Model.layers_at call (O.model_parameters (Inputs.parameter_values weights)) position token incoming layer).A.cache)
      (ensures
        O.layer_structure position layer before (Cache.source_after before m.Attention.prepared layer position)
          (O.AttentionUpdate (Cache.produced_key m.Attention.prepared) (Cache.produced_value m.Attention.prepared)) /\
        Global.represents (Cache.source_after before m.Attention.prepared layer position) state conv kv /\
        O.vector_approximates (Layers.hidden (Native.hidden_values h))
          (fun i -> (Model.layers_at call (O.model_parameters (Inputs.parameter_values weights)) position token incoming (layer + 1)).A.hidden i) /\
        O.cache_approximates (Cache.source_after before m.Attention.prepared layer position)
          (Model.layers_at call (O.model_parameters (Inputs.parameter_values weights)) position token incoming (layer + 1)).A.cache) =
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionBook.valid"]]
    (Attention.valid call position layer weights True anchor 37 m h kv);
  let c = m.Attention.root.Attention.logical in
  let s = m.Attention.root.Attention.stages in
  Global.attention_represents before position layer state conv m.Attention.root.Attention.cache m.Attention.prepared;
  Cache.source_structure before m.Attention.prepared layer position;
  Numeric.install_attention before
    (Model.layers_at call (O.model_parameters (Inputs.parameter_values weights)) position token incoming layer).A.cache
    layer position m.Attention.prepared s.AttentionPipeline.history kv;
  AttentionPipeline.reference_result c s;
  Layers.attention_reference call (O.model_parameters (Inputs.parameter_values weights)) position token incoming layer
