module KPlay.Proof.Base.LayerTrace

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

module Seq = FStar.Seq
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module P = KPlay.Proof.Shared.Projection
module D = KPlay.Proof.Shared.Selection
module A = KPlay.Proof.Shared.Attention
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Cache = KPlay.Proof.Base.Cache
module Transfer = KPlay.Proof.Base.Transfer
module Inputs = KPlay.Proof.Base.Inputs
module Embedding = KPlay.Proof.Base.Embedding
module Prior = KPlay.Proof.Base.RecurrentPipeline
module Attention = KPlay.Proof.Base.AttentionPipeline
module Recurrent = KPlay.Proof.Shared.Recurrent

inline_for_extraction noextract
let stage = O.stage O.cache_configuration BF.t BF.t FP.t BF.t

inline_for_extraction noextract
let update = O.layer_update BF.t FP.t BF.t

noeq type prefix (next:Model.cursor) = {
  stages:D.vec stage (next + 1);
  updates:D.vec update next;
}

inline_for_extraction noextract
let hidden (values:Transfer.scratch) : O.vector BF.t 1024 =
  fun i -> Seq.index values i

inline_for_extraction noextract
let start (values:Transfer.scratch) (cache:Cache.native_cache) : prefix 0 = {
  stages = Seq.create 1 {O.hidden = hidden values; O.cache = cache};
  updates = Seq.empty;
}

inline_for_extraction noextract
let last (#next:Model.cursor) (p:prefix next) : stage =
  Seq.index p.stages next

inline_for_extraction noextract
let append (#next:Model.cursor{next < 24}) (p:prefix next)
  (values:Transfer.scratch) (cache:Cache.native_cache) (update:update) : prefix (next + 1) = {
  stages = Seq.append p.stages (Seq.create 1 {O.hidden = hidden values; O.cache = cache});
  updates = Seq.append p.updates (Seq.create 1 update);
}

inline_for_extraction noextract
let safety (#next:Model.cursor) (weights:O.parameters BF.t) (position:nat) (token:D.token 248320)
  (incoming:Cache.native_cache) (p:prefix next) : prop =
  (Seq.index p.stages 0).O.cache == incoming /\
  (forall (i:P.index 1024).
    (Seq.index p.stages 0).O.hidden i ==
      Seq.index (KPlay.Proof.Shared.Embedding.embedding weights.O.embedding token) i) /\
  O.cache_shape position 0 incoming /\
  (forall (layer:nat{layer < next}).
    O.layer_structure position layer (Seq.index p.stages layer).O.cache (Seq.index p.stages (layer + 1)).O.cache
      (Seq.index p.updates layer))

inline_for_extraction noextract
let numerical (#next:Model.cursor) (call:Model.invocation) (weights:Model.parameters)
  (position:nat{position < Model.calls call}) (token:D.token 248320)
  (incoming:A.hybrid_cache{Model.cache_shape position 0 incoming}) (p:prefix next) : prop =
  forall (layer:nat{layer <= next}).
    let expected = Model.layers_at call weights position token incoming layer in
    O.vector_approximates (Seq.index p.stages layer).O.hidden (fun i -> expected.A.hidden i) /\
    O.cache_approximates (Seq.index p.stages layer).O.cache expected.A.cache

inline_for_extraction noextract
let start_safety (weights:Transfer.packed) (position:nat) (token:D.token 248320)
  (incoming:Cache.native_cache) (values:Transfer.scratch)
  : Lemma
      (requires O.cache_shape position 0 incoming /\
        (forall (i:P.index 1024). Seq.index values i == Seq.index (Embedding.row weights token) i))
      (ensures safety (Inputs.parameter_values weights) position token incoming (start values incoming)) = ()

inline_for_extraction noextract
let start_numerical (call:Model.invocation) (weights:Transfer.packed)
  (position:nat{position < Model.calls call}) (token:D.token 248320)
  (incoming:Cache.native_cache) (ideal:A.hybrid_cache{Model.cache_shape position 0 ideal}) (values:Transfer.scratch)
  : Lemma
      (requires O.cache_approximates incoming ideal /\
        (forall (i:P.index 1024). Seq.index values i == Seq.index (Embedding.row weights token) i))
      (ensures numerical call (O.model_parameters (Inputs.parameter_values weights)) position token ideal (start values incoming)) =
  Inputs.embedding_model weights token

inline_for_extraction noextract
let shape (#next:Model.cursor) (weights:O.parameters BF.t) (position:nat) (token:D.token 248320)
  (incoming:Cache.native_cache) (p:prefix next)
  : Lemma
      (requires safety weights position token incoming p)
      (ensures O.cache_shape position next (last p).O.cache) =
  if next > 0 then
    assert (O.layer_structure position (next - 1)
      (Seq.index p.stages (next - 1)).O.cache (last p).O.cache (Seq.index p.updates (next - 1)))

inline_for_extraction noextract
let append_safety (#next:Model.cursor{next < 24}) (weights:O.parameters BF.t) (position:nat) (token:D.token 248320)
  (incoming:Cache.native_cache) (p:prefix next) (values:Transfer.scratch) (cache:Cache.native_cache) (update:update)
  : Lemma
      (requires safety weights position token incoming p /\
        O.layer_structure position next (last p).O.cache cache update)
      (ensures safety weights position token incoming (append p values cache update)) = ()

inline_for_extraction noextract
let append_numerical (#next:Model.cursor{next < 24}) (call:Model.invocation) (weights:Model.parameters)
  (position:nat{position < Model.calls call}) (token:D.token 248320)
  (incoming:A.hybrid_cache{Model.cache_shape position 0 incoming}) (p:prefix next)
  (values:Transfer.scratch) (cache:Cache.native_cache) (update:update)
  : Lemma
      (requires numerical call weights position token incoming p /\
        O.vector_approximates (hidden values) (fun i -> (Model.layers_at call weights position token incoming (next + 1)).A.hidden i) /\
        O.cache_approximates cache (Model.layers_at call weights position token incoming (next + 1)).A.cache)
      (ensures numerical call weights position token incoming (append p values cache update)) = ()

inline_for_extraction noextract
let recurrent_plan (call:Model.invocation) (weights:Model.parameters)
  (position:nat{position < Model.calls call}) (token:D.token 248320)
  (incoming:A.hybrid_cache{Model.cache_shape position 0 incoming}) (layer:Model.linear_layer)
  : GTot Prior.logical_call =
  let before = Model.layers_at call weights position token incoming layer in
  Model.recurrent_entry call position layer before.A.cache;
  {Prior.call = Model.recurrent_call call position layer;
   Prior.weight = weights.Model.linear layer;
   Prior.input = (fun i -> before.A.hidden i);
   Prior.before = Model.recurrent_cache before.A.cache}

inline_for_extraction noextract
let attention_plan (call:Model.invocation) (weights:Model.parameters)
  (position:nat{position < Model.calls call}) (token:D.token 248320)
  (incoming:A.hybrid_cache{Model.cache_shape position 0 incoming}) (layer:A.full_attention_layer)
  : GTot Attention.logical_call =
  let before = Model.layers_at call weights position token incoming layer in
  Model.attention_entry call position layer before.A.cache;
  {Attention.call = Model.attention_call call position layer;
   Attention.weight = weights.Model.full layer;
   Attention.input = before.A.hidden;
   Attention.before = before.A.cache}

inline_for_extraction noextract
let successor (call:Model.invocation) (weights:Model.parameters)
  (position:nat{position < Model.calls call}) (token:D.token 248320)
  (incoming:A.hybrid_cache{Model.cache_shape position 0 incoming}) (layer:Model.cursor{layer < 24})
  : Lemma (
      Model.layers_at call weights position token incoming (layer + 1) ==
      Model.layer_step call weights position layer
        (Model.layers_at call weights position token incoming layer).A.hidden
        (Model.layers_at call weights position token incoming layer).A.cache) =
  norm_spec [delta_once ["KPlay.Proof.Shared.Model.layers_at"]]
    (Model.layers_at call weights position token incoming (layer + 1))

inline_for_extraction noextract
let recurrent_reference (call:Model.invocation) (weights:Model.parameters)
  (position:nat{position < Model.calls call}) (token:D.token 248320)
  (incoming:A.hybrid_cache{Model.cache_shape position 0 incoming}) (layer:Model.linear_layer)
  : Lemma (
      let c = recurrent_plan call weights position token incoming layer in
      let values = Recurrent.decoder_values c.Prior.call c.Prior.weight c.Prior.input c.Prior.before in
      (forall (i:P.index 1024).
        (Model.layers_at call weights position token incoming (layer + 1)).A.hidden i == values.Recurrent.hidden_value i) /\
      (Model.layers_at call weights position token incoming (layer + 1)).A.cache ==
        Model.install_recurrent (Model.layers_at call weights position token incoming layer).A.cache layer values) =
  successor call weights position token incoming layer;
  norm_spec [delta_once ["KPlay.Proof.Shared.Model.layer_step"]]
    (Model.layer_step call weights position layer
      (Model.layers_at call weights position token incoming layer).A.hidden
      (Model.layers_at call weights position token incoming layer).A.cache)

inline_for_extraction noextract
let attention_reference (call:Model.invocation) (weights:Model.parameters)
  (position:nat{position < Model.calls call}) (token:D.token 248320)
  (incoming:A.hybrid_cache{Model.cache_shape position 0 incoming}) (layer:A.full_attention_layer)
  : Lemma (
      let c = attention_plan call weights position token incoming layer in
      Model.layers_at call weights position token incoming (layer + 1) ==
        A.full_attention_decoder c.Attention.call c.Attention.weight c.Attention.input c.Attention.before) =
  successor call weights position token incoming layer;
  norm_spec [delta_once ["KPlay.Proof.Shared.Model.layer_step"]]
    (Model.layer_step call weights position layer
      (Model.layers_at call weights position token incoming layer).A.hidden
      (Model.layers_at call weights position token incoming layer).A.cache)
