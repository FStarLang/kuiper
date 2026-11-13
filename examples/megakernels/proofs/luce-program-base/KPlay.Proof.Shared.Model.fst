module KPlay.Proof.Shared.Model

module S = FStar.Seq
module E = KPlay.Proof.Shared.Embedding
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module R = KPlay.Proof.Shared.Recurrent
module D = KPlay.Proof.Shared.Selection
module Softplus = Luce.Softplus

inline_for_extraction noextract
let linear_layer = l:A.layer_index{l % 4 < 3}

inline_for_extraction noextract
let cursor = n:nat{n <= 24}

inline_for_extraction noextract
noeq type invocation = {
  smoke: bool;
  prompt_length: p:pos{p <= 520};
  decode_steps: s:pos{
    (if smoke then s == 1 else s == 1 \/ s == 5 \/ s == 30 \/ s == 128) /\
    prompt_length + s - 1 <= 2048
  };
}

inline_for_extraction noextract
let calls (call:invocation) : pos =
  D.call_count call.prompt_length call.decode_steps

inline_for_extraction noextract
noeq type parameters = {
  embedding: E.embedding_weights FStar.Real.real;
  final_norm: P.vector FStar.Real.real 1024;
  linear: linear_layer -> GTot R.weights;
  full: A.full_attention_layer -> GTot A.layer_weights;
}

inline_for_extraction noextract
let visits (position:nat) (next:cursor) (layer:A.layer_index) : nat =
  position + (if layer < next then 1 else 0)

(* The same invariant covers token boundaries and partially visited layers. *)
inline_for_extraction noextract
let cache_shape (position:nat) (next:cursor) (cache:A.hybrid_cache) : prop =
  forall (layer:A.layer_index).
    let count = visits position next layer in
    if layer % 4 < 3 then
      cache.kv layer == None /\
      (Some? (cache.convolution layer) <==> count > 0) /\
      (Some? (cache.recurrent layer) <==> count > 0)
    else
      cache.convolution layer == None /\
      cache.recurrent layer == None /\
      A.history_length (cache.kv layer) == count

inline_for_extraction noextract
let initial_shape () : Lemma (cache_shape 0 0 (A.empty_cache ())) = ()

inline_for_extraction noextract
let token_boundary (position:nat) (cache:A.hybrid_cache)
  : Lemma
      (requires cache_shape position 24 cache)
      (ensures cache_shape (position + 1) 0 cache) = ()

inline_for_extraction noextract
let recurrent_cache (cache:A.hybrid_cache) : GTot R.model_cache = {
  conv_slots = (fun l ->
    match cache.convolution l with
    | None -> None
    | Some c -> Some (fun i j -> c i j));
  recurrent_slots = (fun l ->
    match cache.recurrent l with
    | None -> None
    | Some r -> Some (fun h i j -> r h i j));
  kv_slots = (fun l ->
    match cache.kv l with
    | None -> None
    | Some k -> Some {
        R.kv_length = k.length;
        R.keys = (fun h t d -> k.keys h t d);
        R.values = (fun h t d -> k.values h t d)
      });
}

inline_for_extraction noextract
let recurrent_call
  (call:invocation) (position:nat{position < calls call}) (layer:linear_layer)
  : R.invocation = {
  R.smoke = call.smoke;
  R.prompt_length = call.prompt_length;
  R.decode_steps = call.decode_steps;
  R.cache_position = position;
  R.layer_index = layer;
}

inline_for_extraction noextract
let attention_call
  (call:invocation) (position:nat{position < calls call})
  (layer:A.full_attention_layer)
  : A.invocation = {
  A.prompt_length = call.prompt_length;
  A.decode_steps = call.decode_steps;
  A.position = position;
  A.layer = layer;
}

inline_for_extraction noextract
let recurrent_entry
  (call:invocation) (position:nat{position < calls call}) (layer:linear_layer)
  (cache:A.hybrid_cache)
  : Lemma
      (requires cache_shape position layer cache)
      (ensures R.cache_domain (recurrent_call call position layer)
                 (recurrent_cache cache)) = ()

inline_for_extraction noextract
let attention_entry
  (call:invocation) (position:nat{position < calls call})
  (layer:A.full_attention_layer) (cache:A.hybrid_cache)
  : Lemma
      (requires cache_shape position layer cache)
      (ensures A.domain (attention_call call position layer) cache) = ()

inline_for_extraction noextract
let install_recurrent
  (before:A.hybrid_cache) (layer:linear_layer) (values:R.layer_values)
  : GTot A.hybrid_cache = {
  A.kv = before.kv;
  A.convolution = (fun l ->
    if l = layer then Some (fun c t -> values.convolution_value c t)
    else before.convolution l);
  A.recurrent = (fun l ->
    if l = layer then Some (fun h i j -> values.recurrent_value h i j)
    else before.recurrent l);
}

inline_for_extraction noextract
let install_recurrent_shape
  (position:nat) (layer:linear_layer) (before:A.hybrid_cache)
  (values:R.layer_values)
  : Lemma
      (requires cache_shape position layer before)
      (ensures cache_shape position (layer + 1)
                 (install_recurrent before layer values)) = ()

inline_for_extraction noextract
let attention_shape
  (call:invocation) (position:nat{position < calls call})
  (layer:A.full_attention_layer) (w:A.layer_weights) (hidden:A.vector 1024)
  (before:A.hybrid_cache{cache_shape position layer before})
  : Lemma
      (ensures
        cache_shape position (layer + 1)
          (A.full_attention_decoder (attention_call call position layer)
            w hidden before).cache) =
  attention_entry call position layer before

inline_for_extraction noextract
let layer_step
  (call:invocation) (w:parameters) (position:nat{position < calls call})
  (layer:A.layer_index) (hidden:A.vector 1024)
  (before:A.hybrid_cache{cache_shape position layer before})
  : GTot (result:A.layer_result{
      cache_shape position (layer + 1) result.cache}) =
  if layer % 4 < 3 then (
    recurrent_entry call position layer before;
    let values =
      R.decoder_values (recurrent_call call position layer) (w.linear layer)
        (fun i -> hidden i) (recurrent_cache before)
    in
    install_recurrent_shape position layer before values;
    {
      A.hidden = (fun i -> values.hidden_value i);
      A.cache = install_recurrent before layer values
    }
  )
  else (
    attention_entry call position layer before;
    attention_shape call position layer (w.full layer) hidden before;
    A.full_attention_decoder (attention_call call position layer)
      (w.full layer) hidden before
  )

inline_for_extraction noextract
let rec layers_at
  (call:invocation) (w:parameters) (position:nat{position < calls call})
  (token:(D.token 248320)) (before:A.hybrid_cache{cache_shape position 0 before})
  (next:cursor)
  : GTot (result:A.layer_result{cache_shape position next result.cache})
      (decreases next) =
  if next = 0 then {
    A.hidden = (fun i -> S.index (E.algebraic_embedding w.embedding token) i);
    A.cache = before
  }
  else
    let previous = layers_at call w position token before (next - 1) in
    layer_step call w position (next - 1) previous.hidden previous.cache

inline_for_extraction noextract
noeq type step_result = {
  residual: A.vector 1024;
  logits: P.vector FStar.Real.real 248320;
  cache: A.hybrid_cache;
}

inline_for_extraction noextract
let model_step
  (call:invocation) (w:parameters) (position:nat{position < calls call})
  (token:(D.token 248320)) (before:A.hybrid_cache{cache_shape position 0 before})
  : GTot (result:step_result{cache_shape (position + 1) 0 result.cache}) =
  let final_layer = layers_at call w position token before 24 in
  token_boundary position final_layer.cache;
  {
    residual = final_layer.hidden;
    logits =
      P.project
        (P.normalized_for_logits P.reference_epsilon
          (fun i -> final_layer.hidden i) w.final_norm)
        (fun v i -> S.index (S.index w.embedding v) i);
    cache = final_layer.cache;
  }

inline_for_extraction noextract
let model_step_cache
  (call:invocation) (w:parameters) (position:nat{position < calls call})
  (token:(D.token 248320)) (before:A.hybrid_cache{cache_shape position 0 before})
  : Lemma (
      (model_step call w position token before).cache ==
      (layers_at call w position token before 24).A.cache) = ()

inline_for_extraction noextract
let model_step_residual
  (call:invocation) (w:parameters) (position:nat{position < calls call})
  (token:(D.token 248320)) (before:A.hybrid_cache{cache_shape position 0 before})
  : Lemma (
      (model_step call w position token before).residual ==
      (layers_at call w position token before 24).A.hidden) = ()

inline_for_extraction noextract
let consumed
  (call:invocation) (prompt:D.vec (D.token 248320) call.prompt_length)
  (selected:D.vec (D.token 248320) (calls call))
  (position:nat{position < calls call})
  : (D.token 248320) =
  if position < call.prompt_length then S.index prompt position
  else S.index selected (position - 1)

(* The cache is computed, not supplied as an arbitrary transition witness. *)
inline_for_extraction noextract
let rec cache_at
  (call:invocation) (w:parameters)
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (selected:D.vec (D.token 248320) (calls call))
  (completed:nat{completed <= calls call})
  : GTot (cache:A.hybrid_cache{cache_shape completed 0 cache})
      (decreases completed) =
  if completed = 0 then (
    initial_shape ();
    A.empty_cache ()
  )
  else
    let previous = cache_at call w prompt selected (completed - 1) in
    (model_step call w (completed - 1)
      (consumed call prompt selected (completed - 1)) previous).cache

inline_for_extraction noextract
let cache_at_successor
  (call:invocation) (w:parameters)
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (selected:D.vec (D.token 248320) (calls call))
  (position:nat{position < calls call})
  : Lemma (
      cache_at call w prompt selected (position + 1) ==
      (model_step call w position (consumed call prompt selected position)
        (cache_at call w prompt selected position)).cache) = ()

inline_for_extraction noextract
let computation_at
  (call:invocation) (w:parameters)
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (selected:D.vec (D.token 248320) (calls call))
  (position:nat{position < calls call})
  : GTot step_result =
  model_step call w position (consumed call prompt selected position)
    (cache_at call w prompt selected position)

inline_for_extraction noextract
let full_profile (call:invocation)
  : Lemma (
      D.reference_domain call.smoke 248320 1024
        call.prompt_length call.decode_steps /\
      calls call <= 647 /\ calls call <= 2048) = ()

inline_for_extraction noextract
let authorized_softplus_model (x:FStar.Real.real)
  : Lemma (R.softplus x == Softplus.model x) = ()
