module KPlay.Proof.Shared.Observations

module S = FStar.Seq
module C = FStar.Classical
module F = Kuiper.Floating.Base
module Scalar = Kuiper.Scalars.Base
module Approx = Kuiper.Approximates.Base
module E = KPlay.Proof.Shared.Embedding
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module R = KPlay.Proof.Shared.Recurrent
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model

inline_for_extraction noextract
let vector (a:Type0) (n:nat) = P.vector a n

inline_for_extraction noextract
let matrix (a:Type0) (m n:nat) = P.matrix a m n

inline_for_extraction noextract
let tensor (a:Type0) (h m n:nat) =
  P.index h -> GTot (matrix a m n)

inline_for_extraction noextract
let model_vector (#a:Type0) {| Scalar.scalar a, Approx.real_like a |}
  (#n:nat) (x:vector a n) : vector FStar.Real.real n =
  fun i -> Approx.to_real (x i)

inline_for_extraction noextract
let model_matrix (#a:Type0) {| Scalar.scalar a, Approx.real_like a |}
  (#m #n:nat) (x:matrix a m n) : matrix FStar.Real.real m n =
  fun i -> model_vector (x i)

inline_for_extraction noextract
noeq type linear_weights (a:Type0) = {
  input_norm: vector a 1024;
  post_attention_norm: vector a 1024;
  in_qkv: matrix a 6144 1024;
  in_z: matrix a 2048 1024;
  in_beta: matrix a 16 1024;
  in_a: matrix a 16 1024;
  conv_kernel: matrix a 6144 4;
  a_log: vector a 16;
  dt_bias: vector a 16;
  gated_norm: vector a 128;
  out_projection: matrix a 1024 2048;
  mlp_gate: matrix a 3584 1024;
  mlp_up: matrix a 3584 1024;
  mlp_down: matrix a 1024 3584;
}

inline_for_extraction noextract
noeq type full_weights (a:Type0) = {
  input_norm: vector a 1024;
  post_attention_norm: vector a 1024;
  query_gate_proj: matrix a 4096 1024;
  key_proj: matrix a 512 1024;
  value_proj: matrix a 512 1024;
  query_norm: vector a 256;
  key_norm: vector a 256;
  output_proj: matrix a 1024 2048;
  mlp_gate_proj: matrix a 3584 1024;
  mlp_up_proj: matrix a 3584 1024;
  mlp_down_proj: matrix a 1024 3584;
}

inline_for_extraction noextract
noeq type parameters (a:Type0) = {
  embedding: E.embedding_weights a;
  final_norm: vector a 1024;
  linear: Model.linear_layer -> GTot (linear_weights a);
  full: A.full_attention_layer -> GTot (full_weights a);
}

inline_for_extraction noextract
let model_linear (#a:Type0) {| Scalar.scalar a, Approx.real_like a |}
  (w:linear_weights a) : GTot R.weights = {
  R.input_norm = model_vector w.input_norm;
  R.post_attention_norm = model_vector w.post_attention_norm;
  R.in_qkv = model_matrix w.in_qkv;
  R.in_z = model_matrix w.in_z;
  R.in_beta = model_matrix w.in_beta;
  R.in_a = model_matrix w.in_a;
  R.conv_kernel = model_matrix w.conv_kernel;
  R.a_log = model_vector w.a_log;
  R.dt_bias = model_vector w.dt_bias;
  R.gated_norm = model_vector w.gated_norm;
  R.out_projection = model_matrix w.out_projection;
  R.mlp_gate = model_matrix w.mlp_gate;
  R.mlp_up = model_matrix w.mlp_up;
  R.mlp_down = model_matrix w.mlp_down;
}

inline_for_extraction noextract
let model_full (#a:Type0) {| Scalar.scalar a, Approx.real_like a |}
  (w:full_weights a) : GTot A.layer_weights = {
  A.input_norm = (fun i -> Approx.to_real (w.input_norm i));
  A.post_attention_norm = (fun i -> Approx.to_real (w.post_attention_norm i));
  A.query_gate_proj = (fun i j -> Approx.to_real (w.query_gate_proj i j));
  A.key_proj = (fun i j -> Approx.to_real (w.key_proj i j));
  A.value_proj = (fun i j -> Approx.to_real (w.value_proj i j));
  A.query_norm = (fun i -> Approx.to_real (w.query_norm i));
  A.key_norm = (fun i -> Approx.to_real (w.key_norm i));
  A.output_proj = (fun i j -> Approx.to_real (w.output_proj i j));
  A.mlp_gate_proj = (fun i j -> Approx.to_real (w.mlp_gate_proj i j));
  A.mlp_up_proj = (fun i j -> Approx.to_real (w.mlp_up_proj i j));
  A.mlp_down_proj = (fun i j -> Approx.to_real (w.mlp_down_proj i j));
}

(* Only immutable input cells are anchored with to_real. Intermediate values
   approximate the propagated ideal values; no inverse-cast law is used. *)
inline_for_extraction noextract
let model_parameters (#a:Type0) {| Scalar.scalar a, Approx.real_like a |}
  (w:parameters a) : GTot Model.parameters = {
  Model.embedding =
    S.init #(E.hidden_vector FStar.Real.real) E.vocab_size (fun t ->
      S.init #FStar.Real.real E.hidden_size
        (fun i -> Approx.to_real (S.index (S.index w.embedding t) i)));
  Model.final_norm = model_vector w.final_norm;
  Model.linear = (fun l -> model_linear (w.linear l));
  Model.full = (fun l -> model_full (w.full l));
}

inline_for_extraction noextract
let embedding_models_input (#a:Type0)
  {| Scalar.scalar a, Approx.real_like a |}
  (w:parameters a) (token:(D.token 248320))
  : Lemma (
      forall (i:P.index 1024).
        Approx.v_approximates (S.index (E.embedding w.embedding token) i)
          (S.index (E.algebraic_embedding (model_parameters w).Model.embedding token) i)) =
  let each_token (t:E.token_id) : Lemma (
      forall (i:E.hidden_index).
        Approx.v_approximates (S.index (S.index w.embedding t) i)
          (S.index (S.index (model_parameters w).Model.embedding t) i)) =
    let each_cell (i:E.hidden_index) : Lemma (
        Approx.v_approximates (S.index (S.index w.embedding t) i)
          (S.index (S.index (model_parameters w).Model.embedding t) i)) =
      Approx.to_real_ok (S.index (S.index w.embedding t) i)
    in
    C.forall_intro each_cell
  in
  C.forall_intro each_token;
  E.embedding_transports_relation Approx.v_approximates
    w.embedding (model_parameters w).Model.embedding token

inline_for_extraction noextract
noeq type cache_configuration = {
  linear_layers: A.layer_index -> bool;
  transformer_layers: D.vec nat 6;
  last_linear_layer: nat;
}

inline_for_extraction noextract
let configuration_ok (metadata:cache_configuration) : prop =
  (forall (l:A.layer_index). metadata.linear_layers l == (l % 4 < 3)) /\
  (forall (i:P.index 6). S.index metadata.transformer_layers i == 4 * i + 3) /\
  metadata.last_linear_layer == 22

inline_for_extraction noextract
noeq type kv_history (kv:Type0) = {
  length: pos;
  keys: tensor kv 2 length 256;
  values: tensor kv 2 length 256;
}

inline_for_extraction noextract
noeq type cache (m c r k:Type0) = {
  metadata: m;
  convolution: A.layer_index -> GTot (option (matrix c 6144 4));
  recurrent: A.layer_index -> GTot (option (tensor r 16 128 128));
  kv: A.layer_index -> GTot (option (kv_history k));
}

inline_for_extraction noextract
let empty_cache (#m #c #r #k:Type0) (metadata:m) : cache m c r k = {
  metadata;
  convolution = (fun _ -> None);
  recurrent = (fun _ -> None);
  kv = (fun _ -> None);
}

inline_for_extraction noextract
let history_length (#k:Type0) (history:option (kv_history k)) : nat =
  match history with
  | None -> 0
  | Some h -> h.length

inline_for_extraction noextract
let cache_shape (#m #c #r #k:Type0)
  (position:nat) (next:Model.cursor) (cache:cache m c r k) : GTot prop =
  forall (layer:A.layer_index).
    let count = Model.visits position next layer in
    if layer % 4 < 3 then
      cache.kv layer == None /\
      (Some? (cache.convolution layer) <==> count > 0) /\
      (Some? (cache.recurrent layer) <==> count > 0)
    else
      cache.convolution layer == None /\
      cache.recurrent layer == None /\
      history_length (cache.kv layer) == count

inline_for_extraction noextract
let vector_approximates (#a:Type0)
  {| Scalar.scalar a, Approx.real_like a |}
  (#n:nat) (actual:vector a n) (ideal:vector FStar.Real.real n) : prop =
  forall (i:P.index n). Approx.v_approximates (actual i) (ideal i)

inline_for_extraction noextract
let matrix_approximates (#a:Type0)
  {| Scalar.scalar a, Approx.real_like a |}
  (#m #n:nat) (actual:matrix a m n) (ideal:matrix FStar.Real.real m n) : prop =
  forall (i:P.index m). vector_approximates (actual i) (ideal i)

inline_for_extraction noextract
let tensor_approximates (#a:Type0)
  {| Scalar.scalar a, Approx.real_like a |}
  (#h #m #n:nat) (actual:tensor a h m n)
  (ideal:tensor FStar.Real.real h m n) : prop =
  forall (i:P.index h). matrix_approximates (actual i) (ideal i)

inline_for_extraction noextract
let cache_approximates (#m #c #r #k:Type0)
  {| Scalar.scalar c, Approx.real_like c,
     Scalar.scalar r, Approx.real_like r,
     Scalar.scalar k, Approx.real_like k |}
  (actual:cache m c r k) (ideal:A.hybrid_cache) : GTot prop =
  (forall (l:A.layer_index).
    match actual.convolution l, ideal.A.convolution l with
    | None, None -> True
    | Some x, Some y -> matrix_approximates x (fun i j -> y i j)
    | _ -> False) /\
  (forall (l:A.layer_index).
    match actual.recurrent l, ideal.A.recurrent l with
    | None, None -> True
    | Some x, Some y -> tensor_approximates x (fun h i j -> y h i j)
    | _ -> False) /\
  (forall (l:A.layer_index).
    match actual.kv l, ideal.A.kv l with
    | None, None -> True
    | Some x, Some y ->
        x.length == y.A.length /\
        (forall (h:P.index 2) (t:nat{t < x.length /\ t < y.A.length})
                (i:P.index 256).
          Approx.v_approximates (x.keys h t i) (y.A.keys h t i) /\
          Approx.v_approximates (x.values h t i) (y.A.values h t i))
    | _ -> False)

inline_for_extraction noextract
noeq type layer_update (c r k:Type0) =
  | LinearUpdate :
      old_convolution:matrix c 6144 4 ->
      old_recurrent:tensor r 16 128 128 ->
      produced_raw_qkv:vector c 6144 ->
      produced_recurrent:tensor r 16 128 128 ->
      layer_update c r k
  | AttentionUpdate :
      produced_rotated_key:matrix k 2 256 ->
      produced_raw_value:matrix k 2 256 ->
      layer_update c r k

inline_for_extraction noextract
let convolution_input (#m #c #r #k:Type0) {| Scalar.scalar c |}
  (before:cache m c r k) (l:Model.linear_layer) (old:matrix c 6144 4) : GTot prop =
  match before.convolution l with
  | None -> forall (i:P.index 6144) (j:P.index 4). old i j == Scalar.zero
  | Some stored -> forall (i:P.index 6144) (j:P.index 4). old i j == stored i j

inline_for_extraction noextract
let recurrent_input (#m #c #r #k:Type0) {| Scalar.scalar r |}
  (before:cache m c r k) (l:Model.linear_layer)
  (old:tensor r 16 128 128) : GTot prop =
  match before.recurrent l with
  | None ->
      forall (h:P.index 16) (i j:P.index 128). old h i j == Scalar.zero
  | Some stored ->
      forall (h:P.index 16) (i j:P.index 128). old h i j == stored h i j

inline_for_extraction noextract
let kv_append (#k:Type0)
  (position:nat) (old:option (kv_history k)) (next:kv_history k)
  (key value:matrix k 2 256) : GTot prop =
  history_length old == position /\
  next.length == position + 1 /\
  (forall (h:P.index 2) (d:P.index 256).
    (forall (t:nat{t < next.length /\ t == position}).
      next.keys h t d == key h d /\ next.values h t d == value h d) /\
    (match old with
    | None -> True
    | Some previous ->
        forall (t:nat{t < previous.length /\ t < next.length}).
          next.keys h t d == previous.keys h t d /\
          next.values h t d == previous.values h t d))

inline_for_extraction noextract
let layer_structure (#m #c #r #k:Type0)
  {| Scalar.scalar c, Scalar.scalar r |}
  (position:nat) (layer:A.layer_index) (before after:cache m c r k)
  (update:layer_update c r k) : GTot prop =
  before.metadata == after.metadata /\
  cache_shape position layer before /\
  cache_shape position (layer + 1) after /\
  (if layer % 4 < 3 then
    match update with
    | LinearUpdate old_conv old_rec raw produced_rec ->
        convolution_input before layer old_conv /\
        recurrent_input before layer old_rec /\
        (forall (l:A.layer_index).
          after.kv l == before.kv l /\
          (l <> layer ==>
            after.convolution l == before.convolution l /\
            after.recurrent l == before.recurrent l)) /\
        after.recurrent layer == Some produced_rec /\
        (match after.convolution layer with
        | None -> False
        | Some next ->
            forall (channel:P.index 6144) (tap:P.index 4).
              next channel tap ==
                (if tap < 3 then old_conv channel (tap + 1) else raw channel))
    | _ -> False
  else
    match update with
    | AttentionUpdate key value ->
        (forall (l:A.layer_index).
          after.convolution l == before.convolution l /\
          after.recurrent l == before.recurrent l /\
          (l <> layer ==> after.kv l == before.kv l)) /\
        (match after.kv layer with
        | None -> False
        | Some next -> kv_append position (before.kv layer) next key value)
    | _ -> False)

inline_for_extraction noextract
noeq type stage (m h c r k:Type0) = {
  hidden: vector h 1024;
  cache: cache m c r k;
}

(* These fields observe source-produced values. Roles must bind them to their
   actual operations and storage, not choose a trace satisfying this relation. *)
inline_for_extraction noextract
noeq type model_observation (m h c r k norm score:Type0) = {
  position: nat;
  consumed: (D.token 248320);
  stages: Model.cursor -> GTot (stage m h c r k);
  updates: A.layer_index -> GTot (layer_update c r k);
  norm_input: vector h 1024;
  normalized: vector norm 1024;
  produced_scores: vector score 248320;
  score_keys: D.vec D.score 248320;
  captured: D.vec h 1024;
  cache_after: cache m c r k;
}

inline_for_extraction noextract
let priority_keys (#score:Type0) {| F.floating score |}
  (actual:vector score 248320) (keys:D.vec D.score 248320) : GTot prop =
  (forall (i:(D.token 248320)).
    let x = actual i in
    let key = S.index keys i in
    match F.kind x with
    | F.NaN -> key == D.NotANumber
    | F.Infinite ->
        key == (if F.lt x F.zero then D.NegativeInfinity else D.PositiveInfinity)
    | F.Finite ->
        D.FiniteScore? key /\
        (F.is_zero x ==> key == D.FiniteScore 0.0R)) /\
  (forall (i j:(D.token 248320)).
    ~(F.NaN? (F.kind (actual i))) /\ ~(F.NaN? (F.kind (actual j))) ==>
      (D.score_le (S.index keys i) (S.index keys j) <==>
       F.lte (actual i) (actual j)))

inline_for_extraction noextract
let native_maximum (#score:Type0) {| F.floating score |}
  (actual:vector score 248320) (winner:(D.token 248320)) : GTot prop =
  forall (i:(D.token 248320)).
    F.NaN? (F.kind (actual winner)) \/
    (~(F.NaN? (F.kind (actual i))) /\ F.lte (actual i) (actual winner))

inline_for_extraction noextract
let maximum_is_same_actual_scores (#score:Type0) {| F.floating score |}
  (actual:vector score 248320) (keys:D.vec D.score 248320)
  (winner:(D.token 248320))
  : Lemma
      (requires priority_keys actual keys)
      (ensures D.maximum keys winner <==> native_maximum actual winner) = ()

inline_for_extraction noextract
let structure (#m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar c, Scalar.scalar r, F.floating score |}
  (w:parameters h) (position:nat) (consumed:(D.token 248320))
  (incoming:cache m c r k) (observed:model_observation m h c r k norm score)
  : GTot prop =
  observed.position == position /\
  observed.consumed == consumed /\
  (observed.stages 0).cache == incoming /\
  (forall (i:P.index 1024).
    (observed.stages 0).hidden i == S.index (E.embedding w.embedding consumed) i) /\
  (forall (l:A.layer_index).
    layer_structure position l (observed.stages l).cache
      (observed.stages (l + 1)).cache (observed.updates l)) /\
  (forall (i:P.index 1024).
    observed.norm_input i == (observed.stages 24).hidden i /\
    S.index observed.captured i == observed.norm_input i) /\
  observed.cache_after == (observed.stages 24).cache /\
  priority_keys observed.produced_scores observed.score_keys

inline_for_extraction noextract
let cache_boundary (#m #c #r #k:Type0)
  (position:nat) (cache:cache m c r k)
  : Lemma
      (requires cache_shape position 24 cache)
      (ensures cache_shape (position + 1) 0 cache) = ()

inline_for_extraction noextract
let structure_boundary (#m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar c, Scalar.scalar r, F.floating score |}
  (w:parameters h) (position:nat) (consumed:(D.token 248320))
  (incoming:cache m c r k) (observed:model_observation m h c r k norm score)
  : Lemma
      (requires structure w position consumed incoming observed)
      (ensures
        cache_shape (position + 1) 0 observed.cache_after /\
        observed.cache_after.metadata == incoming.metadata) =
  let rec metadata_frame (next:Model.cursor)
    : Lemma
        (requires structure w position consumed incoming observed)
        (ensures (observed.stages next).cache.metadata == incoming.metadata)
        (decreases next) =
    if next > 0 then (
      metadata_frame (next - 1);
      assert (layer_structure position (next - 1)
        (observed.stages (next - 1)).cache (observed.stages next).cache
        (observed.updates (next - 1)));
      assert ((observed.stages next).cache.metadata ==
        (observed.stages (next - 1)).cache.metadata)
    )
  in
  metadata_frame 24;
  assert (layer_structure position 23
    (observed.stages 23).cache (observed.stages 24).cache (observed.updates 23));
  cache_boundary position observed.cache_after

inline_for_extraction noextract
let numerical (#m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar h, Approx.real_like h,
     Scalar.scalar c, Approx.real_like c,
     Scalar.scalar r, Approx.real_like r,
     Scalar.scalar k, Approx.real_like k,
     Scalar.scalar norm, Approx.real_like norm,
     Scalar.scalar score, Approx.real_like score |}
  (call:Model.invocation) (w:Model.parameters)
  (position:nat{position < Model.calls call}) (consumed:(D.token 248320))
  (incoming:A.hybrid_cache{Model.cache_shape position 0 incoming})
  (observed:model_observation m h c r k norm score)
  : GTot prop =
  (forall (next:Model.cursor).
    let expected = Model.layers_at call w position consumed incoming next in
    vector_approximates (observed.stages next).hidden (fun i -> expected.A.hidden i) /\
    cache_approximates (observed.stages next).cache expected.A.cache) /\
  vector_approximates observed.normalized
    (P.normalized_for_logits P.reference_epsilon
      (fun i -> (Model.layers_at call w position consumed incoming 24).A.hidden i)
      w.final_norm) /\
  vector_approximates observed.produced_scores
    (Model.model_step call w position consumed incoming).logits

inline_for_extraction noextract
let numerical_cache_boundary (#m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar h, Approx.real_like h,
     Scalar.scalar c, Approx.real_like c,
     Scalar.scalar r, Approx.real_like r,
     Scalar.scalar k, Approx.real_like k,
     Scalar.scalar norm, Approx.real_like norm,
     Scalar.scalar score, Approx.real_like score |}
  (call:Model.invocation) (w:Model.parameters)
  (position:nat{position < Model.calls call}) (consumed:(D.token 248320))
  (incoming:A.hybrid_cache{Model.cache_shape position 0 incoming})
  (observed:model_observation m h c r k norm score)
  : Lemma
      (requires numerical call w position consumed incoming observed /\
                observed.cache_after == (observed.stages 24).cache)
      (ensures cache_approximates observed.cache_after
        (Model.model_step call w position consumed incoming).Model.cache) =
  let expected = Model.layers_at call w position consumed incoming 24 in
  assert (cache_approximates (observed.stages 24).cache expected.A.cache);
  Model.model_step_cache call w position consumed incoming

inline_for_extraction noextract
let numerical_residual_boundary (#m #h #c #r #k #norm #score:Type0)
  {| Scalar.scalar h, Approx.real_like h,
     Scalar.scalar c, Approx.real_like c,
     Scalar.scalar r, Approx.real_like r,
     Scalar.scalar k, Approx.real_like k,
     Scalar.scalar norm, Approx.real_like norm,
     Scalar.scalar score, Approx.real_like score |}
  (call:Model.invocation) (w:Model.parameters)
  (position:nat{position < Model.calls call}) (consumed:(D.token 248320))
  (incoming:A.hybrid_cache{Model.cache_shape position 0 incoming})
  (observed:model_observation m h c r k norm score)
  : Lemma
      (requires numerical call w position consumed incoming observed)
      (ensures
        vector_approximates (observed.stages 24).hidden
          (fun i -> (Model.model_step call w position consumed incoming).Model.residual i) /\
        vector_approximates observed.produced_scores
          (Model.model_step call w position consumed incoming).Model.logits) =
  let expected = Model.layers_at call w position consumed incoming 24 in
  assert (vector_approximates (observed.stages 24).hidden (fun i -> expected.A.hidden i));
  Model.model_step_residual call w position consumed incoming
