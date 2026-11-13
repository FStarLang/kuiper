module KPlay.Proof.Optimized.AttentionMath

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open FStar.Real
open KPlay.Proof.Shared.Attention
module A = KPlay.Proof.Shared.Attention
module C = FStar.Classical
module Root = FStar.Math.Sqrt

inline_for_extraction noextract unfold
let equal_vector (#n:nat) (x y:A.vector n) : prop =
  forall (i:A.index n). x i == y i

inline_for_extraction noextract unfold
let equal_history (x y:option A.kv_history) : prop =
  match x, y with
  | None, None -> True
  | Some a, Some b ->
      a.length == b.length /\
      (forall (h:A.index 2) (t:nat{t < a.length /\ t < b.length}) (d:A.index 256).
        a.keys h t d == b.keys h t d /\ a.values h t d == b.values h t d)
  | _ -> False

let rec sum_ext (#n:nat) (x y:A.vector n) (count:nat{count <= n})
  : Lemma
      (requires forall (i:nat{i < count}). x i == y i)
      (ensures A.sum count x == A.sum count y)
      (decreases count) =
  FStar.Pervasives.norm_spec [delta_once ["KPlay.Proof.Shared.Attention.sum"]] (A.sum count x);
  FStar.Pervasives.norm_spec [delta_once ["KPlay.Proof.Shared.Attention.sum"]] (A.sum count y);
  if count > 0 then
    sum_ext #(count - 1) (fun i -> x i) (fun i -> y i) (count - 1)

let rec squares_ext (#n:nat) (x y:A.vector n) (count:nat{count <= n})
  : Lemma
      (requires forall (i:nat{i < count}). x i == y i)
      (ensures A.sum_squares count x == A.sum_squares count y)
      (decreases count) =
  FStar.Pervasives.norm_spec [delta_once ["KPlay.Proof.Shared.Attention.sum_squares"]] (A.sum_squares count x);
  FStar.Pervasives.norm_spec [delta_once ["KPlay.Proof.Shared.Attention.sum_squares"]] (A.sum_squares count y);
  if count > 0 then
    squares_ext #(count - 1) (fun i -> x i) (fun i -> y i) (count - 1)

let rec positive_ext (#n:pos) (x y:(A.index n -> Root.rpos)) (count:pos{count <= n})
  : Lemma
      (requires forall (i:nat{i < count}). x i == y i)
      (ensures A.sum_positive count x == A.sum_positive count y)
      (decreases count) =
  FStar.Pervasives.norm_spec [delta_once ["KPlay.Proof.Shared.Attention.sum_positive"]] (A.sum_positive count x);
  FStar.Pervasives.norm_spec [delta_once ["KPlay.Proof.Shared.Attention.sum_positive"]] (A.sum_positive count y);
  if count > 1 then
    positive_ext #(count - 1) (fun i -> x i) (fun i -> y i) (count - 1)

let linear_ext (rows columns:nat) (w:A.matrix rows columns) (x y:A.vector columns)
  : Lemma
      (requires equal_vector x y)
      (ensures equal_vector (A.linear rows columns w x) (A.linear rows columns w y)) =
  let row (r:A.index rows)
    : Lemma (A.linear rows columns w x r == A.linear rows columns w y r) =
    sum_ext (fun i -> w r i *. x i) (fun i -> w r i *. y i) columns
  in C.forall_intro row

let rms_ext (n:pos) (w x y:A.vector n)
  : Lemma
      (requires equal_vector x y)
      (ensures equal_vector (A.rms_norm n w x) (A.rms_norm n w y)) =
  squares_ext x y n

let softmax_ext (n:pos) (x y:A.vector n)
  : Lemma
      (requires equal_vector x y)
      (ensures equal_vector (A.softmax n x) (A.softmax n y)) =
  let ex (i:A.index n) : Root.rpos =
    FStar.Math.Exp.exp_positive (x i); FStar.Math.Exp.exp (x i) in
  let ey (i:A.index n) : Root.rpos =
    FStar.Math.Exp.exp_positive (y i); FStar.Math.Exp.exp (y i) in
  positive_ext ex ey n

inline_for_extraction noextract
let score_terms (position:nat) (query:A.matrix 8 256)
  (history:A.kv_history{history.length == position + 1}) (h:A.index 8)
  : A.vector (position + 1) =
  fun t ->
    A.sum 256 (fun d -> query h d *. history.keys (A.kv_head h) t d) *.
      (1.0R /. 16.0R) +. A.causal_bias position t

inline_for_extraction noextract
let value_terms (position:nat) (query:A.matrix 8 256)
  (history:A.kv_history{history.length == position + 1}) (h:A.index 8) (d:A.index 256)
  : A.vector (position + 1) =
  fun t -> A.softmax (position + 1) (score_terms position query history h) t *.
    history.values (A.kv_head h) t d

let attention_form (position:nat) (query:A.matrix 8 256)
  (history:A.kv_history{history.length == position + 1}) (h:A.index 8) (d:A.index 256)
  : Lemma (
      A.attention position query history h d ==
        A.sum (position + 1) (value_terms position query history h d)) =
  assert (
    A.attention position query history h d ==
      A.sum (position + 1) (value_terms position query history h d))
  by (FStar.Tactics.V2.norm [delta_only [
        `%A.attention; `%score_terms; `%value_terms]; zeta; iota];
      FStar.Tactics.V2.trefl ())

#push-options "--fuel 0 --ifuel 0 --z3rlimit 60"
let attention_ext (position:nat) (query:A.matrix 8 256)
  (x:A.kv_history{x.length == position + 1})
  (y:A.kv_history{y.length == position + 1})
  : Lemma
      (requires equal_history (Some x) (Some y))
      (ensures forall (h:A.index 8) (d:A.index 256).
        A.attention position query x h d == A.attention position query y h d) =
  let head (h:A.index 8) : Lemma (
      forall (d:A.index 256).
        A.attention position query x h d == A.attention position query y h d) =
    let sx = score_terms position query x h in
    let sy = score_terms position query y h in
    let score (t:A.index (position + 1)) : Lemma (sx t == sy t) =
      sum_ext #256
        (fun d -> query h d *. x.keys (A.kv_head h) t d)
        (fun d -> query h d *. y.keys (A.kv_head h) t d) 256;
      FStar.Pervasives.norm_spec [delta_only [`%score_terms]; zeta; iota]
        (score_terms position query x h t);
      FStar.Pervasives.norm_spec [delta_only [`%score_terms]; zeta; iota]
        (score_terms position query y h t)
    in
    C.forall_intro score;
    softmax_ext (position + 1) sx sy;
    let cell (d:A.index 256) : Lemma (
        A.attention position query x h d == A.attention position query y h d) =
      attention_form position query x h d;
      attention_form position query y h d;
      FStar.Pervasives.norm_spec [delta_only [`%value_terms]; zeta; iota]
        (value_terms position query x h d);
      FStar.Pervasives.norm_spec [delta_only [`%value_terms]; zeta; iota]
        (value_terms position query y h d);
      sum_ext #(position + 1) (value_terms position query x h d)
        (value_terms position query y h d) (position + 1)
    in C.forall_intro cell
  in C.forall_intro head
#pop-options

let append_ext (position:nat)
  (x:option A.kv_history{A.history_length x == position})
  (y:option A.kv_history{A.history_length y == position})
  (key value:A.matrix 2 256)
  : Lemma
      (requires equal_history x y)
      (ensures equal_history
        (Some (A.append_kv position x key value)) (Some (A.append_kv position y key value))) =
  match x, y with
  | None, None -> ()
  | Some _, Some _ -> ()
  | _ -> ()

inline_for_extraction noextract
let decoder_queries (call:A.invocation) (w:A.layer_weights) (input:A.vector 1024)
  : A.matrix 8 256 =
  let normalized = A.rms_norm 1024 w.input_norm input in
  let projected = A.linear 4096 1024 w.query_gate_proj normalized in
  let queries = A.split_queries projected in
  fun h -> A.rotate_partial call.position (A.rms_norm 256 w.query_norm (queries h))

inline_for_extraction noextract
let decoder_next (call:A.invocation) (w:A.layer_weights) (input:A.vector 1024)
  (before:A.hybrid_cache{A.domain call before})
  : h:A.kv_history{h.length == call.position + 1} =
  let normalized = A.rms_norm 1024 w.input_norm input in
  let keys = A.reshape_kv (A.linear 512 1024 w.key_proj normalized) in
  let values = A.reshape_kv (A.linear 512 1024 w.value_proj normalized) in
  let rotated_keys : A.matrix 2 256 =
    fun h -> A.rotate_partial call.position (A.rms_norm 256 w.key_norm (keys h)) in
  A.append_kv call.position (before.kv call.layer) rotated_keys values

inline_for_extraction noextract
let decoder_gated (call:A.invocation) (w:A.layer_weights) (input:A.vector 1024)
  (before:A.hybrid_cache{A.domain call before}) : A.vector 2048 =
  let normalized = A.rms_norm 1024 w.input_norm input in
  let gates = A.split_gates (A.linear 4096 1024 w.query_gate_proj normalized) in
  let context = A.attention call.position (decoder_queries call w input)
    (decoder_next call w input before) in
  fun j -> context (j / 256) (j % 256) *. A.sigmoid (gates (j / 256) (j % 256))

inline_for_extraction noextract
let decoder_residual (call:A.invocation) (w:A.layer_weights) (input:A.vector 1024)
  (before:A.hybrid_cache{A.domain call before}) : A.vector 1024 =
  A.vector_add 1024 input (A.linear 1024 2048 w.output_proj (decoder_gated call w input before))

inline_for_extraction noextract
let decoder_product (call:A.invocation) (w:A.layer_weights) (input:A.vector 1024)
  (before:A.hybrid_cache{A.domain call before}) : A.vector 3584 =
  let normalized = A.rms_norm 1024 w.post_attention_norm (decoder_residual call w input before) in
  fun i -> A.silu (A.linear 3584 1024 w.mlp_gate_proj normalized i) *.
    A.linear 3584 1024 w.mlp_up_proj normalized i

#push-options "--fuel 0 --ifuel 0 --z3rlimit 60"
let decoder_form (call:A.invocation) (w:A.layer_weights) (input:A.vector 1024)
  (before:A.hybrid_cache{A.domain call before})
  : Lemma (
      (A.full_attention_decoder call w input before).hidden ==
        A.vector_add 1024 (decoder_residual call w input before)
          (A.linear 1024 3584 w.mlp_down_proj (decoder_product call w input before)) /\
      (A.full_attention_decoder call w input before).cache.kv call.layer ==
        Some (decoder_next call w input before)) =
  assert (
    (A.full_attention_decoder call w input before).hidden ==
      A.vector_add 1024 (decoder_residual call w input before)
        (A.linear 1024 3584 w.mlp_down_proj (decoder_product call w input before)))
  by (FStar.Tactics.V2.norm [delta_only [
        `%A.full_attention_decoder; `%decoder_queries; `%decoder_next;
        `%decoder_gated; `%decoder_residual; `%decoder_product]; zeta; iota];
      FStar.Tactics.V2.trefl ());
  FStar.Pervasives.norm_spec [delta_only [
    `%A.full_attention_decoder; `%A.replace_kv; `%decoder_next]; zeta; iota]
    ((A.full_attention_decoder call w input before).cache.kv call.layer)

let decoder_next_ext (call:A.invocation) (w:A.layer_weights) (input:A.vector 1024)
  (x:A.hybrid_cache{A.domain call x}) (y:A.hybrid_cache{A.domain call y})
  : Lemma
      (requires equal_history (x.kv call.layer) (y.kv call.layer))
      (ensures equal_history (Some (decoder_next call w input x))
        (Some (decoder_next call w input y))) =
  FStar.Pervasives.norm_spec [delta_only [`%A.domain]; zeta; iota] (A.domain call x);
  FStar.Pervasives.norm_spec [delta_only [`%A.domain]; zeta; iota] (A.domain call y);
  let normalized = A.rms_norm 1024 w.input_norm input in
  let keys = A.reshape_kv (A.linear 512 1024 w.key_proj normalized) in
  let values = A.reshape_kv (A.linear 512 1024 w.value_proj normalized) in
  let rotated_keys : A.matrix 2 256 =
    fun h -> A.rotate_partial call.position (A.rms_norm 256 w.key_norm (keys h)) in
  append_ext call.position (x.kv call.layer) (y.kv call.layer) rotated_keys values;
  FStar.Pervasives.norm_spec [delta_only [`%decoder_next]; zeta; iota] (decoder_next call w input x);
  FStar.Pervasives.norm_spec [delta_only [`%decoder_next]; zeta; iota] (decoder_next call w input y)

let decoder_history_ext (call:A.invocation) (w:A.layer_weights) (input:A.vector 1024)
  (x:A.hybrid_cache{A.domain call x}) (y:A.hybrid_cache{A.domain call y})
  : Lemma
      (requires equal_history (x.kv call.layer) (y.kv call.layer))
      (ensures
        equal_vector (A.full_attention_decoder call w input x).hidden
          (A.full_attention_decoder call w input y).hidden /\
        equal_history ((A.full_attention_decoder call w input x).cache.kv call.layer)
          ((A.full_attention_decoder call w input y).cache.kv call.layer)) =
  decoder_form call w input x;
  decoder_form call w input y;
  decoder_next_ext call w input x y;
  attention_ext call.position (decoder_queries call w input)
    (decoder_next call w input x) (decoder_next call w input y);
  let gated_cell (i:A.index 2048) : Lemma (
      decoder_gated call w input x i == decoder_gated call w input y i) =
    FStar.Pervasives.norm_spec [delta_only [`%decoder_gated]; zeta; iota] (decoder_gated call w input x i);
    FStar.Pervasives.norm_spec [delta_only [`%decoder_gated]; zeta; iota] (decoder_gated call w input y i)
  in
  C.forall_intro gated_cell;
  linear_ext 1024 2048 w.output_proj (decoder_gated call w input x) (decoder_gated call w input y);
  let residual_cell (i:A.index 1024) : Lemma (
      decoder_residual call w input x i == decoder_residual call w input y i) =
    FStar.Pervasives.norm_spec [delta_only [`%decoder_residual; `%A.vector_add]; zeta; iota]
      (decoder_residual call w input x i);
    FStar.Pervasives.norm_spec [delta_only [`%decoder_residual; `%A.vector_add]; zeta; iota]
      (decoder_residual call w input y i)
  in
  C.forall_intro residual_cell;
  rms_ext 1024 w.post_attention_norm (decoder_residual call w input x) (decoder_residual call w input y);
  let mlp_x = A.rms_norm 1024 w.post_attention_norm (decoder_residual call w input x) in
  let mlp_y = A.rms_norm 1024 w.post_attention_norm (decoder_residual call w input y) in
  linear_ext 3584 1024 w.mlp_gate_proj mlp_x mlp_y;
  linear_ext 3584 1024 w.mlp_up_proj mlp_x mlp_y;
  let product_cell (i:A.index 3584) : Lemma (
      decoder_product call w input x i == decoder_product call w input y i) =
    FStar.Pervasives.norm_spec [delta_only [`%decoder_product]; zeta; iota] (decoder_product call w input x i);
    FStar.Pervasives.norm_spec [delta_only [`%decoder_product]; zeta; iota] (decoder_product call w input y i)
  in
  C.forall_intro product_cell;
  linear_ext 1024 3584 w.mlp_down_proj (decoder_product call w input x) (decoder_product call w input y);
  let cell (i:A.index 1024) : Lemma (
      (A.full_attention_decoder call w input x).hidden i ==
        (A.full_attention_decoder call w input y).hidden i) =
    FStar.Pervasives.norm_spec [delta_only [`%A.vector_add]; zeta; iota]
      (A.vector_add 1024 (decoder_residual call w input x)
        (A.linear 1024 3584 w.mlp_down_proj (decoder_product call w input x)) i);
    FStar.Pervasives.norm_spec [delta_only [`%A.vector_add]; zeta; iota]
      (A.vector_add 1024 (decoder_residual call w input y)
        (A.linear 1024 3584 w.mlp_down_proj (decoder_product call w input y)) i)
  in C.forall_intro cell
#pop-options
