module KPlay.Proof.Optimized.Cache

module S = FStar.Seq
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module BF16 = Kuiper.BFloat16
module F32 = Kuiper.Float32
module Cast = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base
module Native = Pulse.Lib.Array
module Mem = Kuiper.Array.Core
module Storage = KPlay.Proof.Shared.Storage
module R = KPlay.Proof.Shared.Recurrent
module E = KPlay.Proof.Shared.Embedding
module Input = Kuiper.Tensor.Input
module Runtime = Kuiper.Tensor.Runtime
module NativeInput = KPlay.Proof.NativeInput
module I64 = FStar.Int64

inline_for_extraction noextract
let position = P.index 2048

inline_for_extraction noextract
let native_kv = s:S.seq BF16.t{S.length s == 6291456}

inline_for_extraction noextract
let rotated_keys = s:S.seq F32.t{S.length s == 512}

inline_for_extraction noextract
let projection_scratch = s:S.seq F32.t{S.length s == 8192}

inline_for_extraction noextract
let convolution_storage = s:S.seq F32.t{S.length s == 442368}

inline_for_extraction noextract
let recurrent_storage = s:S.seq F32.t{S.length s == 4718592}

inline_for_extraction noextract
let native_cache (metadata:Type0) = O.cache metadata F32.t F32.t BF16.t

inline_for_extraction noextract
noeq type kv_buffers = {
  keys: native_kv;
  values: native_kv;
}

inline_for_extraction noextract
let configuration : O.cache_configuration = {
  O.linear_layers = (fun l -> l % 4 < 3);
  O.transformer_layers = S.init #nat 6 (fun i -> 4 * i + 3);
  O.last_linear_layer = 22;
}

inline_for_extraction noextract
let kv_offset (slot:P.index 6) (head:P.index 2) (time:position)
  (coordinate:P.index 256) : P.index 6291456 =
  slot * 1048576 + head * 524288 + time * 256 + coordinate

inline_for_extraction noextract
let write_offset (layer:A.full_attention_layer) (time:position)
  (ordinal:P.index 512) : P.index 6291456 =
  kv_offset (layer / 4) (ordinal / 256) time (ordinal % 256)

inline_for_extraction noextract
let cell_ordinal (i:P.index 6291456) : P.index 512 =
  ((i % 1048576) / 524288) * 256 + i % 256

inline_for_extraction noextract
let current_row (layer:A.full_attention_layer) (time:position)
  (i:P.index 6291456) : bool =
  i / 1048576 = layer / 4 && (i % 524288) / 256 = time

inline_for_extraction noextract
let stored_key (normed:rotated_keys) (ordinal:P.index 512) : BF16.t =
  Cast.cast_f32_to_bf16 (S.index normed ordinal)

inline_for_extraction noextract
let stored_value (projected:projection_scratch) (ordinal:P.index 512) : BF16.t =
  Cast.cast_f32_to_bf16 (S.index projected (4608 + ordinal))

(* reduced.cu:253-268: h outer, i inner; key store precedes value store.
   normed records those source-produced FP32 scalars, not an ideal key vector. *)
inline_for_extraction noextract
let rec write_prefix (layer:A.full_attention_layer) (time:position)
  (normed:rotated_keys) (projected:projection_scratch) (old:kv_buffers)
  (count:nat{count <= 512}) : GTot kv_buffers (decreases count) =
  if count = 0 then old
  else
    let previous = write_prefix layer time normed projected old (count - 1) in
    let offset = write_offset layer time (count - 1) in
    let keys = S.upd previous.keys offset (stored_key normed (count - 1)) in
    let values = S.upd previous.values offset (stored_value projected (count - 1)) in
    { keys; values }

inline_for_extraction noextract
let history_view (slot:P.index 6) (length:pos{length <= 2048})
  (buffers:kv_buffers) : O.kv_history BF16.t = {
  O.length = length;
  O.keys = (fun h t d -> S.index buffers.keys (kv_offset slot h t d));
  O.values = (fun h t d -> S.index buffers.values (kv_offset slot h t d));
}

inline_for_extraction noextract
let old_history (layer:A.full_attention_layer) (time:position)
  (buffers:kv_buffers) : option (O.kv_history BF16.t) =
  if time = 0 then None else Some (history_view (layer / 4) time buffers)

inline_for_extraction noextract
let convolution_view (layer:Model.linear_layer) (buffer:convolution_storage)
  : O.matrix F32.t 6144 4 =
  fun channel tap ->
    S.index buffer (((layer / 4) * 3 + layer % 4) * 24576 + channel * 4 + tap)

inline_for_extraction noextract
let recurrent_view (layer:Model.linear_layer) (buffer:recurrent_storage)
  : O.tensor F32.t 16 128 128 =
  fun head key value ->
    S.index buffer
      (((layer / 4) * 3 + layer % 4) * 262144 + head * 16384 + value * 128 + key)

inline_for_extraction noextract
let cache_view (#m:Type0) (metadata:m) (time:position) (cursor:Model.cursor)
  (convolution:convolution_storage) (states:recurrent_storage) (buffers:kv_buffers)
  : GTot (native_cache m) = {
  O.metadata = metadata;
  O.convolution = (fun l ->
    if l % 4 < 3 && Model.visits time cursor l > 0
    then Some (convolution_view l convolution) else None);
  O.recurrent = (fun l ->
    if l % 4 < 3 && Model.visits time cursor l > 0
    then Some (recurrent_view l states) else None);
  O.kv = (fun l ->
    if l % 4 = 3 && Model.visits time cursor l > 0
    then Some (history_view (l / 4) (Model.visits time cursor l) buffers)
    else None);
}

inline_for_extraction noextract
let install_history (#m:Type0) (before:native_cache m)
  (layer:A.full_attention_layer) (next:O.kv_history BF16.t) : native_cache m = {
  O.metadata = before.metadata;
  O.convolution = before.convolution;
  O.recurrent = before.recurrent;
  O.kv = (fun l -> if l = layer then Some next else before.kv l);
}

inline_for_extraction noextract
let source_after (#m:Type0) (before:native_cache m)
  (layer:A.full_attention_layer) (time:position)
  (normed:rotated_keys) (projected:projection_scratch) (old:kv_buffers)
  : GTot (native_cache m) =
  install_history before layer
    (history_view (layer / 4) (time + 1)
      (write_prefix layer time normed projected old 512))

inline_for_extraction noextract
let source_update (normed:rotated_keys) (projected:projection_scratch)
  : O.layer_update F32.t F32.t BF16.t =
  O.AttentionUpdate
    (fun h d -> stored_key normed (h * 256 + d))
    (fun h d -> stored_value projected (h * 256 + d))

val configuration_checked : unit -> Lemma (O.configuration_ok configuration)

val offset_decoding :
  slot:P.index 6 -> head:P.index 2 -> time:position -> d:P.index 256 ->
  Lemma (
    let i = kv_offset slot head time d in
    i / 1048576 == slot /\
    (i % 1048576) / 524288 == head /\
    (i % 524288) / 256 == time /\
    i % 256 == d)

val write_offset_characterization :
  layer:A.full_attention_layer -> time:position -> ordinal:P.index 512 ->
  i:P.index 6291456 ->
  Lemma (
    (i == write_offset layer time ordinal) <==>
      (current_row layer time i /\ cell_ordinal i == ordinal))

val write_prefix_cell :
  layer:A.full_attention_layer -> time:position ->
  normed:rotated_keys -> projected:projection_scratch -> old:kv_buffers ->
  count:nat{count <= 512} -> i:P.index 6291456 ->
  Lemma (
    let next = write_prefix layer time normed projected old count in
    S.index next.keys i ==
      (if current_row layer time i && cell_ordinal i < count
       then stored_key normed (cell_ordinal i) else S.index old.keys i) /\
    S.index next.values i ==
      (if current_row layer time i && cell_ordinal i < count
       then stored_value projected (cell_ordinal i) else S.index old.values i))

val source_append_checked :
  layer:A.full_attention_layer -> time:position ->
  normed:rotated_keys -> projected:projection_scratch -> old:kv_buffers ->
  Lemma (
    O.kv_append time (old_history layer time old)
      (history_view (layer / 4) (time + 1)
        (write_prefix layer time normed projected old 512))
      (fun h d -> stored_key normed (h * 256 + d))
      (fun h d -> stored_value projected (h * 256 + d)))

val source_kv_transition :
  #m:Type0 -> before:native_cache m ->
  layer:A.full_attention_layer -> time:position ->
  normed:rotated_keys -> projected:projection_scratch -> old:kv_buffers ->
  Lemma
    (requires
      O.cache_shape time layer before /\
      before.kv layer == old_history layer time old)
    (ensures
      O.layer_structure time layer before
        (source_after before layer time normed projected old)
        (source_update normed projected) /\
      (forall (i:P.index 6291456).
        not (current_row layer time i) ==>
          S.index (write_prefix layer time normed projected old 512).keys i ==
            S.index old.keys i /\
          S.index (write_prefix layer time normed projected old 512).values i ==
            S.index old.values i))

val cache_view_shape :
  #m:Type0 -> metadata:m -> time:position -> cursor:Model.cursor ->
  convolution:convolution_storage -> states:recurrent_storage -> buffers:kv_buffers ->
  Lemma (O.cache_shape time cursor
    (cache_view metadata time cursor convolution states buffers))

val first_attention_cache_step :
  #m:Type0 -> metadata:m ->
  convolution:convolution_storage -> states:recurrent_storage -> old:kv_buffers ->
  normed:rotated_keys -> projected:projection_scratch ->
  Lemma (
    let before = cache_view metadata 0 3 convolution states old in
    before.kv 3 == None /\
    O.layer_structure 0 3 before
      (source_after before 3 0 normed projected old)
      (source_update normed projected))

val source_bound_cache_step :
  #m:Type0 -> metadata:m -> layer:A.full_attention_layer -> time:position ->
  convolution:convolution_storage -> states:recurrent_storage -> old:kv_buffers ->
  normed:rotated_keys -> projected:projection_scratch ->
  Lemma (
    let before = cache_view metadata time layer convolution states old in
    O.layer_structure time layer before
      (source_after before layer time normed projected old)
      (source_update normed projected))

val store_load_forward :
  produced:F32.t -> ideal:FStar.Real.real ->
  Lemma
    (requires Approx.v_approximates produced ideal)
    (ensures
      Approx.v_approximates (Cast.cast_f32_to_bf16 produced) ideal /\
      Approx.v_approximates
        (Cast.cast_bf16_to_f32 (Cast.cast_f32_to_bf16 produced)) ideal)

(* reduced.cu:295-296 and 307: consume the very cells written at 267-268. *)
val current_consumers_checked :
  layer:A.full_attention_layer -> time:position ->
  normed:rotated_keys -> projected:projection_scratch -> old:kv_buffers ->
  head:P.index 2 -> d:P.index 256 ->
  key_ideal:FStar.Real.real -> value_ideal:FStar.Real.real ->
  Lemma
    (requires
      Approx.v_approximates (S.index normed (head * 256 + d)) key_ideal /\
      Approx.v_approximates (S.index projected (4608 + head * 256 + d)) value_ideal)
    (ensures (
      let next = write_prefix layer time normed projected old 512 in
      let i = kv_offset (layer / 4) head time d in
      S.index next.keys i == stored_key normed (head * 256 + d) /\
      S.index next.values i == stored_value projected (head * 256 + d) /\
      Approx.v_approximates (S.index next.keys i) key_ideal /\
      Approx.v_approximates (S.index next.values i) value_ideal /\
      Approx.v_approximates (Cast.cast_bf16_to_f32 (S.index next.keys i)) key_ideal /\
      Approx.v_approximates (Cast.cast_bf16_to_f32 (S.index next.values i)) value_ideal))

val append_cell_refines_shared :
  layer:A.full_attention_layer -> time:position ->
  normed:rotated_keys -> projected:projection_scratch -> old:kv_buffers ->
  ideal_old:option A.kv_history{A.history_length ideal_old == time} ->
  ideal_key:A.matrix 2 256 -> ideal_value:A.matrix 2 256 ->
  head:P.index 2 -> t:nat{t <= time} -> d:P.index 256 ->
  Lemma
    (requires
      (forall (h:P.index 2) (coordinate:P.index 256).
        Approx.v_approximates (S.index normed (h * 256 + coordinate))
          (ideal_key h coordinate) /\
        Approx.v_approximates (S.index projected (4608 + h * 256 + coordinate))
          (ideal_value h coordinate)) /\
      (match ideal_old with
       | None -> True
       | Some previous ->
           forall (h:P.index 2) (p:nat{p < time /\ p < previous.A.length})
                  (coordinate:P.index 256).
             Approx.v_approximates
               (S.index old.keys (kv_offset (layer / 4) h p coordinate))
               (previous.A.keys h p coordinate) /\
             Approx.v_approximates
               (S.index old.values (kv_offset (layer / 4) h p coordinate))
               (previous.A.values h p coordinate)))
    (ensures (
      let actual = write_prefix layer time normed projected old 512 in
      let ideal = A.append_kv time ideal_old ideal_key ideal_value in
      let i = kv_offset (layer / 4) head t d in
      Approx.v_approximates (S.index actual.keys i) (ideal.A.keys head t d) /\
      Approx.v_approximates (S.index actual.values i) (ideal.A.values head t d)))

val recurrent_layout_checked :
  layer:Model.linear_layer -> head:P.index 16 ->
  key:P.index 128 -> value:P.index 128 ->
  Lemma (
    let slot = (layer / 4) * 3 + layer % 4 in
    let offset = slot * 262144 + head * 16384 + value * 128 + key in
    slot < 18 /\ offset < 4718592 /\
    offset / 262144 == slot /\
    (offset % 262144) / 16384 == head /\
    (offset % 16384) / 128 == value /\
    offset % 128 == key)

val convolution_layout_checked :
  layer:Model.linear_layer -> channel:P.index 6144 -> tap:P.index 4 ->
  Lemma (
    let slot = (layer / 4) * 3 + layer % 4 in
    let offset = slot * 24576 + channel * 4 + tap in
    slot < 18 /\ offset < 442368 /\
    offset / 24576 == slot /\
    (offset % 24576) / 4 == channel /\
    offset % 4 == tap)

inline_for_extraction noextract
let history_agrees (left right:option (O.kv_history BF16.t)) : GTot prop =
  match left, right with
  | None, None -> True
  | Some x, Some y ->
      x.O.length == y.O.length /\
      (forall (h:P.index 2) (t:nat{t < x.O.length /\ t < y.O.length})
              (d:P.index 256).
        x.O.keys h t d == y.O.keys h t d /\
        x.O.values h t d == y.O.values h t d)
  | _ -> False

(* Frame the original history objects, and separately identify every cell with
   the updated physical buffers. No equality of unrelated function closures. *)
inline_for_extraction noextract
let cache_agrees (#m:Type0) (left right:native_cache m) : GTot prop =
  left.metadata == right.metadata /\
  (forall (l:A.layer_index).
    left.convolution l == right.convolution l /\
    left.recurrent l == right.recurrent l /\
    history_agrees (left.kv l) (right.kv l))

val source_after_matches_buffers :
  #m:Type0 -> metadata:m -> layer:A.full_attention_layer -> time:position ->
  convolution:convolution_storage -> states:recurrent_storage -> old:kv_buffers ->
  normed:rotated_keys -> projected:projection_scratch ->
  Lemma (
    let before = cache_view metadata time layer convolution states old in
    let next = write_prefix layer time normed projected old 512 in
    cache_agrees (source_after before layer time normed projected old)
      (cache_view metadata time (layer + 1) convolution states next))

val source_bound_buffer_step :
  #m:Type0 -> metadata:m -> layer:A.full_attention_layer -> time:position ->
  convolution:convolution_storage -> states:recurrent_storage -> old:kv_buffers ->
  normed:rotated_keys -> projected:projection_scratch ->
  Lemma (
    let before = cache_view metadata time layer convolution states old in
    let after = source_after before layer time normed projected old in
    let next = write_prefix layer time normed projected old 512 in
    let physical = cache_view metadata time (layer + 1) convolution states next in
    O.layer_structure time layer before after (source_update normed projected) /\
    cache_agrees after physical /\
    O.cache_shape time (layer + 1) physical /\
    (forall (i:P.index 6291456).
      not (current_row layer time i) ==>
        S.index next.keys i == S.index old.keys i /\
        S.index next.values i == S.index old.values i))

(* reduced.cu:292-307 uses the grouped KV head, including the current row. *)
inline_for_extraction noextract
let attention_key_load (layer:A.full_attention_layer) (head:P.index 8)
  (time:position) (d:P.index 256) (buffers:kv_buffers) : F32.t =
  Cast.cast_bf16_to_f32
    (S.index buffers.keys (kv_offset (layer / 4) (head / 4) time d))

inline_for_extraction noextract
let attention_value_load (layer:A.full_attention_layer) (head:P.index 8)
  (time:position) (d:P.index 256) (buffers:kv_buffers) : F32.t =
  Cast.cast_bf16_to_f32
    (S.index buffers.values (kv_offset (layer / 4) (head / 4) time d))

val source_grouped_read_addresses :
  layer:A.full_attention_layer -> head:P.index 8 -> time:position ->
  lane:P.index 32 -> element:P.index 8 ->
  Lemma (
    let offset = (layer / 4) * 1048576 + (head / 4) * 524288 + time * 256 in
    offset + lane * 8 + element ==
      kv_offset (layer / 4) (A.kv_head head) time (lane * 8 + element) /\
    offset + lane * 8 + element < 6291456)

val source_current_read_values :
  layer:A.full_attention_layer -> time:position ->
  normed:rotated_keys -> projected:projection_scratch -> old:kv_buffers ->
  head:P.index 8 -> d:P.index 256 ->
  Lemma (
    let next = write_prefix layer time normed projected old 512 in
    attention_key_load layer head time d next ==
      Cast.cast_bf16_to_f32 (stored_key normed ((head / 4) * 256 + d)) /\
    attention_value_load layer head time d next ==
      Cast.cast_bf16_to_f32 (stored_value projected ((head / 4) * 256 + d)))

val source_current_reads_forward :
  layer:A.full_attention_layer -> time:position ->
  normed:rotated_keys -> projected:projection_scratch -> old:kv_buffers ->
  head:P.index 8 -> d:P.index 256 ->
  key_ideal:FStar.Real.real -> value_ideal:FStar.Real.real ->
  Lemma
    (requires
      Approx.v_approximates (S.index normed ((head / 4) * 256 + d)) key_ideal /\
      Approx.v_approximates
        (S.index projected (4608 + (head / 4) * 256 + d)) value_ideal)
    (ensures (
      let next = write_prefix layer time normed projected old 512 in
      Approx.v_approximates (attention_key_load layer head time d next) key_ideal /\
      Approx.v_approximates (attention_value_load layer head time d next) value_ideal))

val full_profile_buffer_step :
  #m:Type0 -> metadata:m -> call:Model.invocation ->
  time:nat{time < Model.calls call} -> layer:A.full_attention_layer ->
  convolution:convolution_storage -> states:recurrent_storage -> old:kv_buffers ->
  normed:rotated_keys -> projected:projection_scratch ->
  Lemma (
    time < 2048 /\
    (let before = cache_view metadata time layer convolution states old in
     let after = source_after before layer time normed projected old in
     let next = write_prefix layer time normed projected old 512 in
     O.layer_structure time layer before after (source_update normed projected) /\
     cache_agrees after
       (cache_view metadata time (layer + 1) convolution states next)))

(* One source-indexed transition supplies both consumer observations.
   Numerical premises guard only approximation, never the state or frames. *)
val full_profile_current_cache_step :
  #m:Type0 -> metadata:m -> call:Model.invocation ->
  time:nat{time < Model.calls call} -> layer:A.full_attention_layer ->
  convolution:convolution_storage -> states:recurrent_storage -> old:kv_buffers ->
  normed:rotated_keys -> projected:projection_scratch ->
  head:P.index 8 -> d:P.index 256 ->
  key_ideal:FStar.Real.real -> value_ideal:FStar.Real.real ->
  Lemma (
    time < 2048 /\
    (let before = cache_view metadata time layer convolution states old in
     let after = source_after before layer time normed projected old in
     let next = write_prefix layer time normed projected old 512 in
     let physical = cache_view metadata time (layer + 1) convolution states next in
     O.layer_structure time layer before after (source_update normed projected) /\
     cache_agrees after physical /\
     O.cache_shape time (layer + 1) physical /\
     (forall (i:P.index 6291456).
       not (current_row layer time i) ==>
         S.index next.keys i == S.index old.keys i /\
         S.index next.values i == S.index old.values i) /\
     attention_key_load layer head time d next ==
       Cast.cast_bf16_to_f32 (stored_key normed ((head / 4) * 256 + d)) /\
     attention_value_load layer head time d next ==
       Cast.cast_bf16_to_f32 (stored_value projected ((head / 4) * 256 + d)) /\
     ((Approx.v_approximates (S.index normed ((head / 4) * 256 + d)) key_ideal /\
       Approx.v_approximates
         (S.index projected (4608 + (head / 4) * 256 + d)) value_ideal) ==>
       Approx.v_approximates (attention_key_load layer head time d next) key_ideal /\
       Approx.v_approximates (attention_value_load layer head time d next) value_ideal)))

(* reduced.cu:291-307 enumerates every live row in exactly one partition. *)
val source_partition_coordinates :
  time:position -> read_time:nat{read_time <= time} ->
  lane:P.index 32 -> element:P.index 8 ->
  Lemma (
    read_time % 16 < 16 /\
    read_time % 16 + 16 * (read_time / 16) == read_time /\
    lane * 8 + element < 256 /\
    (lane * 8 + element) / 8 == lane /\
    (lane * 8 + element) % 8 == element /\
    (forall (partition:P.index 16) (iteration:nat).
      partition + 16 * iteration == read_time ==>
        partition == read_time % 16 /\ iteration == read_time / 16))

val source_history_read_values :
  layer:A.full_attention_layer -> time:position ->
  normed:rotated_keys -> projected:projection_scratch -> old:kv_buffers ->
  head:P.index 8 -> read_time:nat{read_time <= time} -> d:P.index 256 ->
  Lemma (
    let next = write_prefix layer time normed projected old 512 in
    attention_key_load layer head read_time d next ==
      (if read_time = time
       then Cast.cast_bf16_to_f32 (stored_key normed ((head / 4) * 256 + d))
       else attention_key_load layer head read_time d old) /\
    attention_value_load layer head read_time d next ==
      (if read_time = time
       then Cast.cast_bf16_to_f32 (stored_value projected ((head / 4) * 256 + d))
       else attention_value_load layer head read_time d old))

inline_for_extraction noextract
let cache_edge_inputs
  (layer:A.full_attention_layer) (time:position)
  (normed:rotated_keys) (projected:projection_scratch) (old:kv_buffers)
  (ideal_old:option A.kv_history{A.history_length ideal_old == time})
  (ideal_key ideal_value:A.matrix 2 256) : prop =
  (forall (h:P.index 2) (d:P.index 256).
    Approx.v_approximates (S.index normed (h * 256 + d)) (ideal_key h d) /\
    Approx.v_approximates (S.index projected (4608 + h * 256 + d))
      (ideal_value h d)) /\
  (match ideal_old with
   | None -> True
   | Some previous ->
       forall (h:P.index 2) (p:nat{p < time /\ p < previous.A.length})
              (d:P.index 256).
         Approx.v_approximates
           (S.index old.keys (kv_offset (layer / 4) h p d))
           (previous.A.keys h p d) /\
         Approx.v_approximates
           (S.index old.values (kv_offset (layer / 4) h p d))
           (previous.A.values h p d))

(* The incoming numerical invariant guards only the two forward-cast results.
   Both consumers read the same updated buffers used in the state conclusion. *)
val full_profile_partition_cache_step :
  #m:Type0 -> metadata:m -> call:Model.invocation ->
  time:nat{time < Model.calls call} -> layer:A.full_attention_layer ->
  convolution:convolution_storage -> states:recurrent_storage -> old:kv_buffers ->
  normed:rotated_keys -> projected:projection_scratch ->
  head:P.index 8 -> partition:P.index 16 ->
  iteration:nat{partition + 16 * iteration <= time} ->
  lane:P.index 32 -> element:P.index 8 ->
  ideal_old:option A.kv_history{A.history_length ideal_old == time} ->
  ideal_key:A.matrix 2 256 -> ideal_value:A.matrix 2 256 ->
  Lemma (
    time < 2048 /\
    (let read_time = partition + 16 * iteration in
     let d = lane * 8 + element in
     let before = cache_view metadata time layer convolution states old in
     let next = write_prefix layer time normed projected old 512 in
     let after = source_after before layer time normed projected old in
     let physical = cache_view metadata time (layer + 1) convolution states next in
     read_time % 16 == partition /\ read_time / 16 == iteration /\
     O.layer_structure time layer before after (source_update normed projected) /\
     cache_agrees after physical /\
     O.cache_shape time (layer + 1) physical /\
     (forall (i:P.index 6291456).
       not (current_row layer time i) ==>
         S.index next.keys i == S.index old.keys i /\
         S.index next.values i == S.index old.values i) /\
     attention_key_load layer head read_time d next ==
       (if read_time = time
        then Cast.cast_bf16_to_f32 (stored_key normed ((head / 4) * 256 + d))
        else attention_key_load layer head read_time d old) /\
     attention_value_load layer head read_time d next ==
       (if read_time = time
        then Cast.cast_bf16_to_f32 (stored_value projected ((head / 4) * 256 + d))
        else attention_value_load layer head read_time d old) /\
     (cache_edge_inputs layer time normed projected old ideal_old ideal_key ideal_value ==>
       (let expected = A.append_kv time ideal_old ideal_key ideal_value in
        Approx.v_approximates (attention_key_load layer head read_time d next)
          (expected.A.keys (A.kv_head head) read_time d) /\
        Approx.v_approximates (attention_value_load layer head read_time d next)
          (expected.A.values (A.kv_head head) read_time d)))))

(* These are the actual typed pointer extents at reduced.cu:399-402.
   Length/location metadata gives no cell permission or allocation ownership. *)
inline_for_extraction noextract
noeq type native_buffers = {
  convolution_buffer: b:Native.array F32.t{
    Native.length b == 442368 /\ Mem.is_global_array b};
  recurrent_buffer: b:Native.array F32.t{
    Native.length b == 4718592 /\ Mem.is_global_array b};
  key_buffer: b:Native.array BF16.t{
    Native.length b == 6291456 /\ Mem.is_global_array b};
  value_buffer: b:Native.array BF16.t{
    Native.length b == 6291456 /\ Mem.is_global_array b};
}

inline_for_extraction noextract
let native_buffers_from_allocations
  (states:Runtime.tensor F32.t{Runtime.metadata states [18; 16; 128; 128]})
  (convolution:Runtime.tensor F32.t{Runtime.metadata convolution [18; 6144; 4]})
  (keys:Runtime.tensor BF16.t{Runtime.metadata keys [6; 2; 2048; 256]})
  (values:Runtime.tensor BF16.t{Runtime.metadata values [6; 2; 2048; 256]})
  : GTot native_buffers = {
  convolution_buffer = Runtime.data convolution;
  recurrent_buffer = Runtime.data states;
  key_buffer = Runtime.data keys;
  value_buffer = Runtime.data values;
}

(* Pair this map with Storage.cache_owns of the SAME cache_view. In particular,
   a sequence snapshot alone is not a statement about initialized GPU cells. *)
inline_for_extraction noextract
let cache_addresses (#m:Type0) (native:native_buffers)
  (metadata:m) (time:position) (cursor:Model.cursor)
  (convolution:convolution_storage) (states:recurrent_storage) (buffers:kv_buffers)
  : GTot (Storage.cache_storage
      (cache_view metadata time cursor convolution states buffers)) = {
  Storage.convolution = (fun l channel tap -> {
    Storage.buffer = native.convolution_buffer;
    Storage.offset =
      ((l / 4) * 3 + l % 4) * 24576 + channel * 4 + tap;
  });
  Storage.recurrent = (fun l head key value -> {
    Storage.buffer = native.recurrent_buffer;
    Storage.offset =
      ((l / 4) * 3 + l % 4) * 262144 + head * 16384 + value * 128 + key;
  });
  Storage.keys = (fun l head t d -> {
    Storage.buffer = native.key_buffer;
    Storage.offset = kv_offset (l / 4) head t d;
  });
  Storage.values = (fun l head t d -> {
    Storage.buffer = native.value_buffer;
    Storage.offset = kv_offset (l / 4) head t d;
  });
}

val native_cache_cell_binding :
  #m:Type0 -> native:native_buffers -> metadata:m ->
  layer:A.full_attention_layer -> time:position ->
  convolution:convolution_storage -> states:recurrent_storage -> buffers:kv_buffers ->
  head:P.index 8 -> read_time:nat{read_time <= time} -> d:P.index 256 ->
  Lemma (
    let snapshot = cache_view metadata time (layer + 1) convolution states buffers in
    let at = cache_addresses native metadata time (layer + 1) convolution states buffers in
    let history = R.present_value (snapshot.kv layer) in
    let key = at.keys layer (head / 4) read_time d in
    let value = at.values layer (head / 4) read_time d in
    key.buffer == native.key_buffer /\ value.buffer == native.value_buffer /\
    key.offset == kv_offset (layer / 4) (head / 4) read_time d /\
    value.offset == key.offset /\
    history.keys (head / 4) read_time d == S.index buffers.keys key.offset /\
    history.values (head / 4) read_time d == S.index buffers.values value.offset /\
    attention_key_load layer head read_time d buffers ==
      Cast.cast_bf16_to_f32 (history.keys (head / 4) read_time d) /\
    attention_value_load layer head read_time d buffers ==
      Cast.cast_bf16_to_f32 (history.values (head / 4) read_time d))

val first_attention_native_storage_step :
  #m:Type0 -> native:native_buffers -> metadata:m ->
  convolution:convolution_storage -> states:recurrent_storage -> old:kv_buffers ->
  normed:rotated_keys -> projected:projection_scratch ->
  head:P.index 8 -> d:P.index 256 ->
  Lemma (
    let before = cache_view metadata 0 3 convolution states old in
    let next = write_prefix 3 0 normed projected old 512 in
    let after = source_after before 3 0 normed projected old in
    let physical = cache_view metadata 0 4 convolution states next in
    let at = cache_addresses native metadata 0 4 convolution states next in
    before.kv 3 == None /\
    O.layer_structure 0 3 before after (source_update normed projected) /\
    cache_agrees after physical /\
    O.cache_shape 0 4 physical /\
    (at.keys 3 (head / 4) 0 d).buffer == native.key_buffer /\
    (at.values 3 (head / 4) 0 d).buffer == native.value_buffer /\
    S.index next.keys (at.keys 3 (head / 4) 0 d).offset ==
      stored_key normed ((head / 4) * 256 + d) /\
    S.index next.values (at.values 3 (head / 4) 0 d).offset ==
      stored_value projected ((head / 4) * 256 + d) /\
    (forall (i:P.index 6291456).
      not (current_row 3 0 i) ==>
        S.index next.keys i == S.index old.keys i /\
        S.index next.values i == S.index old.values i))

(* reduced.cu:181 and 362 use the same tied BF16 input cells. This is a cell
   map, not another weight allocation or an interpretation of intermediate data. *)
inline_for_extraction noextract
let input_embedding_row (#p:pos{p <= 520}) (inputs:NativeInput.arguments p)
  (token:E.token_id) : GTot (E.hidden_vector (Storage.cell BF16.t)) =
  S.init_ghost 1024 (fun d -> NativeInput.cell inputs.weights (token * 1024 + d))

inline_for_extraction noextract
let input_embedding_cells (#p:pos{p <= 520}) (inputs:NativeInput.arguments p)
  : GTot (E.embedding_weights (Storage.cell BF16.t)) =
  S.init_ghost #(E.hidden_vector (Storage.cell BF16.t)) 248320
    (input_embedding_row inputs)

val input_embedding_projection_binding :
  #p:pos{p <= 520} -> inputs:NativeInput.arguments p ->
  token:E.token_id -> d:E.hidden_index ->
  Lemma (
    let at = S.index (E.embedding (input_embedding_cells inputs) token) d in
    at == NativeInput.cell inputs.weights (token * 1024 + d) /\
    at.buffer == Input.data inputs.weights /\
    at.offset == token * 1024 + d /\
    Native.base_of at.buffer == Native.base_of (Input.backing inputs.weights) /\
    Native.offset_of at.buffer + at.offset ==
      Native.offset_of (Input.backing inputs.weights) + Input.offset inputs.weights +
        token * 1024 + d)

val source_pointer_cells_checked :
  #p:pos{p <= 520} -> inputs:NativeInput.arguments p ->
  addresses:(Native.array BF16.t & Native.array I64.t & I64.t) ->
  token:E.token_id -> d:E.hidden_index -> prompt_index:P.index p ->
  Lemma
    (requires NativeInput.pointer_result inputs addresses)
    (ensures (
      let weight = S.index (E.embedding (input_embedding_cells inputs) token) d in
      let prompt = NativeInput.cell inputs.prompt prompt_index in
      weight.buffer == addresses._1 /\ prompt.buffer == addresses._2 /\
      I64.v addresses._3 == p /\
      weight.offset == token * 1024 + d /\ prompt.offset == prompt_index /\
      Native.offset_of weight.buffer + weight.offset ==
        Native.offset_of (Input.backing inputs.weights) + Input.offset inputs.weights +
          token * 1024 + d /\
      Native.offset_of prompt.buffer + prompt.offset ==
        Native.offset_of (Input.backing inputs.prompt) + Input.offset inputs.prompt +
          prompt_index))
