module KPlay.Proof.Optimized.Cache

module C = FStar.Classical
module S = FStar.Seq
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Cast = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base

let configuration_checked () = ()

let offset_decoding slot head time d = ()

let write_offset_characterization layer time ordinal i =
  offset_decoding (layer / 4) (ordinal / 256) time (ordinal % 256);
  assert (i ==
    (i / 1048576) * 1048576 +
    ((i % 1048576) / 524288) * 524288 +
    ((i % 524288) / 256) * 256 + i % 256)

let rec write_prefix_cell (layer:A.full_attention_layer) (time:position)
  (normed:rotated_keys) (projected:projection_scratch) (old:kv_buffers)
  (count:nat{count <= 512}) (i:P.index 6291456)
  : Lemma
      (ensures (
        let next = write_prefix layer time normed projected old count in
        S.index next.keys i ==
          (if current_row layer time i && cell_ordinal i < count
           then stored_key normed (cell_ordinal i) else S.index old.keys i) /\
        S.index next.values i ==
          (if current_row layer time i && cell_ordinal i < count
           then stored_value projected (cell_ordinal i) else S.index old.values i)))
      (decreases count) =
  if count > 0 then (
    write_prefix_cell layer time normed projected old (count - 1) i;
    write_offset_characterization layer time (count - 1) i
  )

let source_append_checked layer time normed projected old =
  let current (h:P.index 2) (d:P.index 256)
    : Lemma (
        let next = write_prefix layer time normed projected old 512 in
        S.index next.keys (kv_offset (layer / 4) h time d) ==
          stored_key normed (h * 256 + d) /\
        S.index next.values (kv_offset (layer / 4) h time d) ==
          stored_value projected (h * 256 + d)) =
    offset_decoding (layer / 4) h time d;
    write_prefix_cell layer time normed projected old 512
      (kv_offset (layer / 4) h time d)
  in
  C.forall_intro_2 current;
  let prefix (h:P.index 2) (t:nat{t < time}) (d:P.index 256)
    : Lemma (
        let next = write_prefix layer time normed projected old 512 in
        S.index next.keys (kv_offset (layer / 4) h t d) ==
          S.index old.keys (kv_offset (layer / 4) h t d) /\
        S.index next.values (kv_offset (layer / 4) h t d) ==
          S.index old.values (kv_offset (layer / 4) h t d)) =
    offset_decoding (layer / 4) h t d;
    write_prefix_cell layer time normed projected old 512
      (kv_offset (layer / 4) h t d)
  in
  C.forall_intro_3 prefix

let source_kv_transition #m before layer time normed projected old =
  source_append_checked layer time normed projected old;
  let frame (i:P.index 6291456)
    : Lemma (
        not (current_row layer time i) ==>
          S.index (write_prefix layer time normed projected old 512).keys i ==
            S.index old.keys i /\
          S.index (write_prefix layer time normed projected old 512).values i ==
            S.index old.values i) =
    write_prefix_cell layer time normed projected old 512 i
  in
  C.forall_intro frame;
  let shape (l:A.layer_index)
    : Lemma (
        let after = source_after before layer time normed projected old in
        let count = Model.visits time (layer + 1) l in
        if l % 4 < 3 then
          after.kv l == None /\
          (Some? (after.convolution l) <==> count > 0) /\
          (Some? (after.recurrent l) <==> count > 0)
        else
          after.convolution l == None /\
          after.recurrent l == None /\
          O.history_length (after.kv l) == count) =
    assert (O.cache_shape time layer before)
  in
  C.forall_intro shape

let cache_view_shape #m metadata time cursor convolution states buffers = ()

let first_attention_cache_step #m metadata convolution states old normed projected =
  cache_view_shape metadata 0 3 convolution states old;
  source_kv_transition (cache_view metadata 0 3 convolution states old)
    3 0 normed projected old

let source_bound_cache_step #m metadata layer time convolution states old normed projected =
  cache_view_shape metadata time layer convolution states old;
  source_kv_transition (cache_view metadata time layer convolution states old)
    layer time normed projected old

let store_load_forward produced ideal =
  Cast.cast_f32_to_bf16_ok produced ideal;
  Cast.cast_bf16_to_f32_ok (Cast.cast_f32_to_bf16 produced) ideal

let current_consumers_checked layer time normed projected old head d key_ideal value_ideal =
  offset_decoding (layer / 4) head time d;
  write_prefix_cell layer time normed projected old 512
    (kv_offset (layer / 4) head time d);
  store_load_forward (S.index normed (head * 256 + d)) key_ideal;
  store_load_forward (S.index projected (4608 + head * 256 + d)) value_ideal

let append_cell_refines_shared layer time normed projected old
  ideal_old ideal_key ideal_value head t d =
  if t = time then (
    current_consumers_checked layer time normed projected old head d
      (ideal_key head d) (ideal_value head d);
    A.append_includes_current time ideal_old ideal_key ideal_value head d
  ) else (
    offset_decoding (layer / 4) head t d;
    write_prefix_cell layer time normed projected old 512
      (kv_offset (layer / 4) head t d);
    match ideal_old with
    | None -> ()
    | Some previous ->
        A.append_preserves_prefix previous ideal_key ideal_value head t d
  )

let recurrent_layout_checked layer head key value = ()

let convolution_layout_checked layer channel tap = ()

let source_after_matches_buffers #m metadata layer time convolution states old normed projected =
  let before = cache_view metadata time layer convolution states old in
  let next = write_prefix layer time normed projected old 512 in
  let after = source_after before layer time normed projected old in
  let physical = cache_view metadata time (layer + 1) convolution states next in
  let each_layer (l:A.layer_index)
    : Lemma (
        after.convolution l == physical.convolution l /\
        after.recurrent l == physical.recurrent l /\
        history_agrees (after.kv l) (physical.kv l)) =
    if l % 4 = 3 && l <> layer && Model.visits time layer l > 0 then (
      let count = Model.visits time layer l in
      let each_cell (h:P.index 2) (t:nat{t < count}) (d:P.index 256)
        : Lemma (
            S.index next.keys (kv_offset (l / 4) h t d) ==
              S.index old.keys (kv_offset (l / 4) h t d) /\
            S.index next.values (kv_offset (l / 4) h t d) ==
              S.index old.values (kv_offset (l / 4) h t d)) =
        offset_decoding (l / 4) h t d;
        write_prefix_cell layer time normed projected old 512
          (kv_offset (l / 4) h t d)
      in
      C.forall_intro_3 each_cell
    )
  in
  C.forall_intro each_layer

let source_bound_buffer_step #m metadata layer time convolution states old normed projected =
  cache_view_shape metadata time layer convolution states old;
  source_kv_transition (cache_view metadata time layer convolution states old)
    layer time normed projected old;
  source_after_matches_buffers metadata layer time convolution states old normed projected;
  cache_view_shape metadata time (layer + 1) convolution states
    (write_prefix layer time normed projected old 512)

let source_grouped_read_addresses layer head time lane element = ()

let source_current_read_values layer time normed projected old head d =
  offset_decoding (layer / 4) (head / 4) time d;
  write_prefix_cell layer time normed projected old 512
    (kv_offset (layer / 4) (head / 4) time d)

let source_current_reads_forward layer time normed projected old head d key_ideal value_ideal =
  current_consumers_checked layer time normed projected old (head / 4) d key_ideal value_ideal

let full_profile_buffer_step #m metadata call time layer convolution states old normed projected =
  Model.full_profile call;
  source_bound_buffer_step metadata layer time convolution states old normed projected

let full_profile_current_cache_step #m metadata call time layer
  convolution states old normed projected head d key_ideal value_ideal =
  Model.full_profile call;
  source_bound_buffer_step metadata layer time convolution states old normed projected;
  source_current_read_values layer time normed projected old head d;
  let numerical (_:(
      Approx.v_approximates (S.index normed ((head / 4) * 256 + d)) key_ideal /\
      Approx.v_approximates
        (S.index projected (4608 + (head / 4) * 256 + d)) value_ideal))
    : Lemma (
        let next = write_prefix layer time normed projected old 512 in
        Approx.v_approximates (attention_key_load layer head time d next) key_ideal /\
        Approx.v_approximates (attention_value_load layer head time d next) value_ideal) =
    source_current_reads_forward layer time normed projected old head d key_ideal value_ideal
  in
  C.impl_intro numerical

let source_partition_coordinates time read_time lane element = ()

let source_history_read_values layer time normed projected old head read_time d =
  offset_decoding (layer / 4) (head / 4) read_time d;
  write_prefix_cell layer time normed projected old 512
    (kv_offset (layer / 4) (head / 4) read_time d)

let full_profile_partition_cache_step #m metadata call time layer
  convolution states old normed projected head partition iteration lane element
  ideal_old ideal_key ideal_value =
  Model.full_profile call;
  let read_time = partition + 16 * iteration in
  let d = lane * 8 + element in
  source_partition_coordinates time read_time lane element;
  source_bound_buffer_step metadata layer time convolution states old normed projected;
  source_history_read_values layer time normed projected old head read_time d;
  let numerical (_:cache_edge_inputs layer time normed projected old
      ideal_old ideal_key ideal_value)
    : Lemma (
        let next = write_prefix layer time normed projected old 512 in
        let expected = A.append_kv time ideal_old ideal_key ideal_value in
        Approx.v_approximates (attention_key_load layer head read_time d next)
          (expected.A.keys (A.kv_head head) read_time d) /\
        Approx.v_approximates (attention_value_load layer head read_time d next)
          (expected.A.values (A.kv_head head) read_time d)) =
    append_cell_refines_shared layer time normed projected old
      ideal_old ideal_key ideal_value (head / 4) read_time d;
    let next = write_prefix layer time normed projected old 512 in
    let expected = A.append_kv time ideal_old ideal_key ideal_value in
    let i = kv_offset (layer / 4) (head / 4) read_time d in
    Cast.cast_bf16_to_f32_ok (S.index next.keys i)
      (expected.A.keys (A.kv_head head) read_time d);
    Cast.cast_bf16_to_f32_ok (S.index next.values i)
      (expected.A.values (A.kv_head head) read_time d)
  in
  C.impl_intro numerical

let native_cache_cell_binding #m native metadata layer time convolution states buffers
  head read_time d = ()

let first_attention_native_storage_step #m native metadata
  convolution states old normed projected head d =
  source_bound_buffer_step metadata 3 0 convolution states old normed projected;
  native_cache_cell_binding native metadata 3 0 convolution states
    (write_prefix 3 0 normed projected old 512) head 0 d;
  offset_decoding 0 (head / 4) 0 d;
  write_prefix_cell 3 0 normed projected old 512
    (kv_offset 0 (head / 4) 0 d)

let input_embedding_projection_binding #p inputs token d =
  S.init_ghost_index_ 248320 (input_embedding_row inputs) token;
  S.init_ghost_index_ 1024
    (fun i -> KPlay.Proof.NativeInput.cell inputs.weights (token * 1024 + i)) d;
  KPlay.Proof.NativeInput.cell_binding inputs.weights (token * 1024 + d)

let source_pointer_cells_checked #p inputs addresses token d prompt_index =
  input_embedding_projection_binding inputs token d;
  KPlay.Proof.NativeInput.cell_binding inputs.prompt prompt_index
