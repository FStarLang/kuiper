module KPlay.Proof.Optimized.State

#set-options "--fuel 3 --ifuel 1 --z3rlimit 80"

open Kuiper
module S = FStar.Seq
module C = FStar.Classical
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Cache = KPlay.Proof.Optimized.Cache
module Memory = KPlay.Proof.Optimized.Memory
module Conv = KPlay.Proof.Optimized.Convolution
module Recurrent = KPlay.Proof.Optimized.Recurrent
module Attention = KPlay.Proof.Optimized.Attention
module Ops = KPlay.Proof.Optimized.AttentionOps

inline_for_extraction noextract
let native_cache = Cache.native_cache O.cache_configuration

noextract
noeq type physical = {
  states: Cache.recurrent_storage;
  convolution: Cache.convolution_storage;
  keys: Cache.native_kv;
  values: Cache.native_kv;
}

noextract
noeq type stage = {
  hidden: lseq bf16 1024;
  cache: native_cache;
  memory: physical;
}

inline_for_extraction noextract
let logical_stage (s:stage)
  : O.stage O.cache_configuration bf16 f32 f32 bf16 = {
  O.hidden = (fun i -> S.index s.hidden i);
  O.cache = s.cache;
}

inline_for_extraction noextract
let linear_slot (layer:Model.linear_layer) : P.index 18 =
  (layer / 4) * 3 + layer % 4

let linear_slot_inverse (layer:Model.linear_layer)
  : Lemma (
      layer == 4 * (linear_slot layer / 3) + linear_slot layer % 3) = ()

let linear_slot_injective (left right:Model.linear_layer)
  : Lemma (linear_slot left == linear_slot right ==> left == right) =
  linear_slot_inverse left;
  linear_slot_inverse right

inline_for_extraction noextract
let local_convolution (memory:physical) (layer:Model.linear_layer) : lseq f32 24576 =
  S.slice memory.convolution (linear_slot layer * 24576) ((linear_slot layer + 1) * 24576)

inline_for_extraction noextract
let local_recurrent (memory:physical) (layer:Model.linear_layer) : lseq f32 262144 =
  S.slice memory.states (linear_slot layer * 262144) ((linear_slot layer + 1) * 262144)

inline_for_extraction noextract
let local_keys (memory:physical) (layer:A.full_attention_layer) : lseq bf16 1048576 =
  S.slice memory.keys (layer / 4 * 1048576) ((layer / 4 + 1) * 1048576)

inline_for_extraction noextract
let local_values (memory:physical) (layer:A.full_attention_layer) : lseq bf16 1048576 =
  S.slice memory.values (layer / 4 * 1048576) ((layer / 4 + 1) * 1048576)

inline_for_extraction noextract
let represents (position:nat{position < 2048}) (cursor:Model.cursor)
  (cache:native_cache) (memory:physical) : prop =
  O.cache_shape position cursor cache /\
  cache.metadata == Cache.configuration /\
  (forall (l:Model.linear_layer) (channel:P.index 6144) (tap:P.index 4).
    match cache.convolution l with
    | None -> Cache.convolution_view l memory.convolution channel tap == (zero #f32)
    | Some c -> c channel tap == Cache.convolution_view l memory.convolution channel tap) /\
  (forall (l:Model.linear_layer) (head:P.index 16) (key value:P.index 128).
    match cache.recurrent l with
    | None -> Cache.recurrent_view l memory.states head key value == (zero #f32)
    | Some r -> r head key value == Cache.recurrent_view l memory.states head key value) /\
  (forall (l:A.full_attention_layer) (head:P.index 2) (time:P.index 2048) (d:P.index 256).
    let offset = Cache.kv_offset (l / 4) head time d in
    (match cache.kv l with
    | None -> S.index memory.keys offset == (zero #bf16) /\ S.index memory.values offset == (zero #bf16)
    | Some history ->
        if time < history.length then
          history.keys head time d == S.index memory.keys offset /\
          history.values head time d == S.index memory.values offset
        else S.index memory.keys offset == (zero #bf16) /\ S.index memory.values offset == (zero #bf16)))

inline_for_extraction noextract
let zeroed : physical = {
  states = S.create 4718592 (zero #f32);
  convolution = S.create 442368 (zero #f32);
  keys = S.create 6291456 (zero #bf16);
  values = S.create 6291456 (zero #bf16);
}

let initial () : Lemma (represents 0 0 (O.empty_cache Cache.configuration) zeroed) = ()

inline_for_extraction noextract
let linear_update (before:stage) (layer:Model.linear_layer) (produced:Recurrent.observation)
  : O.layer_update f32 f32 bf16 =
  O.LinearUpdate
    (Cache.convolution_view layer before.memory.convolution)
    (Cache.recurrent_view layer before.memory.states)
    (fun c -> S.index produced.projected c)
    (fun h key value -> S.index produced.recurrent (h * 16384 + value * 128 + key))

inline_for_extraction noextract
let linear_after (before:stage) (layer:Model.linear_layer) (produced:Recurrent.observation) : GTot stage = {
  hidden = produced.hidden;
  memory = {
    states = Memory.overwrite #f32 #4718592 #262144
      before.memory.states (linear_slot layer * 262144) produced.recurrent;
    convolution = Memory.overwrite #f32 #442368 #24576
      before.memory.convolution (linear_slot layer * 24576) produced.convolution;
    keys = before.memory.keys;
    values = before.memory.values;
  };
  cache = {
    O.metadata = before.cache.metadata;
    O.kv = before.cache.kv;
    O.convolution = (fun l -> if l = layer
      then Some (fun (c:P.index 6144) (t:P.index 4) -> S.index produced.convolution (c * 4 + t))
      else before.cache.convolution l);
    O.recurrent = (fun l -> if l = layer
      then Some (fun (h:P.index 16) (key value:P.index 128) ->
        S.index produced.recurrent (h * 16384 + value * 128 + key))
      else before.cache.recurrent l);
  };
}

let linear_structure (position:nat{position < 2048}) (layer:Model.linear_layer)
  (before:stage) (produced:Recurrent.observation)
  : Lemma
      (requires represents position layer before.cache before.memory /\
        produced.convolution == Conv.heads_prefix (local_convolution before.memory layer) produced.projected 16)
      (ensures
        O.layer_structure position layer before.cache (linear_after before layer produced).cache
          (linear_update before layer produced)) =
  let shape (l:A.layer_index)
    : Lemma (
        let next = (linear_after before layer produced).cache in
        let count = Model.visits position (layer + 1) l in
        if l % 4 < 3 then
          next.kv l == None /\
          (Some? (next.convolution l) <==> count > 0) /\
          (Some? (next.recurrent l) <==> count > 0)
        else next.convolution l == None /\ next.recurrent l == None /\
          O.history_length (next.kv l) == count) = ()
  in
  C.forall_intro shape

let linear_represents (position:nat{position < 2048}) (layer:Model.linear_layer)
  (before:stage) (produced:Recurrent.observation)
  : Lemma
      (requires represents position layer before.cache before.memory /\
        produced.convolution == Conv.heads_prefix (local_convolution before.memory layer) produced.projected 16)
      (ensures represents position (layer + 1)
        (linear_after before layer produced).cache (linear_after before layer produced).memory) =
  linear_structure position layer before produced;
  let after = linear_after before layer produced in
  let convolution_cell (l:Model.linear_layer) (channel:P.index 6144) (tap:P.index 4)
    : Lemma (
        match after.cache.convolution l with
        | None -> Cache.convolution_view l after.memory.convolution channel tap == (zero #f32)
        | Some c -> c channel tap == Cache.convolution_view l after.memory.convolution channel tap) =
    linear_slot_injective layer l;
    Memory.overwrite_at #f32 #442368 #24576 before.memory.convolution
      (linear_slot layer * 24576) produced.convolution
      (linear_slot l * 24576 + channel * 4 + tap);
    if l <> layer then (
      assert (Memory.outside (linear_slot layer * 24576) 24576
        (linear_slot l * 24576 + channel * 4 + tap));
      assert (after.cache.convolution l == before.cache.convolution l);
      assert (Cache.convolution_view l after.memory.convolution channel tap ==
        Cache.convolution_view l before.memory.convolution channel tap);
      match before.cache.convolution l with
      | None -> assert (Cache.convolution_view l before.memory.convolution channel tap == (zero #f32))
      | Some stored -> assert (stored channel tap ==
          Cache.convolution_view l before.memory.convolution channel tap)
    )
  in
  C.forall_intro_3 convolution_cell;
  let recurrent_cell (l:Model.linear_layer) (head:P.index 16) (key value:P.index 128)
    : Lemma (
        match after.cache.recurrent l with
        | None -> Cache.recurrent_view l after.memory.states head key value == (zero #f32)
        | Some r -> r head key value == Cache.recurrent_view l after.memory.states head key value) =
    linear_slot_injective layer l;
    Memory.overwrite_at #f32 #4718592 #262144 before.memory.states
      (linear_slot layer * 262144) produced.recurrent
      (linear_slot l * 262144 + head * 16384 + value * 128 + key);
    if l <> layer then (
      assert (Memory.outside (linear_slot layer * 262144) 262144
        (linear_slot l * 262144 + head * 16384 + value * 128 + key));
      assert (after.cache.recurrent l == before.cache.recurrent l);
      assert (Cache.recurrent_view l after.memory.states head key value ==
        Cache.recurrent_view l before.memory.states head key value);
      match before.cache.recurrent l with
      | None -> assert (Cache.recurrent_view l before.memory.states head key value == (zero #f32))
      | Some stored -> assert (stored head key value ==
          Cache.recurrent_view l before.memory.states head key value)
    )
  in
  C.forall_intro_4 recurrent_cell;
  assert (after.cache.kv == before.cache.kv /\
    after.memory.keys == before.memory.keys /\ after.memory.values == before.memory.values)

inline_for_extraction noextract
let attention_history (position:nat{position < 2048}) (produced:Attention.observation position)
  : O.kv_history bf16 = {
  O.length = position + 1;
  O.keys = (fun h t d -> S.index produced.keys (h * 524288 + t * 256 + d));
  O.values = (fun h t d -> S.index produced.values (h * 524288 + t * 256 + d));
}

inline_for_extraction noextract
let attention_update (position:nat{position < 2048}) (produced:Attention.observation position)
  : O.layer_update f32 f32 bf16 =
  O.AttentionUpdate
    (fun h d -> S.index produced.keys (h * 524288 + position * 256 + d))
    (fun h d -> S.index produced.values (h * 524288 + position * 256 + d))

inline_for_extraction noextract
let attention_after (position:nat{position < 2048}) (before:stage)
  (layer:A.full_attention_layer) (produced:Attention.observation position) : GTot stage = {
  hidden = produced.hidden;
  memory = {
    states = before.memory.states;
    convolution = before.memory.convolution;
    keys = Memory.overwrite #bf16 #6291456 #1048576
      before.memory.keys (layer / 4 * 1048576) produced.keys;
    values = Memory.overwrite #bf16 #6291456 #1048576
      before.memory.values (layer / 4 * 1048576) produced.values;
  };
  cache = Cache.install_history before.cache layer (attention_history position produced);
}

let attention_structure (position:nat{position < 2048}) (layer:A.full_attention_layer)
  (before:stage) (produced:Attention.observation position)
  : Lemma
      (requires represents position layer before.cache before.memory /\
        Ops.cache_frame position (local_keys before.memory layer) produced.keys /\
        Ops.cache_frame position (local_values before.memory layer) produced.values)
      (ensures O.layer_structure position layer before.cache
        (attention_after position before layer produced).cache (attention_update position produced)) =
  let prefix (h:P.index 2) (t:nat{t < position}) (d:P.index 256)
    : Lemma (
        S.index produced.keys (h * 524288 + t * 256 + d) ==
          S.index before.memory.keys (Cache.kv_offset (layer / 4) h t d) /\
        S.index produced.values (h * 524288 + t * 256 + d) ==
          S.index before.memory.values (Cache.kv_offset (layer / 4) h t d)) =
    Ops.key_index_time h t d
  in C.forall_intro_3 prefix

let attention_represents (position:nat{position < 2048}) (layer:A.full_attention_layer)
  (before:stage) (produced:Attention.observation position)
  : Lemma
      (requires represents position layer before.cache before.memory /\
        Ops.cache_frame position (local_keys before.memory layer) produced.keys /\
        Ops.cache_frame position (local_values before.memory layer) produced.values)
      (ensures represents position (layer + 1)
        (attention_after position before layer produced).cache (attention_after position before layer produced).memory) =
  attention_structure position layer before produced;
  assert (O.history_length (before.cache.kv layer) == position);
  let tail (h:P.index 2) (t:P.index 2048{position < t}) (d:P.index 256)
    : Lemma (
        S.index produced.keys (h * 524288 + t * 256 + d) == (zero #bf16) /\
        S.index produced.values (h * 524288 + t * 256 + d) == (zero #bf16)) =
    Ops.key_index_time h t d;
    let offset = Cache.kv_offset (layer / 4) h t d in
    (match before.cache.kv layer with
    | None -> ()
    | Some history -> assert (history.length == position));
    assert (S.index before.memory.keys offset == (zero #bf16) /\
      S.index before.memory.values offset == (zero #bf16));
    assert (S.index (local_keys before.memory layer) (h * 524288 + t * 256 + d) ==
      S.index before.memory.keys offset);
    assert (S.index (local_values before.memory layer) (h * 524288 + t * 256 + d) ==
      S.index before.memory.values offset)
  in
  C.forall_intro_3 tail;
  let after = attention_after position before layer produced in
  let kv_cell (l:A.full_attention_layer) (h:P.index 2) (t:P.index 2048) (d:P.index 256)
    : Lemma (
        let offset = Cache.kv_offset (l / 4) h t d in
        match after.cache.kv l with
        | None -> S.index after.memory.keys offset == (zero #bf16) /\
            S.index after.memory.values offset == (zero #bf16)
        | Some history ->
            if t < history.length then
              history.keys h t d == S.index after.memory.keys offset /\
              history.values h t d == S.index after.memory.values offset
            else S.index after.memory.keys offset == (zero #bf16) /\
              S.index after.memory.values offset == (zero #bf16)) =
    let offset = Cache.kv_offset (l / 4) h t d in
    Memory.overwrite_at #bf16 #6291456 #1048576 before.memory.keys
      (layer / 4 * 1048576) produced.keys offset;
    Memory.overwrite_at #bf16 #6291456 #1048576 before.memory.values
      (layer / 4 * 1048576) produced.values offset;
    if l <> layer then (
      assert (l / 4 <> layer / 4);
      assert (Memory.outside (layer / 4 * 1048576) 1048576 offset);
      assert (after.cache.kv l == before.cache.kv l);
      assert (S.index after.memory.keys offset == S.index before.memory.keys offset /\
        S.index after.memory.values offset == S.index before.memory.values offset);
      match before.cache.kv l with
      | None -> assert (S.index before.memory.keys offset == (zero #bf16) /\
          S.index before.memory.values offset == (zero #bf16))
      | Some history ->
          if t < history.length then
            assert (history.keys h t d == S.index before.memory.keys offset /\
              history.values h t d == S.index before.memory.values offset)
          else assert (S.index before.memory.keys offset == (zero #bf16) /\
              S.index before.memory.values offset == (zero #bf16))
    )
  in
  C.forall_intro_4 kv_cell;
  assert (after.cache.convolution == before.cache.convolution /\
    after.cache.recurrent == before.cache.recurrent /\
    after.memory.states == before.memory.states /\
    after.memory.convolution == before.memory.convolution)

let boundary (position:nat{position + 1 < 2048}) (cache:native_cache) (memory:physical)
  : Lemma
      (requires represents position 24 cache memory)
      (ensures represents (position + 1) 0 cache memory) =
  O.cache_boundary position cache
