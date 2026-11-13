module KPlay.Proof.Base.GlobalCache

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
module Seq = FStar.Seq
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Init = KPlay.Proof.Base.Initialization
module Cache = KPlay.Proof.Base.Cache
module Native = KPlay.Proof.Base.CacheNative
module Heads = KPlay.Proof.Base.RecurrentHeads
module Conv = KPlay.Proof.Base.Convolution
module Convolution = KPlay.Proof.Base.ConvolutionLoop
module Transfer = KPlay.Proof.Base.Transfer
module CacheMath = KPlay.Proof.Base.CacheMath
module Approx = Kuiper.Approximates.Base

inline_for_extraction noextract
let contents (values:Native.snapshot) : Cache.buffer = fun i -> Seq.index values i

inline_for_extraction noextract
let available_contents (kv:Native.snapshot) : prop =
  forall (i:P.index 12582912). Some? (Seq.index kv i)

inline_for_extraction noextract
let represents (observed:Cache.native_cache) (state:lseq FP.t 4718592) (conv:Conv.cells) (kv:Native.snapshot) : prop =
  available_contents kv /\
  (forall (layer:Model.linear_layer).
    O.convolution_input observed layer (Init.convolution_layer conv layer) /\
    O.recurrent_input observed layer (Init.state_layer state layer)) /\
  (forall (layer:A.full_attention_layer). Cache.represents_slot (contents kv) layer (observed.O.kv layer))

inline_for_extraction noextract
let initial (metadata:O.cache_configuration) (kv:Native.snapshot)
  : Lemma
      (requires available_contents kv)
      (ensures represents (O.empty_cache #O.cache_configuration #BF.t #FP.t #BF.t metadata)
        Init.state_values Init.convolution_values kv) =
  Init.all_first_use_inputs metadata

inline_for_extraction noextract
let initialized (observed:Cache.native_cache) (state:lseq FP.t 4718592) (conv:Conv.cells) (kv:Native.snapshot)
  (position:nat{position <= 2048}) (next:Model.cursor) (layer:A.full_attention_layer)
  : Lemma
      (requires represents observed state conv kv /\ O.cache_shape position next observed /\
        Model.visits position next layer <= 2048)
      (ensures Native.initialized kv layer (Model.visits position next layer)) =
  let at_cell (value:bool) (i:nat{i < Model.visits position next layer * 512})
    : Lemma (Some? (Seq.index kv (Native.plane_offset layer value + i))) =
    assert (Model.visits position next layer > 0);
    assert (Some? (observed.O.kv layer));
    let Some history = observed.O.kv layer in
    let time = i / 512 in
    let head = (i % 512) / 256 in
    let d = i % 256 in
    assert (head < 2 /\ d < 256 /\ time < history.O.length /\ time < 2048);
    assert (time * 512 + head * 256 + d == i);
    if value then
      assert (Seq.index kv (Cache.value_address layer time (head * 256 + d)) == Some (history.O.values head time d))
    else
      assert (Seq.index kv (Cache.key_address layer time (head * 256 + d)) == Some (history.O.keys head time d))
  in
  FStar.Classical.forall_intro_2 at_cell

inline_for_extraction noextract
let convolution_other (changed other:Model.linear_layer{changed <> other})
  (before:Conv.cells) (raw:Transfer.scratch) (channel:P.index 6144) (tap:P.index 4)
  : Lemma (
      Init.convolution_layer (Convolution.after before raw (changed - changed / 4) 6144) other channel tap ==
      Init.convolution_layer before other channel tap) = ()

inline_for_extraction noextract
let recurrent_other (changed other:Model.linear_layer{changed <> other})
  (before after:lseq FP.t 4718592) (head:P.index 16) (i j:P.index 128)
  : Lemma
      (requires Heads.state_frame before after (changed - changed / 4) 16)
      (ensures Init.state_layer after other head i j == Init.state_layer before other head i j) =
  Init.state_layer_frame other changed head i j

inline_for_extraction noextract
let raw_values (hidden:Transfer.scratch) : O.vector BF.t 6144 =
  fun i -> Seq.index hidden (2048 + i)

inline_for_extraction noextract
let linear_after (before:Cache.native_cache) (layer:Model.linear_layer)
  (state:lseq FP.t 4718592) (conv:Conv.cells) : Cache.native_cache = {
  before with
  O.convolution = (fun l -> if l = layer then Some (Init.convolution_layer conv layer) else before.O.convolution l);
  O.recurrent = (fun l -> if l = layer then Some (Init.state_layer state layer) else before.O.recurrent l);
}

inline_for_extraction noextract
let linear_update (layer:Model.linear_layer) (old_state new_state:lseq FP.t 4718592)
  (old_conv:Conv.cells) (raw:Transfer.scratch) : O.layer_update BF.t FP.t BF.t =
  O.LinearUpdate (Init.convolution_layer old_conv layer) (Init.state_layer old_state layer)
    (raw_values raw) (Init.state_layer new_state layer)

inline_for_extraction noextract
let linear_shape (before:Cache.native_cache) (position:nat) (layer:Model.linear_layer)
  (state:lseq FP.t 4718592) (conv:Conv.cells)
  : Lemma
      (requires O.cache_shape position layer before)
      (ensures O.cache_shape position (layer + 1) (linear_after before layer state conv)) =
  let at_layer (l:A.layer_index) : Lemma (
      let after = linear_after before layer state conv in
      let count = Model.visits position (layer + 1) l in
      if l % 4 < 3 then
        after.O.kv l == None /\ (Some? (after.O.convolution l) <==> count > 0) /\
        (Some? (after.O.recurrent l) <==> count > 0)
      else
        after.O.convolution l == None /\ after.O.recurrent l == None /\ O.history_length (after.O.kv l) == count) =
    if l = layer then assert (Model.visits position (layer + 1) l == position + 1)
    else assert (Model.visits position (layer + 1) l == Model.visits position layer l)
  in
  FStar.Classical.forall_intro at_layer

inline_for_extraction noextract
let linear_structure (before:Cache.native_cache) (position:nat) (layer:Model.linear_layer)
  (old_state new_state:lseq FP.t 4718592) (old_conv:Conv.cells) (raw:Transfer.scratch) (kv:Native.snapshot)
  : Lemma
      (requires represents before old_state old_conv kv /\ O.cache_shape position layer before)
      (ensures O.layer_structure position layer before
        (linear_after before layer new_state (Convolution.after old_conv raw (layer - layer / 4) 6144))
        (linear_update layer old_state new_state old_conv raw)) =
  linear_shape before position layer new_state (Convolution.after old_conv raw (layer - layer / 4) 6144)

inline_for_extraction noextract
let linear_represents (before:Cache.native_cache) (layer:Model.linear_layer)
  (old_state new_state:lseq FP.t 4718592) (old_conv:Conv.cells) (raw:Transfer.scratch) (kv:Native.snapshot)
  : Lemma
      (requires represents before old_state old_conv kv /\ Heads.state_frame old_state new_state (layer - layer / 4) 16)
      (ensures represents
        (linear_after before layer new_state (Convolution.after old_conv raw (layer - layer / 4) 6144))
        new_state (Convolution.after old_conv raw (layer - layer / 4) 6144) kv) =
  let at_layer (l:Model.linear_layer)
    : Lemma (
        O.convolution_input
          (linear_after before layer new_state (Convolution.after old_conv raw (layer - layer / 4) 6144))
          l (Init.convolution_layer (Convolution.after old_conv raw (layer - layer / 4) 6144) l) /\
        O.recurrent_input
          (linear_after before layer new_state (Convolution.after old_conv raw (layer - layer / 4) 6144))
          l (Init.state_layer new_state l)) =
    if l <> layer then (
      FStar.Classical.forall_intro_2 (convolution_other layer l old_conv raw);
      FStar.Classical.forall_intro_3 (recurrent_other layer l old_state new_state)
    )
  in
  FStar.Classical.forall_intro at_layer

inline_for_extraction noextract
let append_binding (before:Native.snapshot) (hidden:Transfer.scratch) (layer:A.full_attention_layer)
  (position:nat{position < 2048}) (i:P.index 12582912)
  : Lemma (contents (Native.appended before hidden layer position 512) i ==
      Cache.append_loop (contents before) hidden layer position 512 i) =
  Cache.append_loop_cell (contents before) hidden layer position 512 i

inline_for_extraction noextract
let attention_other (before:Native.snapshot) (hidden:Transfer.scratch)
  (changed other:A.full_attention_layer{changed <> other}) (position:nat{position < 2048})
  (history:option (O.kv_history BF.t))
  : Lemma
      (requires Cache.represents_slot (contents before) other history)
      (ensures Cache.represents_slot (contents (Native.appended before hidden changed position 512)) other history) =
  match history with
  | None -> ()
  | Some old ->
    let at_cell (head:P.index 2) (time:nat{time < old.O.length /\ time < 2048}) (d:P.index 256)
      : Lemma (
          Seq.index (Native.appended before hidden changed position 512)
            (Cache.key_address other time (head * 256 + d)) == Some (old.O.keys head time d) /\
          Seq.index (Native.appended before hidden changed position 512)
            (Cache.value_address other time (head * 256 + d)) == Some (old.O.values head time d)) =
      Native.append_frame before hidden changed position (Cache.key_address other time (head * 256 + d));
      Native.append_frame before hidden changed position (Cache.value_address other time (head * 256 + d))
    in
    FStar.Classical.forall_intro_3 at_cell

inline_for_extraction noextract
let append_available (before:Native.snapshot) (hidden:Transfer.scratch) (layer:A.full_attention_layer)
  (position:nat{position < 2048})
  : Lemma
      (requires available_contents before)
      (ensures available_contents (Native.appended before hidden layer position 512)) =
  let cell (i:P.index 12582912)
    : Lemma (Some? (Seq.index (Native.appended before hidden layer position 512) i)) = ()
  in
  FStar.Classical.forall_intro cell

inline_for_extraction noextract
let attention_represents (before:Cache.native_cache) (position:nat{position < 2048})
  (layer:A.full_attention_layer) (state:lseq FP.t 4718592) (conv:Conv.cells)
  (kv:Native.snapshot) (hidden:Transfer.scratch)
  : Lemma
      (requires represents before state conv kv /\ O.cache_shape position layer before)
      (ensures represents (Cache.source_after before hidden layer position) state conv
        (Native.appended kv hidden layer position 512)) =
  append_available kv hidden layer position;
  Cache.append_represents (contents kv) hidden layer position (before.O.kv layer);
  FStar.Classical.forall_intro (append_binding kv hidden layer position);
  let at_layer (other:A.full_attention_layer)
    : Lemma (Cache.represents_slot (contents (Native.appended kv hidden layer position 512)) other
        ((Cache.source_after before hidden layer position).O.kv other)) =
    if other <> layer then attention_other kv hidden layer other position (before.O.kv other)
  in
  FStar.Classical.forall_intro at_layer

inline_for_extraction noextract
let history_approx (observed:Cache.native_cache) (ideal:A.hybrid_cache)
  (state:lseq FP.t 4718592) (conv:Conv.cells) (kv:Native.snapshot) (layer:A.full_attention_layer)
  : Lemma
      (requires represents observed state conv kv /\ O.cache_approximates observed ideal)
      (ensures CacheMath.optional_history kv layer (ideal.A.kv layer)) =
  match observed.O.kv layer, ideal.A.kv layer with
  | Some actual, Some expected ->
    assert (actual.O.length == expected.A.length /\ expected.A.length <= 2048);
    let at_cell (head:P.index 2) (time:P.index expected.A.length) (d:P.index 256)
      : Lemma (
          CacheMath.cell_approximates (Seq.index kv (Cache.key_address layer time (head * 256 + d)))
            (expected.A.keys head time d) /\
          CacheMath.cell_approximates (Seq.index kv (Cache.value_address layer time (head * 256 + d)))
            (expected.A.values head time d)) =
      assert (Seq.index kv (Cache.key_address layer time (head * 256 + d)) == Some (actual.O.keys head time d));
      assert (Seq.index kv (Cache.value_address layer time (head * 256 + d)) == Some (actual.O.values head time d))
    in
    FStar.Classical.forall_intro_3 at_cell
  | _ -> ()
