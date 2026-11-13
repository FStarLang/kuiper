module KPlay.Proof.Base.LayerNumerics

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
module Seq = FStar.Seq
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module Approx = Kuiper.Approximates.Base
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module R = KPlay.Proof.Shared.Recurrent
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Init = KPlay.Proof.Base.Initialization
module Cache = KPlay.Proof.Base.Cache
module Native = KPlay.Proof.Base.CacheNative
module Global = KPlay.Proof.Base.GlobalCache
module CacheMath = KPlay.Proof.Base.CacheMath
module Prior = KPlay.Proof.Base.RecurrentPipeline
module Mixer = KPlay.Proof.Base.RecurrentMixer
module Heads = KPlay.Proof.Base.RecurrentHeads
module Conv = KPlay.Proof.Base.Convolution
module Convolution = KPlay.Proof.Base.ConvolutionLoop
module Math = KPlay.Proof.Base.ScratchMath
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
let recurrent_inputs (c:Prior.logical_call) (observed:Cache.native_cache) (ideal:A.hybrid_cache)
  (state:lseq FP.t 4718592) (conv:Conv.cells) (kv:Native.snapshot)
  : Lemma
      (requires c.Prior.before == Model.recurrent_cache ideal /\
        O.cache_shape c.Prior.call.R.cache_position c.Prior.call.R.layer_index observed /\
        Global.represents observed state conv kv /\ O.cache_approximates observed ideal)
      (ensures
        O.matrix_approximates (Init.convolution_layer conv c.Prior.call.R.layer_index)
          (Prior.old_convolution c.Prior.call c.Prior.before) /\
        O.tensor_approximates (Init.state_layer state c.Prior.call.R.layer_index)
          (Prior.old_recurrent c.Prior.call c.Prior.before)) =
  let layer = c.Prior.call.R.layer_index in
  let convolution_cell (channel:P.index 6144) (tap:P.index 4)
    : Lemma (Approx.v_approximates (Init.convolution_layer conv layer channel tap)
        (Prior.old_convolution c.Prior.call c.Prior.before channel tap)) =
    if c.Prior.call.R.cache_position = 0 then (
      assert (observed.O.convolution layer == None);
      assert (Init.convolution_layer conv layer channel tap == zero #BF.t)
    ) else (
      assert (Some? (observed.O.convolution layer) /\ Some? (ideal.A.convolution layer));
      let actual:O.matrix BF.t 6144 4 = R.present_value (observed.O.convolution layer) in
      let expected:A.convolution_state = R.present_value (ideal.A.convolution layer) in
      assert (Init.convolution_layer conv layer channel tap == actual channel tap);
      assert (Prior.old_convolution c.Prior.call c.Prior.before channel tap == expected channel tap)
    )
  in
  FStar.Classical.forall_intro_2 convolution_cell;
  let recurrent_cell (head:P.index 16) (i j:P.index 128)
    : Lemma (Approx.v_approximates (Init.state_layer state layer head i j)
        (Prior.old_recurrent c.Prior.call c.Prior.before head i j)) =
    if c.Prior.call.R.cache_position = 0 then (
      assert (observed.O.recurrent layer == None);
      assert (Init.state_layer state layer head i j == zero #FP.t)
    ) else (
      assert (Some? (observed.O.recurrent layer) /\ Some? (ideal.A.recurrent layer));
      let actual:O.tensor FP.t 16 128 128 = R.present_value (observed.O.recurrent layer) in
      let expected:A.recurrent_state = R.present_value (ideal.A.recurrent layer) in
      assert (Init.state_layer state layer head i j == actual head i j);
      assert (Prior.old_recurrent c.Prior.call c.Prior.before head i j == expected head i j)
    )
  in
  FStar.Classical.forall_intro_3 recurrent_cell

inline_for_extraction noextract
let convolution_result (layer:Model.linear_layer) (before:Conv.cells) (input:Transfer.scratch)
  (old:R.convolution_state) (raw:R.vector 6144)
  : Lemma
      (requires O.matrix_approximates (Init.convolution_layer before layer) old /\
        Math.represents #6144 input 2048 raw)
      (ensures O.matrix_approximates
        (Init.convolution_layer (Convolution.after before input (layer - layer / 4) 6144) layer)
        (R.shift_convolution old raw)) =
  let at_cell (channel:P.index 6144) (tap:P.index 4)
    : Lemma (Approx.v_approximates
        (Init.convolution_layer (Convolution.after before input (layer - layer / 4) 6144) layer channel tap)
        (R.shift_convolution old raw channel tap)) =
    if tap < 3 then
      assert (Init.convolution_layer (Convolution.after before input (layer - layer / 4) 6144) layer channel tap ==
        Init.convolution_layer before layer channel (tap + 1))
    else
      assert (Init.convolution_layer (Convolution.after before input (layer - layer / 4) 6144) layer channel tap ==
        Seq.index input (2048 + channel))
  in
  FStar.Classical.forall_intro_2 at_cell

inline_for_extraction noextract
let install_linear (before:Cache.native_cache) (ideal:A.hybrid_cache) (layer:Model.linear_layer)
  (state:lseq FP.t 4718592) (conv:Conv.cells) (values:R.layer_values)
  : Lemma
      (requires O.cache_approximates before ideal /\
        O.tensor_approximates (Init.state_layer state layer) values.R.recurrent_value /\
        O.matrix_approximates (Init.convolution_layer conv layer) values.R.convolution_value)
      (ensures O.cache_approximates (Global.linear_after before layer state conv)
        (Model.install_recurrent ideal layer values)) = ()

inline_for_extraction noextract
let recurrent_result (c:Prior.logical_call) (s:Prior.decoder_stages) (layer:Model.linear_layer)
  (state:lseq FP.t 4718592) (mixer_result:Transfer.scratch)
  : Lemma
      (requires s == Prior.build_stages c /\
        Heads.completed state mixer_result (layer - layer / 4) (Mixer.head_models s) 16)
      (ensures O.tensor_approximates (Init.state_layer state layer)
        (R.decoder_values c.Prior.call c.Prior.weight c.Prior.input c.Prior.before).R.recurrent_value) =
  Mixer.output c s state mixer_result (layer - layer / 4);
  let at_cell (head:P.index 16) (i j:P.index 128)
    : Lemma (Approx.v_approximates (Init.state_layer state layer head i j)
        ((R.decoder_values c.Prior.call c.Prior.weight c.Prior.input c.Prior.before).R.recurrent_value head i j)) = ()
  in
  FStar.Classical.forall_intro_3 at_cell

inline_for_extraction noextract
let history_values (physical:Native.snapshot) (layer:A.full_attention_layer)
  (actual:O.kv_history BF.t) (ideal:A.kv_history)
  : Lemma
      (requires actual.O.length == ideal.A.length /\ ideal.A.length <= 2048 /\
        Cache.represents_slot (Global.contents physical) layer (Some actual) /\
        CacheMath.history_approximates physical layer ideal)
      (ensures forall (head:P.index 2) (time:P.index actual.O.length) (d:P.index 256).
        Approx.v_approximates (actual.O.keys head time d) (ideal.A.keys head time d) /\
        Approx.v_approximates (actual.O.values head time d) (ideal.A.values head time d)) =
  let at_cell (head:P.index 2) (time:P.index actual.O.length) (d:P.index 256)
    : Lemma (
        Approx.v_approximates (actual.O.keys head time d) (ideal.A.keys head time d) /\
        Approx.v_approximates (actual.O.values head time d) (ideal.A.values head time d)) =
    assert (Seq.index physical (Cache.key_address layer time (head * 256 + d)) == Some (actual.O.keys head time d));
    assert (Seq.index physical (Cache.value_address layer time (head * 256 + d)) == Some (actual.O.values head time d))
  in
  FStar.Classical.forall_intro_3 at_cell

inline_for_extraction noextract
let install_attention (before:Cache.native_cache) (ideal:A.hybrid_cache)
  (layer:A.full_attention_layer) (position:nat{position < 2048}) (prepared:Transfer.scratch)
  (history:A.kv_history{history.A.length == position + 1}) (physical:Native.snapshot)
  : Lemma
      (requires O.cache_shape position layer before /\ O.cache_approximates before ideal /\
        Cache.represents_slot (Global.contents physical) layer
          (Some (Cache.appended_history position (before.O.kv layer) prepared)) /\
        CacheMath.history_approximates physical layer history)
      (ensures O.cache_approximates (Cache.source_after before prepared layer position)
        (A.replace_kv ideal layer history)) =
  history_values physical layer (Cache.appended_history position (before.O.kv layer) prepared) history
