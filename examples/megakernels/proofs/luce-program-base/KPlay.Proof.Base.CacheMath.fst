module KPlay.Proof.Base.CacheMath

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open FStar.Real
module Seq = FStar.Seq
module BF = Kuiper.BFloat16
module Approx = Kuiper.Approximates.Base
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module O = KPlay.Proof.Shared.Observations
module R = KPlay.Proof.Shared.Recurrent
module Model = KPlay.Proof.Shared.Model
module Cache = KPlay.Proof.Base.Cache
module Native = KPlay.Proof.Base.CacheNative
module Projection = KPlay.Proof.Base.CacheProjection
module Transfer = KPlay.Proof.Base.Transfer
module Schedule = KPlay.Proof.Base.Schedule
module Layout = KPlay.Proof.Base.MatvecLayout
module Math = KPlay.Proof.Base.AttentionAlgebra
module SM = KPlay.Proof.Base.SIMTMath
module C = Kuiper.Chest
module T = Kuiper.Tensor
module SZ = Kuiper.SizeT
module Affine = KPlay.Proof.Base.AffineMMA
module Scratch = KPlay.Proof.Base.ScratchMath

inline_for_extraction noextract
let cell_approximates (cell:option BF.t) (value:real) : prop =
  exists (actual:BF.t). cell == Some actual /\ Approx.v_approximates actual value

inline_for_extraction noextract
let history_approximates (physical:Native.snapshot) (layer:A.full_attention_layer)
  (history:A.kv_history{history.A.length <= 2048}) : prop =
  forall (head:P.index 2) (time:P.index history.A.length) (d:P.index 256).
    cell_approximates (Seq.index physical (Cache.key_address layer time (head * 256 + d)))
      (history.A.keys head time d) /\
    cell_approximates (Seq.index physical (Cache.value_address layer time (head * 256 + d)))
      (history.A.values head time d)

inline_for_extraction noextract
let optional_history (physical:Native.snapshot) (layer:A.full_attention_layer) (old:option A.kv_history) : prop =
  match old with
  | None -> True
  | Some history -> history.A.length <= 2048 /\ history_approximates physical layer history

inline_for_extraction noextract
let prepared_matrix (hidden:Transfer.scratch) (keys values:A.matrix 2 256)
  : Lemma
      (requires Scratch.represents #512 hidden 6144 (Math.flat_heads 2 keys) /\
        Scratch.represents #512 hidden 6656 (Math.flat_heads 2 values))
      (ensures O.matrix_approximates (Cache.produced_key hidden) keys /\
        O.matrix_approximates (Cache.produced_value hidden) values) =
  let at_cell (head:P.index 2) (d:P.index 256)
    : Lemma (
        Approx.v_approximates (Seq.index hidden (6144 + head * 256 + d)) (keys head d) /\
        Approx.v_approximates (Seq.index hidden (6656 + head * 256 + d)) (values head d)) =
    assert (Math.flat_heads 2 keys (head * 256 + d) == keys head d);
    assert (Math.flat_heads 2 values (head * 256 + d) == values head d)
  in
  FStar.Classical.forall_intro_2 at_cell

inline_for_extraction noextract
let append (physical:Native.snapshot) (hidden:Transfer.scratch) (layer:A.full_attention_layer)
  (position:nat{position < 2048}) (old:option A.kv_history{A.history_length old == position})
  (keys values:A.matrix 2 256)
  : Lemma
      (requires optional_history physical layer old /\
        O.matrix_approximates (Cache.produced_key hidden) keys /\
        O.matrix_approximates (Cache.produced_value hidden) values)
      (ensures history_approximates (Native.appended physical hidden layer position 512) layer
        (A.append_kv position old keys values)) =
  let at_cell (head:P.index 2) (time:P.index (position + 1)) (d:P.index 256)
    : Lemma (
        cell_approximates
          (Seq.index (Native.appended physical hidden layer position 512)
            (Cache.key_address layer time (head * 256 + d)))
          ((A.append_kv position old keys values).A.keys head time d) /\
        cell_approximates
          (Seq.index (Native.appended physical hidden layer position 512)
            (Cache.value_address layer time (head * 256 + d)))
          ((A.append_kv position old keys values).A.values head time d)) =
    if time < position then (
      assert (Some? old);
      Native.append_frame physical hidden layer position (Cache.key_address layer time (head * 256 + d));
      Native.append_frame physical hidden layer position (Cache.value_address layer time (head * 256 + d))
    ) else (
      assert (time == position);
      match old with | None -> () | Some _ -> ()
    )
  in
  FStar.Classical.forall_intro_3 at_cell

inline_for_extraction noextract
let prefix_cell (physical:Native.snapshot) (layer:A.full_attention_layer) (length:pos{length <= 2048})
  (value:bool) (ready:squash (Native.initialized physical layer length)) (i:P.index (length * 512)) (r:real)
  : Lemma
      (requires cell_approximates (Seq.index physical (Native.plane_offset layer value + i)) r)
      (ensures Approx.v_approximates (Seq.index (Native.prefix_contents physical layer length value ready) i) r) = ()

inline_for_extraction noextract
let rows (position:nat) (score:bool) : pos = if score then position + 1 else 256

inline_for_extraction noextract
let columns (position:nat) (score:bool) : pos = if score then 256 else position + 1

inline_for_extraction noextract
let matrix (position:nat) (history:A.kv_history{history.A.length == position + 1})
  (head:P.index 8) (score:bool) : A.matrix (rows position score) (columns position score) =
  if score then Math.score_matrix (position + 1) history head else Math.value_matrix (position + 1) history head

inline_for_extraction noextract
let matrix_cell (physical:Native.snapshot) (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:A.full_attention_layer) (phase:Schedule.phase layer{Schedule.attention layer phase})
  (history:A.kv_history{history.A.length == position + 1})
  (ready:squash (Native.initialized physical layer (position + 1)))
  (row:P.index (Schedule.projection_at call position layer phase).Schedule.rows)
  (col:P.index (Schedule.projection_at call position layer phase).Schedule.columns)
  : Lemma
      (requires history_approximates physical layer history)
      (ensures Approx.v_approximates
        (Layout.matrix (Projection.geometry call position layer phase)
          (Projection.tensor_contents physical call position layer phase ready) row col)
        (matrix position history ((phase - 3) / 2) (phase % 2 = 1) row col)) =
  Projection.source_address call position layer phase row col;
  if phase % 2 = 1 then
    prefix_cell physical layer (position + 1) false ready
      (Layout.tile_offset (Projection.geometry call position layer phase) row col)
      (history.A.keys (((phase - 3) / 2) / 4) row col)
  else
    prefix_cell physical layer (position + 1) true ready
      (Layout.tile_offset (Projection.geometry call position layer phase) row col)
      (history.A.values (((phase - 3) / 2) / 4) col row)

inline_for_extraction noextract
let matrix_approx (scope:prop) (physical:Native.snapshot) (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:A.full_attention_layer) (phase:Schedule.phase layer{Schedule.attention layer phase})
  (history:A.kv_history{history.A.length == position + 1})
  (ready:squash (Native.initialized physical layer (position + 1)))
  : Lemma
      (requires scope ==> history_approximates physical layer history)
      (ensures scope ==> O.matrix_approximates
        (Layout.matrix (Projection.geometry call position layer phase)
          (Projection.tensor_contents physical call position layer phase ready))
        (matrix position history ((phase - 3) / 2) (phase % 2 = 1))) =
  let under (_:scope)
    : Lemma (O.matrix_approximates
        (Layout.matrix (Projection.geometry call position layer phase)
          (Projection.tensor_contents physical call position layer phase ready))
        (matrix position history ((phase - 3) / 2) (phase % 2 = 1))) =
    FStar.Classical.forall_intro_2 (matrix_cell physical call position layer phase history ready)
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
let simt_matrix (scope:prop) (physical:Native.snapshot) (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:A.full_attention_layer) (phase:Schedule.phase layer{Schedule.attention layer phase})
  (history:A.kv_history{history.A.length == position + 1})
  (ready:squash (Native.initialized physical layer (position + 1)))
  : Lemma
      (requires scope ==> history_approximates physical layer history)
      (ensures scope ==> SM.matrix_rep (Projection.geometry call position layer phase)
        (Projection.contents physical call position layer phase ready)
        (matrix position history ((phase - 3) / 2) (phase % 2 = 1))) =
  matrix_approx scope physical call position layer phase history ready
