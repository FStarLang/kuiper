module KPlay.Proof.Base.SourceProjection

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor

module SZ = Kuiper.SizeT
module C = Kuiper.Chest
module P = KPlay.Proof.Shared.Projection
module O = KPlay.Proof.Shared.Observations
module A = KPlay.Proof.Shared.Attention
module Model = KPlay.Proof.Shared.Model
module Schedule = KPlay.Proof.Base.Schedule
module Layout = KPlay.Proof.Base.MatvecLayout
module Affine = KPlay.Proof.Base.AffineMMA
module Matmul = KPlay.Proof.Base.Matmul
module Matvec = KPlay.Proof.Base.Matvec
module Writes = KPlay.Proof.Base.MatvecWrites
module Native = KPlay.Proof.Base.NativeProjection
module Approx = Kuiper.Approximates.Base

inline_for_extraction noextract
let decoder_bounds (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Model.cursor{layer < 24}) (phase:Schedule.phase layer)
  : Lemma (
      let d = Schedule.projection_at call position layer phase in
      d.rows <= 6144 /\ d.columns <= 3584 /\
      (d.splits = 1 \/ (d.splits = 2 /\ d.columns % 32 = 0))) =
  Schedule.bounds call position layer phase;
  Schedule.phase_kinds layer phase;
  if Schedule.attention layer phase then ()
  else (
    let k = Schedule.kind layer phase in
    if k = 0 then ()
    else if k = 1 then ()
    else if k = 2 then ()
    else if k = 3 then ()
    else if k = 4 then ()
    else if k = 5 then ()
    else if k = 6 then ()
    else assert (k = 7)
  )

inline_for_extraction noextract
let weight_geometry (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Model.cursor{layer < 24})
  (phase:Schedule.phase layer{not (Schedule.attention layer phase)})
  : Layout.geometry 752393024 =
  Schedule.bounds call position layer phase;
  decoder_bounds call position layer phase;
  Layout.source_nonoverlap call position layer phase;
  let d = Schedule.projection_at call position layer phase in
  {
    Layout.rows = SZ.uint_to_t d.Schedule.rows;
    Layout.columns = SZ.uint_to_t d.Schedule.columns;
    Layout.offset = SZ.uint_to_t d.Schedule.weight_offset;
    Layout.row_stride = SZ.uint_to_t d.Schedule.columns;
    Layout.col_stride = 1sz;
    Layout.input_offset = SZ.uint_to_t d.Schedule.input_offset;
    Layout.output_offset = SZ.uint_to_t d.Schedule.output_offset;
    Layout.splits = SZ.uint_to_t d.Schedule.splits;
    Layout.bounds = ();
    Layout.separated = ();
  }

inline_for_extraction noextract
let cache_geometry (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:A.full_attention_layer)
  (phase:Schedule.phase layer{Schedule.attention layer phase})
  : Layout.geometry 12582912 =
  Schedule.bounds call position layer phase;
  decoder_bounds call position layer phase;
  Layout.source_nonoverlap call position layer phase;
  let d = Schedule.projection_at call position layer phase in
  let head = (phase - 3) / 2 in
  let score = phase % 2 = 1 in
  let offset = ((layer / 4) * 4096 + (if score then 0 else 2048)) * 512 + (head / 4) * 256 in
  let row_stride = if score then 512 else 1 in
  let col_stride = if score then 1 else 512 in
  let at_index (row:P.index d.Schedule.rows) (col:P.index d.Schedule.columns)
    : Lemma (offset + row * row_stride + col * col_stride < 12582912) =
    let address = Schedule.source_cache_address layer head (position + 1) score row col in
    assert (address == offset + row * row_stride + col * col_stride)
  in
  FStar.Classical.forall_intro_2 at_index;
  {
    Layout.rows = SZ.uint_to_t d.Schedule.rows;
    Layout.columns = SZ.uint_to_t d.Schedule.columns;
    Layout.offset = SZ.uint_to_t offset;
    Layout.row_stride = SZ.uint_to_t row_stride;
    Layout.col_stride = SZ.uint_to_t col_stride;
    Layout.input_offset = SZ.uint_to_t d.Schedule.input_offset;
    Layout.output_offset = SZ.uint_to_t d.Schedule.output_offset;
    Layout.splits = 1sz;
    Layout.bounds = ();
    Layout.separated = ();
  }

inline_for_extraction noextract
let weight_cell (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Model.cursor{layer < 24})
  (phase:Schedule.phase layer{not (Schedule.attention layer phase)})
  (values:chest2 bf16 752393024 1)
  (row:P.index (Schedule.projection_at call position layer phase).Schedule.rows)
  (col:P.index (Schedule.projection_at call position layer phase).Schedule.columns)
  : Lemma (
      Layout.matrix (weight_geometry call position layer phase) values row col ==
      acc2 values (Schedule.source_weight_address call position layer phase row col) 0) =
  Schedule.bounds call position layer phase

inline_for_extraction noextract
let cache_cell (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:A.full_attention_layer)
  (phase:Schedule.phase layer{Schedule.attention layer phase})
  (values:chest2 bf16 12582912 1)
  (row:P.index (Schedule.projection_at call position layer phase).Schedule.rows)
  (col:P.index (Schedule.projection_at call position layer phase).Schedule.columns)
  : Lemma (
      Layout.matrix (cache_geometry call position layer phase) values row col ==
      acc2 values (Schedule.source_cache_address layer ((phase - 3) / 2) (position + 1)
        (phase % 2 = 1) row col) 0) =
  Schedule.bounds call position layer phase

inline_for_extraction noextract
fn weight_projection
  (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Model.cursor{layer < 24})
  (phase:Schedule.phase layer{not (Schedule.attention layer phase)})
  (weight:array2 bf16 Native.weight_layout) (hidden:array2 bf16 Native.hidden_layout)
  (#vw:chest2 bf16 752393024 1) (#vh:chest2 bf16 16000 1) (#weight_fraction:perm)
  preserves gpu ** weight |-> Frac weight_fraction vw
  requires hidden |-> Frac (1.0R /. 32) vh
  ensures hidden |-> Frac (1.0R /. 32)
    (Writes.after (weight_geometry call position layer phase) vw vh
      (Schedule.projection_at call position layer phase).Schedule.rows)
{
  let g = weight_geometry call position layer phase;
  Matvec.projection g weight hidden;
}

inline_for_extraction noextract
fn cache_projection
  (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:A.full_attention_layer)
  (phase:Schedule.phase layer{
    Schedule.attention layer phase /\ Schedule.mma_path layer phase position})
  (cache:array2 bf16 (Affine.parent_layout 12582912)) (hidden:array2 bf16 Native.hidden_layout)
  (#vk:chest2 bf16 12582912 1) (#vh:chest2 bf16 16000 1) (#cache_fraction:perm)
  preserves gpu ** cache |-> Frac cache_fraction vk
  requires hidden |-> Frac (1.0R /. 32) vh
  ensures hidden |-> Frac (1.0R /. 32)
    (Writes.after (cache_geometry call position layer phase) vk vh
      (Schedule.projection_at call position layer phase).Schedule.rows)
{
  let g = cache_geometry call position layer phase;
  Matvec.projection g cache hidden;
}

inline_for_extraction noextract
let output_approximates (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (vp:chest2 bf16 length 1) (vh:chest2 bf16 16000 1)
  (wr:P.matrix FStar.Real.real (SZ.v g.Layout.rows) (SZ.v g.Layout.columns))
  (xr:P.vector FStar.Real.real (SZ.v g.Layout.columns))
  (row:P.index (SZ.v g.Layout.rows))
  : Lemma
      (requires O.matrix_approximates (Layout.matrix g vp) wr /\
        O.vector_approximates (Layout.vector g vh) xr)
      (ensures Approx.v_approximates
        (acc2 (Writes.after g vp vh (SZ.v g.Layout.rows)) (SZ.v g.Layout.output_offset + row) 0)
        (Matmul.project #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns) wr xr row)) =
  Writes.result_cell g vp vh (SZ.v g.Layout.output_offset + row);
  Matmul.output_approx #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
    (Layout.matrix g vp) (Layout.vector g vh) wr xr (SZ.v g.Layout.splits) row
