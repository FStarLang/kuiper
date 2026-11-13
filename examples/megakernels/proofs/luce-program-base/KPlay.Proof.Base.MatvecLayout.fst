module KPlay.Proof.Base.MatvecLayout

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor
module SZ = Kuiper.SizeT
module C = Kuiper.Chest
module V = Kuiper.TensorCore.MMA.Views
module Affine = KPlay.Proof.Base.AffineMMA
module Matmul = KPlay.Proof.Base.Matmul
module P = KPlay.Proof.Shared.Projection
module Schedule = KPlay.Proof.Base.Schedule
module Model = KPlay.Proof.Shared.Model

noeq type geometry (length:pos{length <= 752393024}) = {
  rows: value:sz{0 < SZ.v value /\ SZ.v value <= 6144};
  columns: value:sz{0 < SZ.v value /\ SZ.v value <= 3584};
  offset: sz;
  row_stride: sz;
  col_stride: sz;
  input_offset: value:sz{SZ.v value + SZ.v columns <= 16000};
  output_offset: value:sz{SZ.v value + SZ.v rows <= 16000};
  splits: value:sz{SZ.v value = 1 \/ (SZ.v value = 2 /\ SZ.v columns % 32 = 0)};
  bounds: squash (Affine.fits length (SZ.v rows) (SZ.v columns)
    (SZ.v offset) (SZ.v row_stride) (SZ.v col_stride));
  separated: squash (
    SZ.v input_offset + SZ.v columns <= SZ.v output_offset \/
    SZ.v output_offset + SZ.v rows <= SZ.v input_offset);
}

inline_for_extraction noextract
let matrix (#length:pos{length <= 752393024}) (g:geometry length)
  (values:chest2 bf16 length 1) : P.matrix bf16 (SZ.v g.rows) (SZ.v g.columns) =
  fun row col -> acc2 values
    (SZ.v g.offset + row * SZ.v g.row_stride + col * SZ.v g.col_stride) 0

inline_for_extraction noextract
let vector (#length:pos{length <= 752393024}) (g:geometry length)
  (values:chest2 bf16 16000 1) : P.vector bf16 (SZ.v g.columns) =
  fun i -> acc2 values (SZ.v g.input_offset + i) 0

inline_for_extraction noextract
let extent (size:pos) (first:nat{first < size}) : (value:pos{value <= 16}) =
  if first + 16 <= size then 16 else size - first

inline_for_extraction noextract
let extent_bounds (size:pos) (first:nat{first < size})
  : Lemma (
      first + extent size first <= size /\
      (forall (i:P.index 16). i < extent size first <==> first + i < size)) = ()

inline_for_extraction noextract
let tile_offset (#length:pos{length <= 752393024}) (g:geometry length)
  (row:P.index (SZ.v g.rows)) (depth:P.index (SZ.v g.columns)) : P.index length =
  SZ.v g.offset + row * SZ.v g.row_stride + depth * SZ.v g.col_stride

inline_for_extraction noextract
let tile_bounds (#length:pos{length <= 752393024}) (g:geometry length)
  (row:P.index (SZ.v g.rows)) (depth:P.index (SZ.v g.columns))
  : Lemma (
      Affine.fits length (extent (SZ.v g.rows) row) (extent (SZ.v g.columns) depth)
        (tile_offset g row depth) (SZ.v g.row_stride) (SZ.v g.col_stride) /\
      SZ.v g.input_offset + depth + extent (SZ.v g.columns) depth <= 16000) =
  extent_bounds (SZ.v g.rows) row;
  extent_bounds (SZ.v g.columns) depth;
  let at_index (i:P.index (extent (SZ.v g.rows) row))
    (j:P.index (extent (SZ.v g.columns) depth))
    : Lemma (
        tile_offset g row depth + i * SZ.v g.row_stride + j * SZ.v g.col_stride < length) =
    assert (row + i < SZ.v g.rows /\ depth + j < SZ.v g.columns);
    assert (SZ.v g.offset + (row + i) * SZ.v g.row_stride +
      (depth + j) * SZ.v g.col_stride < length)
  in
  FStar.Classical.forall_intro_2 at_index

inline_for_extraction noextract
let operand_binding (#length:pos{length <= 752393024}) (g:geometry length)
  (vp:chest2 bf16 length 1) (vh:chest2 bf16 16000 1)
  (row:P.index (SZ.v g.rows)) (depth:P.index (SZ.v g.columns))
  : Lemma (
      V.pad_a (Affine.a_values (extent (SZ.v g.rows) row) (extent (SZ.v g.columns) depth)
        vp (tile_offset g row depth) (SZ.v g.row_stride) (SZ.v g.col_stride)
        (tile_bounds g row depth)) ==
        Matmul.a_tile #(SZ.v g.rows) #(SZ.v g.columns) (matrix g vp) row depth /\
      V.pad_b (Affine.h_values (extent (SZ.v g.columns) depth) vh (SZ.v g.input_offset + depth)) ==
        Matmul.b_tile #(SZ.v g.columns) (vector g vh) depth) =
  tile_bounds g row depth;
  extent_bounds (SZ.v g.rows) row;
  extent_bounds (SZ.v g.columns) depth;
  C.ext
    (V.pad_a (Affine.a_values (extent (SZ.v g.rows) row) (extent (SZ.v g.columns) depth)
      vp (tile_offset g row depth) (SZ.v g.row_stride) (SZ.v g.col_stride) ()))
    (Matmul.a_tile #(SZ.v g.rows) #(SZ.v g.columns) (matrix g vp) row depth);
  C.ext
    (V.pad_b (Affine.h_values (extent (SZ.v g.columns) depth) vh (SZ.v g.input_offset + depth)))
    (Matmul.b_tile #(SZ.v g.columns) (vector g vh) depth)

inline_for_extraction noextract
let source_nonoverlap (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Model.cursor) (phase:Schedule.phase layer)
  : Lemma (
      let d = Schedule.projection_at call position layer phase in
      d.input_offset + d.columns <= d.output_offset \/
      d.output_offset + (if layer = 24 then 16 else d.rows) <= d.input_offset) =
  Schedule.phase_kinds layer phase;
  if layer = 24 then ()
  else if Schedule.attention layer phase then ()
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
