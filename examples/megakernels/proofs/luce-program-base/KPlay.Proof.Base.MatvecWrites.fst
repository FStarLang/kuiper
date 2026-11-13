module KPlay.Proof.Base.MatvecWrites

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor
module SZ = Kuiper.SizeT
module C = Kuiper.Chest
module Layout = KPlay.Proof.Base.MatvecLayout
module Matmul = KPlay.Proof.Base.Matmul
module Scratch = KPlay.Proof.Base.Scratch
module P = KPlay.Proof.Shared.Projection

inline_for_extraction noextract
let column (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (vp:chest2 bf16 length 1) (vh:chest2 bf16 16000 1)
  (row:P.index (SZ.v g.Layout.rows))
  : GTot (chest2 bf16 (Layout.extent (SZ.v g.Layout.rows) row) 1) =
  Layout.extent_bounds (SZ.v g.Layout.rows) row;
  mk2 (fun i _ -> Matmul.output #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
    (Layout.matrix g vp) (Layout.vector g vh) (SZ.v g.Layout.splits) (row + i))

inline_for_extraction noextract
let published (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (vp:chest2 bf16 length 1) (vh:chest2 bf16 16000 1)
  (row:P.index (SZ.v g.Layout.rows)) : GTot (chest2 bf16 16000 1) =
  Layout.extent_bounds (SZ.v g.Layout.rows) row;
  Scratch.replace_window vh (SZ.v g.Layout.output_offset + row)
    (Layout.extent (SZ.v g.Layout.rows) row) (column g vp vh row)

inline_for_extraction noextract
let after (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (vp:chest2 bf16 length 1) (vh:chest2 bf16 16000 1)
  (count:nat{count <= SZ.v g.Layout.rows}) : GTot (chest2 bf16 16000 1) =
  mk2 (fun i _ ->
    if SZ.v g.Layout.output_offset <= i && i < SZ.v g.Layout.output_offset + count
    then Matmul.output #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
      (Layout.matrix g vp) (Layout.vector g vh) (SZ.v g.Layout.splits)
      (i - SZ.v g.Layout.output_offset)
    else acc2 vh i 0)

inline_for_extraction noextract
let after_zero (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (vp:chest2 bf16 length 1) (vh:chest2 bf16 16000 1)
  : Lemma (after g vp vh 0 == vh) =
  C.ext (after g vp vh 0) vh

inline_for_extraction noextract
let input_frame (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (vp:chest2 bf16 length 1) (vh:chest2 bf16 16000 1)
  (count:nat{count <= SZ.v g.Layout.rows}) (i:P.index (SZ.v g.Layout.columns))
  : Lemma (
      Layout.vector g (after g vp vh count) i == Layout.vector g vh i) = ()

inline_for_extraction noextract
let tile_frame (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (vp:chest2 bf16 length 1) (vh:chest2 bf16 16000 1)
  (count:nat{count <= SZ.v g.Layout.rows}) (depth:nat)
  : Lemma (
      Matmul.b_tile #(SZ.v g.Layout.columns) (Layout.vector g (after g vp vh count)) depth ==
      Matmul.b_tile #(SZ.v g.Layout.columns) (Layout.vector g vh) depth) =
  FStar.Classical.forall_intro (input_frame g vp vh count);
  C.ext (Matmul.b_tile #(SZ.v g.Layout.columns) (Layout.vector g (after g vp vh count)) depth)
    (Matmul.b_tile #(SZ.v g.Layout.columns) (Layout.vector g vh) depth)

inline_for_extraction noextract
let rec accumulator_frame (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (vp:chest2 bf16 length 1) (vh:chest2 bf16 16000 1)
  (count:nat{count <= SZ.v g.Layout.rows}) (row start tiles:nat)
  : Lemma (
      Matmul.accumulator #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
        (Layout.matrix g vp) (Layout.vector g (after g vp vh count)) row start tiles ==
      Matmul.accumulator #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
        (Layout.matrix g vp) (Layout.vector g vh) row start tiles)
      (decreases tiles) =
  if tiles > 0 then (
    accumulator_frame g vp vh count row start (tiles - 1);
    tile_frame g vp vh count (start + 16 * (tiles - 1))
  )

inline_for_extraction noextract
let rec totals_frame (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (vp:chest2 bf16 length 1) (vh:chest2 bf16 16000 1)
  (count:nat{count <= SZ.v g.Layout.rows}) (row:nat)
  (parts:nat{parts <= SZ.v g.Layout.splits})
  : Lemma (
      Matmul.totals #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
        (Layout.matrix g vp) (Layout.vector g (after g vp vh count))
        row (SZ.v g.Layout.splits) parts ==
      Matmul.totals #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
        (Layout.matrix g vp) (Layout.vector g vh) row (SZ.v g.Layout.splits) parts)
      (decreases parts) =
  norm_spec [delta_once ["KPlay.Proof.Base.Matmul.totals"]]
    (Matmul.totals #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
      (Layout.matrix g vp) (Layout.vector g (after g vp vh count)) row (SZ.v g.Layout.splits) parts);
  norm_spec [delta_once ["KPlay.Proof.Base.Matmul.totals"]]
    (Matmul.totals #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
      (Layout.matrix g vp) (Layout.vector g vh) row (SZ.v g.Layout.splits) parts);
  if parts > 0 then (
    totals_frame g vp vh count row (parts - 1);
    accumulator_frame g vp vh count row
      ((parts - 1) * (SZ.v g.Layout.columns / SZ.v g.Layout.splits))
      (Matmul.part_tiles (SZ.v g.Layout.columns) (SZ.v g.Layout.splits))
  );
  C.ext
    (Matmul.totals #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
      (Layout.matrix g vp) (Layout.vector g (after g vp vh count)) row (SZ.v g.Layout.splits) parts)
    (Matmul.totals #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
      (Layout.matrix g vp) (Layout.vector g vh) row (SZ.v g.Layout.splits) parts)

inline_for_extraction noextract
let output_frame (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (vp:chest2 bf16 length 1) (vh:chest2 bf16 16000 1)
  (count:nat{count <= SZ.v g.Layout.rows}) (row:P.index (SZ.v g.Layout.rows))
  : Lemma (
      Matmul.output #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
        (Layout.matrix g vp) (Layout.vector g (after g vp vh count)) (SZ.v g.Layout.splits) row ==
      Matmul.output #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
        (Layout.matrix g vp) (Layout.vector g vh) (SZ.v g.Layout.splits) row) =
  totals_frame g vp vh count ((row / 16) * 16) (SZ.v g.Layout.splits)

inline_for_extraction noextract
let block_step (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (vp:chest2 bf16 length 1) (vh:chest2 bf16 16000 1)
  (row:P.index (SZ.v g.Layout.rows))
  : Lemma (
      published g vp (after g vp vh row) row ==
        after g vp vh (row + Layout.extent (SZ.v g.Layout.rows) row)) =
  Layout.extent_bounds (SZ.v g.Layout.rows) row;
  FStar.Classical.forall_intro (output_frame g vp vh row);
  C.ext (published g vp (after g vp vh row) row)
    (after g vp vh (row + Layout.extent (SZ.v g.Layout.rows) row))

inline_for_extraction noextract
let result_cell (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (vp:chest2 bf16 length 1) (vh:chest2 bf16 16000 1)
  (i:P.index 16000)
  : Lemma (
      let result = after g vp vh (SZ.v g.Layout.rows) in
      (SZ.v g.Layout.output_offset <= i /\ i < SZ.v g.Layout.output_offset + SZ.v g.Layout.rows ==>
        acc2 result i 0 == Matmul.output #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
          (Layout.matrix g vp) (Layout.vector g vh) (SZ.v g.Layout.splits)
          (i - SZ.v g.Layout.output_offset)) /\
      (~(SZ.v g.Layout.output_offset <= i /\ i < SZ.v g.Layout.output_offset + SZ.v g.Layout.rows) ==>
        acc2 result i 0 == acc2 vh i 0)) = ()
