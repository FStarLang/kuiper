module KPlay.Proof.Base.MatvecNative

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor
open Kuiper.Array2.Strided

module SZ = Kuiper.SizeT
module C = Kuiper.Chest
module M = Kuiper.TensorCore.MMA
module V = Kuiper.TensorCore.MMA.Views
module Affine = KPlay.Proof.Base.AffineMMA
module Layout = KPlay.Proof.Base.MatvecLayout
module Matmul = KPlay.Proof.Base.Matmul
module Native = KPlay.Proof.Base.NativeProjection
module P = KPlay.Proof.Shared.Projection

inline_for_extraction noextract
fn instruction
  (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (row:sz{SZ.v row < SZ.v g.Layout.rows})
  (depth:sz{SZ.v depth < SZ.v g.Layout.columns})
  (parent:array2 bf16 (Affine.parent_layout length)) (hidden:array2 bf16 Native.hidden_layout)
  (fragment:M.fragment)
  (#vp:chest2 bf16 length 1) (#vh:chest2 bf16 16000 1) (#vc:chest2 float 16 8)
  (#parent_fraction #hidden_fraction:perm)
  preserves gpu ** parent |-> Frac parent_fraction vp ** hidden |-> Frac hidden_fraction vh
  requires fragment |-> vc
  ensures fragment |-> M.emma vc
    (Matmul.a_tile #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
      (Layout.matrix g vp) (SZ.v row) (SZ.v depth))
    (Matmul.b_tile #(SZ.v g.Layout.columns) (Layout.vector g vh) (SZ.v depth))
{
  Layout.tile_bounds g (SZ.v row) (SZ.v depth);
  Layout.operand_binding g vp vh (SZ.v row) (SZ.v depth);
  let m = if (SZ.add row 16sz <=^ g.Layout.rows) { 16sz }
    else { g.Layout.rows -^ row };
  let k = if (SZ.add depth 16sz <=^ g.Layout.columns) { 16sz }
    else { g.Layout.columns -^ depth };
  assert pure (SZ.v m == Layout.extent (SZ.v g.Layout.rows) (SZ.v row));
  assert pure (SZ.v k == Layout.extent (SZ.v g.Layout.columns) (SZ.v depth));
  let offset = SZ.add
    (SZ.add g.Layout.offset (SZ.mul row g.Layout.row_stride))
    (SZ.mul depth g.Layout.col_stride);
  let input_offset = SZ.add g.Layout.input_offset depth;
  Affine.instruction m k parent hidden offset g.Layout.row_stride g.Layout.col_stride
    input_offset fragment ();
}

inline_for_extraction noextract
let part_value (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (vp:chest2 bf16 length 1) (vh:chest2 bf16 16000 1)
  (row:nat) (part:nat{part < SZ.v g.Layout.splits}) : GTot (chest2 float 16 8) =
  Matmul.accumulator #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
    (Layout.matrix g vp) (Layout.vector g vh) row
    (part * (SZ.v g.Layout.columns / SZ.v g.Layout.splits))
    (Matmul.part_tiles (SZ.v g.Layout.columns) (SZ.v g.Layout.splits))

inline_for_extraction noextract
let rec masked_totals (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (vp:chest2 bf16 length 1) (vh:chest2 bf16 16000 1)
  (initial:chest2 float 16 8) (row:nat) (count:nat{count <= SZ.v g.Layout.splits})
  : GTot (chest2 float 16 8) (decreases count) =
  if count = 0 then V.filled 5 initial (zero #float)
  else V.added 5 (masked_totals g vp vh initial row (count - 1))
    (part_value g vp vh row (count - 1))

inline_for_extraction noextract
let rec masked_column (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (vp:chest2 bf16 length 1) (vh:chest2 bf16 16000 1)
  (initial:chest2 float 16 8) (row:nat) (count:nat{count <= SZ.v g.Layout.splits})
  (i:P.index 16)
  : Lemma (
      acc2 (masked_totals g vp vh initial row count) i 0 ==
        acc2 (Matmul.totals #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
          (Layout.matrix g vp) (Layout.vector g vh) row (SZ.v g.Layout.splits) count) i 0)
      (decreases count) =
  if count > 0 then masked_column g vp vh initial row (count - 1) i

inline_for_extraction noextract
fn accumulate_part
  (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (row:sz{SZ.v row < SZ.v g.Layout.rows})
  (part:sz{SZ.v part < SZ.v g.Layout.splits})
  (parent:array2 bf16 (Affine.parent_layout length)) (hidden:array2 bf16 Native.hidden_layout)
  (fragment:M.fragment)
  (#vp:chest2 bf16 length 1) (#vh:chest2 bf16 16000 1) (#vc:chest2 float 16 8)
  (#parent_fraction #hidden_fraction:perm)
  preserves gpu ** parent |-> Frac parent_fraction vp ** hidden |-> Frac hidden_fraction vh
  requires fragment |-> vc
  ensures fragment |-> part_value g vp vh (SZ.v row) (SZ.v part)
{
  Matmul.part_geometry (SZ.v g.Layout.columns) (SZ.v g.Layout.splits) (SZ.v part);
  let width = g.Layout.columns /^ g.Layout.splits;
  let start = SZ.mul part width;
  let finish = SZ.mul (SZ.add part 1sz) width;
  M.fill fragment (zero #float);
  C.ext (const (16 @| 8 @| INil) (zero #float))
    (Matmul.accumulator #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
      (Layout.matrix g vp) (Layout.vector g vh) (SZ.v row) (SZ.v start) 0);
  let mut depth = start;
  while (!depth <^ finish)
    invariant exists* (d:sz{SZ.v start <= SZ.v d /\ SZ.v d < SZ.v finish + 16 /\
      (SZ.v d - SZ.v start) % 16 == 0}).
      depth |-> d **
      fragment |-> Matmul.accumulator #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
        (Layout.matrix g vp) (Layout.vector g vh) (SZ.v row) (SZ.v start)
        ((SZ.v d - SZ.v start) / 16)
    invariant gpu ** parent |-> Frac parent_fraction vp ** hidden |-> Frac hidden_fraction vh
    decreases (SZ.v finish + 16 - SZ.v !depth)
  {
    let d = !depth;
    instruction g row d parent hidden fragment;
    let next = SZ.add d 16sz;
    assert pure (SZ.v d == SZ.v start + 16 * ((SZ.v d - SZ.v start) / 16));
    assert pure ((SZ.v next - SZ.v start) / 16 == (SZ.v d - SZ.v start) / 16 + 1);
    assert pure (
      M.emma (Matmul.accumulator #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
        (Layout.matrix g vp) (Layout.vector g vh) (SZ.v row) (SZ.v start)
        ((SZ.v d - SZ.v start) / 16))
        (Matmul.a_tile #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
          (Layout.matrix g vp) (SZ.v row) (SZ.v d))
        (Matmul.b_tile #(SZ.v g.Layout.columns) (Layout.vector g vh) (SZ.v d)) ==
      Matmul.accumulator #(SZ.v g.Layout.rows) #(SZ.v g.Layout.columns)
        (Layout.matrix g vp) (Layout.vector g vh) (SZ.v row) (SZ.v start)
        ((SZ.v next - SZ.v start) / 16));
    depth := next;
  };
  let d = !depth;
  assert pure ((SZ.v d - SZ.v start) / 16 ==
    Matmul.part_tiles (SZ.v g.Layout.columns) (SZ.v g.Layout.splits));
}

inline_for_extraction noextract
fn compute_block
  (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (row:sz{SZ.v row < SZ.v g.Layout.rows})
  (parent:array2 bf16 (Affine.parent_layout length)) (hidden:array2 bf16 Native.hidden_layout)
  (accum fragment:M.fragment)
  (#vp:chest2 bf16 length 1) (#vh:chest2 bf16 16000 1)
  (#initial #vc:chest2 float 16 8)
  (#parent_fraction #hidden_fraction:perm)
  preserves gpu ** parent |-> Frac parent_fraction vp ** hidden |-> Frac hidden_fraction vh
  requires accum |-> initial ** fragment |-> vc
  ensures accum |-> masked_totals g vp vh initial (SZ.v row) (SZ.v g.Layout.splits)
  ensures exists* (last:chest2 float 16 8). fragment |-> last
{
  V.fill_masked accum 5sz (zero #float);
  let mut part = 0sz;
  while (!part <^ g.Layout.splits)
    invariant exists* (p:sz{SZ.v p <= SZ.v g.Layout.splits}) (last:chest2 float 16 8).
      part |-> p ** fragment |-> last **
      accum |-> masked_totals g vp vh initial (SZ.v row) (SZ.v p)
    invariant gpu ** parent |-> Frac parent_fraction vp ** hidden |-> Frac hidden_fraction vh
    decreases (SZ.v g.Layout.splits - SZ.v !part)
  {
    let p = !part;
    accumulate_part g row p parent hidden fragment;
    V.add_masked accum fragment 5sz;
    assert pure (
      V.added 5 (masked_totals g vp vh initial (SZ.v row) (SZ.v p))
        (part_value g vp vh (SZ.v row) (SZ.v p)) ==
      masked_totals g vp vh initial (SZ.v row) (SZ.v p + 1));
    part := SZ.add p 1sz;
  };
  with p last. assert part |-> p ** fragment |-> last;
  assert pure (SZ.v p == SZ.v g.Layout.splits);
  rewrite accum |-> masked_totals g vp vh initial (SZ.v row) (SZ.v p)
    as accum |-> masked_totals g vp vh initial (SZ.v row) (SZ.v g.Layout.splits);
}
