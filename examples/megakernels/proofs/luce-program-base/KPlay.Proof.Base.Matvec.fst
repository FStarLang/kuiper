module KPlay.Proof.Base.Matvec

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
module Compute = KPlay.Proof.Base.MatvecNative
module Writes = KPlay.Proof.Base.MatvecWrites
module Scratch = KPlay.Proof.Base.Scratch
module P = KPlay.Proof.Shared.Projection

inline_for_extraction noextract
let window_layout (offset:nat) (count:pos{offset + count <= 16000})
  : Scratch.window_layout offset count = {
  ulen = 16000;
  imap = Kuiper.Injection.mk_injection #(abs (count @| 1 @| INil)) #(natlt 16000)
    (fun (index:abs (count @| 1 @| INil)) -> let i, (_, ()) = index in offset + i)
    (fun x y ->
      let i, (j, ()) = x in
      let i', (j', ()) = y in
      assert (j == 0 /\ j' == 0 /\ i == i'));
}

inline_for_extraction noextract
instance window_column (offset:nat) (count:pos{offset + count <= 16000})
  : strided_col_major (window_layout offset count) = {
  offset = SZ.uint_to_t offset;
  stride = SZ.uint_to_t count;
  pf = (fun i j -> ());
}

inline_for_extraction noextract
let stored_binding (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (vp:chest2 bf16 length 1) (vh:chest2 bf16 16000 1)
  (initial:chest2 float 16 8) (row:P.index (SZ.v g.Layout.rows){row % 16 = 0})
  : Lemma (
      V.rounded_window 0 0 (Layout.extent (SZ.v g.Layout.rows) row) 1
        (Compute.masked_totals g vp vh initial row (SZ.v g.Layout.splits)) ==
      Writes.column g vp vh row) =
  Layout.extent_bounds (SZ.v g.Layout.rows) row;
  FStar.Classical.forall_intro
    (Compute.masked_column g vp vh initial row (SZ.v g.Layout.splits));
  let at_index (i:P.index (Layout.extent (SZ.v g.Layout.rows) row))
    : Lemma (
        (row + i) / 16 * 16 == row /\ (row + i) % 16 == i) = ()
  in
  FStar.Classical.forall_intro at_index;
  C.ext
    (V.rounded_window 0 0 (Layout.extent (SZ.v g.Layout.rows) row) 1
      (Compute.masked_totals g vp vh initial row (SZ.v g.Layout.splits)))
    (Writes.column g vp vh row)

inline_for_extraction noextract
fn block
  (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (row:sz{SZ.v row < SZ.v g.Layout.rows /\ SZ.v row % 16 == 0})
  (parent:array2 bf16 (Affine.parent_layout length)) (hidden:array2 bf16 Native.hidden_layout)
  (#vp:chest2 bf16 length 1) (#vh:chest2 bf16 16000 1)
  (#parent_fraction:perm)
  preserves gpu ** parent |-> Frac parent_fraction vp
  requires hidden |-> Frac (1.0R /. 32) vh
  ensures hidden |-> Frac (1.0R /. 32) (Writes.published g vp vh (SZ.v row))
{
  let accum = M.alloc_fragment ();
  let fragment = M.alloc_fragment ();
  with initial vc. assert accum |-> initial ** fragment |-> vc;
  Compute.compute_block g row parent hidden accum fragment;
  let m = if (SZ.add row 16sz <=^ g.Layout.rows) { 16sz }
    else { g.Layout.rows -^ row };
  let offset = SZ.add g.Layout.output_offset row;
  assert pure (SZ.v m == Layout.extent (SZ.v g.Layout.rows) (SZ.v row));
  stored_binding g vp vh initial (SZ.v row);
  Scratch.extract_window hidden (SZ.v offset) (SZ.v m) (window_layout (SZ.v offset) (SZ.v m));
  V.store_bf16 accum 0sz 0sz m 1sz
    (Scratch.window_view hidden (SZ.v offset) (SZ.v m) (window_layout (SZ.v offset) (SZ.v m)));
  Scratch.restore_window hidden (SZ.v offset) (SZ.v m) (window_layout (SZ.v offset) (SZ.v m));
  with av pv. assert accum |-> av ** fragment |-> pv;
  drop_ (accum |-> av);
  drop_ (fragment |-> pv);
}

inline_for_extraction noextract
let prefix (#length:pos{length <= 752393024}) (g:Layout.geometry length) (row:nat)
  : (count:nat{count <= SZ.v g.Layout.rows}) =
  if row < SZ.v g.Layout.rows then row else SZ.v g.Layout.rows

inline_for_extraction noextract
fn projection
  (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (parent:array2 bf16 (Affine.parent_layout length)) (hidden:array2 bf16 Native.hidden_layout)
  (#vp:chest2 bf16 length 1) (#vh:chest2 bf16 16000 1)
  (#parent_fraction:perm)
  preserves gpu ** parent |-> Frac parent_fraction vp
  requires hidden |-> Frac (1.0R /. 32) vh
  ensures hidden |-> Frac (1.0R /. 32) (Writes.after g vp vh (SZ.v g.Layout.rows))
{
  Writes.after_zero g vp vh;
  let mut row = 0sz;
  while (!row <^ g.Layout.rows)
    invariant exists* (r:sz{SZ.v r < SZ.v g.Layout.rows + 16 /\ SZ.v r % 16 == 0}).
      row |-> r ** hidden |-> Frac (1.0R /. 32) (Writes.after g vp vh (prefix g (SZ.v r)))
    invariant gpu ** parent |-> Frac parent_fraction vp
    decreases (SZ.v g.Layout.rows + 16 - SZ.v !row)
  {
    let r = !row;
    assert pure (prefix g (SZ.v r) == SZ.v r);
    block g r parent hidden;
    Writes.block_step g vp vh (SZ.v r);
    let next = SZ.add r 16sz;
    assert pure (prefix g (SZ.v next) == SZ.v r + Layout.extent (SZ.v g.Layout.rows) (SZ.v r));
    rewrite hidden |-> Frac (1.0R /. 32) (Writes.published g vp (Writes.after g vp vh (SZ.v r)) (SZ.v r))
      as hidden |-> Frac (1.0R /. 32) (Writes.after g vp vh (prefix g (SZ.v next)));
    row := next;
  };
  let r = !row;
  assert pure (prefix g (SZ.v r) == SZ.v g.Layout.rows);
}
