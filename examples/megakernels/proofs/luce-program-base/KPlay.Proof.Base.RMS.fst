module KPlay.Proof.Base.RMS

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Array
open FStar.Ghost

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module P = KPlay.Proof.Shared.Projection
module Native = KPlay.Proof.Base.RMSNative
module Norm = KPlay.Proof.Base.Norm
module Tree = KPlay.Proof.Base.NormNative
module Transfer = KPlay.Proof.Base.Transfer

noeq type layout = {
  width: n:sz{SZ.v n = 128 \/ SZ.v n = 256 \/ SZ.v n = 1024};
  rows: n:sz{0 < SZ.v n /\ SZ.v n <= 8};
  source: sz;
  destination: sz;
  stride: sz;
  weight_offset: value:sz{SZ.v value + SZ.v width <= 752393024};
  bounds: squash (
    SZ.v source + (SZ.v rows - 1) * SZ.v stride + SZ.v width <= 16000 /\
    SZ.v destination + SZ.v rows * SZ.v width <= 16000 /\
    ((SZ.v destination <= SZ.v source /\ SZ.v width <= SZ.v stride) \/
      SZ.v source + (SZ.v rows - 1) * SZ.v stride + SZ.v width <= SZ.v destination));
}

inline_for_extraction noextract
let row_bounds (g:layout) (row:P.index (SZ.v g.rows))
  : Lemma (
      row * SZ.v g.stride < 16000 /\ row * SZ.v g.width < 16000 /\
      SZ.v g.source + row * SZ.v g.stride + SZ.v g.width <= 16000 /\
      SZ.v g.destination + row * SZ.v g.width + SZ.v g.width <= 16000 /\
      (SZ.v g.destination + row * SZ.v g.width <= SZ.v g.source + row * SZ.v g.stride \/
        SZ.v g.source + row * SZ.v g.stride + SZ.v g.width <= SZ.v g.destination + row * SZ.v g.width)) = ()

inline_for_extraction noextract
let rec after (g:layout) (before:Transfer.scratch) (weights:Transfer.packed)
  (count:nat{count <= SZ.v g.rows}) : GTot Transfer.scratch (decreases count) =
  if count = 0 then before
  else (
    row_bounds g (count - 1);
    Native.row_writes (SZ.v g.width) (after g before weights (count - 1)) weights
      (SZ.v g.source + (count - 1) * SZ.v g.stride)
      (SZ.v g.destination + (count - 1) * SZ.v g.width)
      (SZ.v g.weight_offset) (SZ.v g.width)
  )

inline_for_extraction noextract
let step (g:layout) (before:Transfer.scratch) (weights:Transfer.packed)
  (count:nat{count < SZ.v g.rows})
  : Lemma (
      Native.row_writes (SZ.v g.width) (after g before weights count) weights
        (SZ.v g.source + count * SZ.v g.stride) (SZ.v g.destination + count * SZ.v g.width)
        (SZ.v g.weight_offset) (SZ.v g.width) ==
      after g before weights (count + 1)) =
  row_bounds g count;
  norm_spec [delta_once ["KPlay.Proof.Base.RMS.after"]] (after g before weights (count + 1))

inline_for_extraction noextract
let rec unread_frame (g:layout) (before:Transfer.scratch) (weights:Transfer.packed)
  (count:nat{count <= SZ.v g.rows}) (row:P.index (SZ.v g.rows){count <= row})
  (i:P.index (SZ.v g.width))
  : Lemma (
      Seq.index (after g before weights count) (SZ.v g.source + row * SZ.v g.stride + i) ==
      Seq.index before (SZ.v g.source + row * SZ.v g.stride + i))
      (decreases count) =
  row_bounds g row;
  if count > 0 then (
    unread_frame g before weights (count - 1) row i;
    row_bounds g (count - 1);
    assert (
      SZ.v g.source + row * SZ.v g.stride + i <
        SZ.v g.destination + (count - 1) * SZ.v g.width \/
      SZ.v g.destination + count * SZ.v g.width <=
        SZ.v g.source + row * SZ.v g.stride + i);
    norm_spec [delta_once ["KPlay.Proof.Base.RMS.after"]] (after g before weights count)
  )

inline_for_extraction noextract
let rec group_equal (width:Norm.width) (left right:P.vector BF.t width)
  (group:nat{group < width / 4}) (count:nat{count <= 4})
  : Lemma
      (requires forall (i:P.index width). left i == right i)
      (ensures Norm.group width left group count == Norm.group width right group count)
      (decreases count) =
  if count > 0 then group_equal width left right group (count - 1)

inline_for_extraction noextract
let scale_equal (width:Norm.width) (left right:P.vector BF.t width)
  : Lemma
      (requires forall (i:P.index width). left i == right i)
      (ensures Norm.source_scale width left == Norm.source_scale width right) =
  let at_index (i:nat{i < width / 4})
    : Lemma (Norm.grouped width left i == Norm.grouped width right i) =
    group_equal width left right i 4
  in
  FStar.Classical.forall_intro at_index;
  Tree.tree_ext (width / 4) (Norm.grouped width left) (Norm.grouped width right)

inline_for_extraction noextract
fn apply
  (g:layout) (weights:larray BF.t 752393024) (hidden:larray BF.t 16000) (partial:larray FP.t 256)
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch)
  (#temporary:erased (lseq FP.t 256)) (#fraction:perm)
  preserves gpu ** A.pts_to weights #fraction vw
  requires A.pts_to hidden vh ** partial |-> temporary
  ensures exists* (next:lseq FP.t 256). partial |-> next **
    A.pts_to hidden (after g vh vw (SZ.v g.rows))
{
  let mut row = 0sz;
  while (!row <^ g.rows)
    invariant exists* (r:sz{SZ.v r <= SZ.v g.rows}) (current:lseq FP.t 256).
      row |-> r ** A.pts_to hidden (after g vh vw (SZ.v r)) ** partial |-> current
    invariant gpu ** A.pts_to weights #fraction vw
    decreases (SZ.v g.rows - SZ.v !row)
  {
    let r = !row;
    row_bounds g (SZ.v r);
    let source = SZ.add g.source (SZ.mul r g.stride);
    let destination = SZ.add g.destination (SZ.mul r g.width);
    Native.apply_row g.width weights hidden partial source destination g.weight_offset;
    step g vh vw (SZ.v r);
    row := SZ.add r 1sz;
  }

}

inline_for_extraction noextract
fn source_apply
  (g:layout) (weights:larray BF.t 752393024) (hidden:larray BF.t 16000)
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch) (#fraction:perm)
  preserves gpu ** A.pts_to weights #fraction vw
  requires A.pts_to hidden vh
  ensures A.pts_to hidden (after g vh vw (SZ.v g.rows))
{
  let mut row = 0sz;
  while (!row <^ g.rows)
    invariant exists* (r:sz{SZ.v r <= SZ.v g.rows}).
      row |-> r ** A.pts_to hidden (after g vh vw (SZ.v r))
    invariant gpu ** A.pts_to weights #fraction vw
    decreases (SZ.v g.rows - SZ.v !row)
  {
    let r = !row;
    row_bounds g (SZ.v r);
    let source = SZ.add g.source (SZ.mul r g.stride);
    let destination = SZ.add g.destination (SZ.mul r g.width);
    Native.source_row g.width weights hidden source destination g.weight_offset;
    step g vh vw (SZ.v r);
    row := SZ.add r 1sz;
  }
}
