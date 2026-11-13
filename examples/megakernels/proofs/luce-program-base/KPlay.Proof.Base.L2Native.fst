module KPlay.Proof.Base.L2Native

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
module F = Kuiper.Floating.Base
module Cast = Kuiper.Float.Casts.Base
module P = KPlay.Proof.Shared.Projection
module Norm = KPlay.Proof.Base.Norm
module Prior = KPlay.Proof.Base.NormNative
module RMS = KPlay.Proof.Base.RMSNative
module L2 = KPlay.Proof.Base.L2
module Reduction = KPlay.Proof.Base.ReductionNative
module Transfer = KPlay.Proof.Base.Transfer
module Local = KPlay.Proof.Base.LocalArray
module Init = KPlay.Proof.Base.Initialization

inline_for_extraction noextract
fn square_group
  (hidden:larray BF.t 16000) (offset:sz{SZ.v offset + 128 <= 16000})
  (group:sz{SZ.v group < 16}) (#before:erased Transfer.scratch) (#fraction:perm)
  preserves gpu ** A.pts_to hidden #fraction before
  returns result:FP.t
  ensures pure (result == L2.group (RMS.input 128 before (SZ.v offset)) (SZ.v group) 8)
{
  let mut j = 0sz;
  let mut sum:FP.t = F.zero #FP.t;
  while (!j <^ 8sz)
    invariant exists* (count:sz{SZ.v count <= 8}) (current:FP.t).
      j |-> count ** sum |-> current **
      pure (current == L2.group (RMS.input 128 before (SZ.v offset)) (SZ.v group) (SZ.v count))
    invariant gpu ** A.pts_to hidden #fraction before
    decreases (8 - SZ.v !j)
  {
    let count = !j;
    let value = hidden.(SZ.add offset (SZ.add (SZ.mul group 8sz) count));
    let wide = Cast.cast_bf16_to_f32 value;
    let square = Cast.cast_bf16_to_f32 (Cast.cast_f32_to_bf16 (F.mul wide wide));
    let current = !sum;
    norm_spec [delta_once ["KPlay.Proof.Base.L2.group"]]
      (L2.group (RMS.input 128 before (SZ.v offset)) (SZ.v group) (SZ.v count + 1));
    sum := F.add current square;
    j := SZ.add count 1sz;
  };
  !sum
}

inline_for_extraction noextract
let grouped (before:lseq FP.t 16) (hidden:Transfer.scratch)
  (offset:nat{offset + 128 <= 16000}) (count:nat{count <= 16}) : GTot (lseq FP.t 16) =
  Seq.init_ghost 16 (fun i ->
    if i < count then L2.group (RMS.input 128 hidden offset) i 8 else Seq.index before i)

inline_for_extraction noextract
let grouped_zero (before:lseq FP.t 16) (hidden:Transfer.scratch)
  (offset:nat{offset + 128 <= 16000})
  : Lemma (grouped before hidden offset 0 == before) =
  assert (Seq.equal (grouped before hidden offset 0) before);
  Seq.lemma_eq_elim (grouped before hidden offset 0) before

inline_for_extraction noextract
let grouped_step (before:lseq FP.t 16) (hidden:Transfer.scratch)
  (offset:nat{offset + 128 <= 16000}) (i:P.index 16)
  : Lemma (
      Seq.upd (grouped before hidden offset i) i (L2.group (RMS.input 128 hidden offset) i 8) ==
      grouped before hidden offset (i + 1)) =
  assert (Seq.equal
    (Seq.upd (grouped before hidden offset i) i (L2.group (RMS.input 128 hidden offset) i 8))
    (grouped before hidden offset (i + 1)));
  Seq.lemma_eq_elim
    (Seq.upd (grouped before hidden offset i) i (L2.group (RMS.input 128 hidden offset) i 8))
    (grouped before hidden offset (i + 1))

inline_for_extraction noextract
fn scale
  (hidden:larray BF.t 16000) (offset:sz{SZ.v offset + 128 <= 16000}) (partial:larray FP.t 16)
  (#before:erased Transfer.scratch) (#temporary:erased (lseq FP.t 16)) (#fraction:perm)
  preserves gpu ** A.pts_to hidden #fraction before
  requires partial |-> temporary
  returns result:FP.t
  ensures exists* (after:lseq FP.t 16). partial |-> after **
    pure (result == L2.source_scale (RMS.input 128 before (SZ.v offset)))
{
  grouped_zero temporary before (SZ.v offset);
  let mut i = 0sz;
  while (!i <^ 16sz)
    invariant exists* (count:sz{SZ.v count <= 16}).
      i |-> count ** partial |-> grouped temporary before (SZ.v offset) (SZ.v count)
    invariant gpu ** A.pts_to hidden #fraction before
    decreases (16 - SZ.v !i)
  {
    let count = !i;
    let value = square_group hidden offset count;
    grouped_step temporary before (SZ.v offset) (SZ.v count);
    partial.(count) <- value;
    i := SZ.add count 1sz;
  };
  let sum = Reduction.tree partial 16sz;
  Prior.tree_ext 16 (Reduction.values (grouped temporary before (SZ.v offset) 16))
    (L2.grouped (RMS.input 128 before (SZ.v offset)));
  let rounded = Cast.cast_bf16_to_f32 (Cast.cast_f32_to_bf16 sum);
  let shifted = Cast.cast_bf16_to_f32 (Cast.cast_f32_to_bf16 (F.add rounded Norm.epsilon));
  Cast.cast_bf16_to_f32 (Cast.cast_f32_to_bf16 (F.rsqrt shifted))
}

inline_for_extraction noextract
let square_values (hidden:Transfer.scratch) (offset:nat{offset + 128 <= 16000}) : P.vector FP.t 16 =
  fun i -> L2.group (RMS.input 128 hidden offset) i 8

inline_for_extraction noextract
fn source_scale
  (hidden:larray BF.t 16000) (offset:sz{SZ.v offset + 128 <= 16000})
  (#before:erased Transfer.scratch) (#fraction:perm)
  preserves gpu ** A.pts_to hidden #fraction before
  returns result:FP.t
  ensures pure (result == L2.source_scale (RMS.input 128 before (SZ.v offset)))
{
  let mut partial : A.array FP.t = [| 16sz |];
  with initial. assert A.pts_to_mask partial initial Init.all_cells;
  let original : erased (Local.snapshot 16) = hide #(Local.snapshot 16) initial;
  Local.written_zero #16 #16 original (square_values before (SZ.v offset));
  let mut i = 0sz;
  while (!i <^ 16sz)
    invariant exists* (count:sz{SZ.v count <= 16}).
      i |-> count ** A.pts_to_mask partial
        (Local.written #16 #16 original (square_values before (SZ.v offset)) (SZ.v count)) Init.all_cells
    invariant gpu ** A.pts_to hidden #fraction before
    decreases (16 - SZ.v !i)
  {
    let count = !i;
    let value = square_group hidden offset count;
    Local.written_step #16 #16 original (square_values before (SZ.v offset)) (SZ.v count);
    A.mask_write partial count value;
    i := SZ.add count 1sz;
  };
  Local.written_ready #16 #16 original (square_values before (SZ.v offset)) 16;
  let sum = Local.reduce #16 partial 16sz false #(Local.written #16 #16 original (square_values before (SZ.v offset)) 16) ();
  Prior.tree_ext 16
    (Reduction.values (Local.prefix_values (Local.written #16 #16 original (square_values before (SZ.v offset)) 16) 16 ()))
    (L2.grouped (RMS.input 128 before (SZ.v offset)));
  let rounded = Cast.cast_bf16_to_f32 (Cast.cast_f32_to_bf16 sum);
  let shifted = Cast.cast_bf16_to_f32 (Cast.cast_f32_to_bf16 (F.add rounded Norm.epsilon));
  Cast.cast_bf16_to_f32 (Cast.cast_f32_to_bf16 (F.rsqrt shifted))
}
