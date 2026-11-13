module KPlay.Proof.Base.RMSNative

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
module IC = FStar.Int.Cast
module I64 = FStar.Int64
module P = KPlay.Proof.Shared.Projection
module Norm = KPlay.Proof.Base.Norm
module Prior = KPlay.Proof.Base.NormNative
module Reduction = KPlay.Proof.Base.ReductionNative
module Transfer = KPlay.Proof.Base.Transfer
module Local = KPlay.Proof.Base.LocalArray
module Init = KPlay.Proof.Base.Initialization

inline_for_extraction noextract
let input (width:Norm.width) (before:Transfer.scratch)
  (offset:nat{offset + width <= 16000}) : P.vector BF.t width =
  fun i -> Seq.index before (offset + i)

inline_for_extraction noextract
let grouped (width:Norm.width) (before:lseq FP.t 256) (hidden:Transfer.scratch)
  (offset:nat{offset + width <= 16000}) (count:nat{count <= width / 4}) : GTot (lseq FP.t 256) =
  Seq.init_ghost 256 (fun i ->
    if i < count then Norm.group width (input width hidden offset) i 4 else Seq.index before i)

inline_for_extraction noextract
let grouped_zero (width:Norm.width) (before:lseq FP.t 256) (hidden:Transfer.scratch)
  (offset:nat{offset + width <= 16000})
  : Lemma (grouped width before hidden offset 0 == before) =
  assert (Seq.equal (grouped width before hidden offset 0) before);
  Seq.lemma_eq_elim (grouped width before hidden offset 0) before

inline_for_extraction noextract
let grouped_step (width:Norm.width) (before:lseq FP.t 256) (hidden:Transfer.scratch)
  (offset:nat{offset + width <= 16000}) (i:nat{i < width / 4})
  : Lemma (
      Seq.upd (grouped width before hidden offset i) i
        (Norm.group width (input width hidden offset) i 4) ==
      grouped width before hidden offset (i + 1)) =
  assert (Seq.equal
    (Seq.upd (grouped width before hidden offset i) i
      (Norm.group width (input width hidden offset) i 4))
    (grouped width before hidden offset (i + 1)));
  Seq.lemma_eq_elim
    (Seq.upd (grouped width before hidden offset i) i
      (Norm.group width (input width hidden offset) i 4))
    (grouped width before hidden offset (i + 1))

inline_for_extraction noextract
fn square_group
  (width:sz{SZ.v width = 128 \/ SZ.v width = 256 \/ SZ.v width = 1024})
  (hidden:larray BF.t 16000) (offset:sz{SZ.v offset + SZ.v width <= 16000})
  (group:sz{SZ.v group < SZ.v width / 4})
  (#before:erased Transfer.scratch) (#fraction:perm)
  preserves gpu ** A.pts_to hidden #fraction before
  returns result:FP.t
  ensures pure (result == Norm.group (SZ.v width) (input (SZ.v width) before (SZ.v offset)) (SZ.v group) 4)
{
  let mut j = 0sz;
  let mut sum:FP.t = F.zero #FP.t;
  while (!j <^ 4sz)
    invariant exists* (count:sz{SZ.v count <= 4}) (current:FP.t).
      j |-> count ** sum |-> current **
      pure (current == Norm.group (SZ.v width) (input (SZ.v width) before (SZ.v offset))
        (SZ.v group) (SZ.v count))
    invariant gpu ** A.pts_to hidden #fraction before
    decreases (4 - SZ.v !j)
  {
    let count = !j;
    let value = hidden.(SZ.add offset (SZ.add (SZ.mul group 4sz) count));
    let wide = Cast.cast_bf16_to_f32 value;
    let current = !sum;
    norm_spec [delta_once ["KPlay.Proof.Base.Norm.group"]]
      (Norm.group (SZ.v width) (input (SZ.v width) before (SZ.v offset)) (SZ.v group) (SZ.v count + 1));
    sum := F.add current (F.mul wide wide);
    j := SZ.add count 1sz;
  };
  !sum
}

inline_for_extraction noextract
fn scale
  (width:sz{SZ.v width = 128 \/ SZ.v width = 256 \/ SZ.v width = 1024})
  (hidden:larray BF.t 16000) (offset:sz{SZ.v offset + SZ.v width <= 16000})
  (partial:larray FP.t 256)
  (#before:erased Transfer.scratch) (#temporary:erased (lseq FP.t 256)) (#fraction:perm)
  preserves gpu ** A.pts_to hidden #fraction before
  requires partial |-> temporary
  returns result:FP.t
  ensures exists* (after:lseq FP.t 256). partial |-> after **
    pure (result == Norm.source_scale (SZ.v width) (input (SZ.v width) before (SZ.v offset)))
{
  grouped_zero (SZ.v width) temporary before (SZ.v offset);
  let groups = width /^ 4sz;
  let mut i = 0sz;
  while (!i <^ groups)
    invariant exists* (count:sz{SZ.v count <= SZ.v groups}).
      i |-> count ** partial |-> grouped (SZ.v width) temporary before (SZ.v offset) (SZ.v count)
    invariant gpu ** A.pts_to hidden #fraction before
    decreases (SZ.v groups - SZ.v !i)
  {
    let count = !i;
    let value = square_group width hidden offset count;
    grouped_step (SZ.v width) temporary before (SZ.v offset) (SZ.v count);
    partial.(count) <- value;
    i := SZ.add count 1sz;
  };
  let sum = Reduction.tree partial groups;
  Prior.tree_ext (SZ.v groups)
    (Reduction.values (grouped (SZ.v width) temporary before (SZ.v offset) (SZ.v groups)))
    (Norm.grouped (SZ.v width) (input (SZ.v width) before (SZ.v offset)));
  let integer_width = IC.uint32_to_int64 (SZ.sizet_to_u32 width);
  let reciprocal = F.div (F.one #FP.t) (F.of_int #FP.t integer_width);
  assert pure (reciprocal == Norm.reciprocal (SZ.v width));
  F.rsqrt (F.add (F.mul sum reciprocal) Norm.epsilon)
}

inline_for_extraction noextract
let square_values (width:Norm.width) (hidden:Transfer.scratch) (offset:nat{offset + width <= 16000})
  : P.vector FP.t (width / 4) =
  fun i -> Norm.group width (input width hidden offset) i 4

inline_for_extraction noextract
fn source_scale
  (width:sz{SZ.v width = 128 \/ SZ.v width = 256 \/ SZ.v width = 1024})
  (hidden:larray BF.t 16000) (offset:sz{SZ.v offset + SZ.v width <= 16000})
  (#before:erased Transfer.scratch) (#fraction:perm)
  preserves gpu ** A.pts_to hidden #fraction before
  returns result:FP.t
  ensures pure (result == Norm.source_scale (SZ.v width) (input (SZ.v width) before (SZ.v offset)))
{
  let mut partial : A.array FP.t = [| 256sz |];
  with initial. assert A.pts_to_mask partial initial Init.all_cells;
  let original : erased (Local.snapshot 256) = hide #(Local.snapshot 256) initial;
  Local.written_zero #256 #(SZ.v width / 4) original (square_values (SZ.v width) before (SZ.v offset));
  let groups = width /^ 4sz;
  let mut i = 0sz;
  while (!i <^ groups)
    invariant exists* (count:sz{SZ.v count <= SZ.v groups}).
      i |-> count ** A.pts_to_mask partial
        (Local.written #256 #(SZ.v width / 4) original
          (square_values (SZ.v width) before (SZ.v offset)) (SZ.v count)) Init.all_cells
    invariant gpu ** A.pts_to hidden #fraction before
    decreases (SZ.v groups - SZ.v !i)
  {
    let count = !i;
    let value = square_group width hidden offset count;
    Local.written_step #256 #(SZ.v width / 4) original (square_values (SZ.v width) before (SZ.v offset)) (SZ.v count);
    A.mask_write partial count value;
    i := SZ.add count 1sz;
  };
  Local.written_ready #256 #(SZ.v width / 4) original (square_values (SZ.v width) before (SZ.v offset)) (SZ.v groups);
  let sum = Local.reduce #256 partial groups false
    #(Local.written #256 #(SZ.v width / 4) original
      (square_values (SZ.v width) before (SZ.v offset)) (SZ.v groups)) ();
  Prior.tree_ext (SZ.v groups)
    (Reduction.values (Local.prefix_values
      (Local.written #256 #(SZ.v width / 4) original
        (square_values (SZ.v width) before (SZ.v offset)) (SZ.v groups)) (SZ.v groups) ()))
    (Norm.grouped (SZ.v width) (input (SZ.v width) before (SZ.v offset)));
  let integer_width = IC.uint32_to_int64 (SZ.sizet_to_u32 width);
  let reciprocal = F.div (F.one #FP.t) (F.of_int #FP.t integer_width);
  assert pure (reciprocal == Norm.reciprocal (SZ.v width));
  F.rsqrt (F.add (F.mul sum reciprocal) Norm.epsilon)
}

inline_for_extraction noextract
let row_value (width:Norm.width) (before:Transfer.scratch) (weights:Transfer.packed)
  (source:nat{source + width <= 16000}) (weight_offset:nat{weight_offset + width <= 752393024})
  (i:P.index width) : GTot BF.t =
  let scale = Norm.source_scale width (input width before source) in
  let x = F.mul (Cast.cast_bf16_to_f32 (Seq.index before (source + i))) scale in
  Cast.cast_f32_to_bf16
    (F.mul x (F.add (F.one #FP.t) (Cast.cast_bf16_to_f32 (Seq.index weights (weight_offset + i)))))

inline_for_extraction noextract
let row_writes (width:Norm.width) (before:Transfer.scratch) (weights:Transfer.packed)
  (source destination:nat{source + width <= 16000 /\ destination + width <= 16000})
  (weight_offset:nat{weight_offset + width <= 752393024})
  (count:nat{count <= width}) : GTot Transfer.scratch =
  Seq.init_ghost 16000 (fun i ->
    if destination <= i && i < destination + count
    then row_value width before weights source weight_offset (i - destination)
    else Seq.index before i)

inline_for_extraction noextract
let row_zero (width:Norm.width) (before:Transfer.scratch) (weights:Transfer.packed)
  (source destination:nat{source + width <= 16000 /\ destination + width <= 16000})
  (weight_offset:nat{weight_offset + width <= 752393024})
  : Lemma (row_writes width before weights source destination weight_offset 0 == before) =
  assert (Seq.equal (row_writes width before weights source destination weight_offset 0) before);
  Seq.lemma_eq_elim (row_writes width before weights source destination weight_offset 0) before

inline_for_extraction noextract
let row_step (width:Norm.width) (before:Transfer.scratch) (weights:Transfer.packed)
  (source destination:nat{source + width <= 16000 /\ destination + width <= 16000 /\
    (destination <= source \/ source + width <= destination)})
  (weight_offset:nat{weight_offset + width <= 752393024}) (i:P.index width)
  : Lemma (
      Seq.index (row_writes width before weights source destination weight_offset i) (source + i) ==
        Seq.index before (source + i) /\
      Seq.upd (row_writes width before weights source destination weight_offset i) (destination + i)
        (row_value width before weights source weight_offset i) ==
      row_writes width before weights source destination weight_offset (i + 1)) =
  assert (Seq.equal
    (Seq.upd (row_writes width before weights source destination weight_offset i) (destination + i)
      (row_value width before weights source weight_offset i))
    (row_writes width before weights source destination weight_offset (i + 1)));
  Seq.lemma_eq_elim
    (Seq.upd (row_writes width before weights source destination weight_offset i) (destination + i)
      (row_value width before weights source weight_offset i))
    (row_writes width before weights source destination weight_offset (i + 1))

inline_for_extraction noextract
fn write_row
  (width:sz{SZ.v width = 128 \/ SZ.v width = 256 \/ SZ.v width = 1024})
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000)
  (source:sz{SZ.v source + SZ.v width <= 16000})
  (destination:sz{SZ.v destination + SZ.v width <= 16000 /\
    (SZ.v destination <= SZ.v source \/ SZ.v source + SZ.v width <= SZ.v destination)})
  (weight_offset:sz{SZ.v weight_offset + SZ.v width <= 752393024})
  (value_scale:FP.t)
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch) (#fraction:perm)
  preserves gpu ** A.pts_to weights #fraction vw
  requires A.pts_to hidden vh
  requires pure (value_scale == Norm.source_scale (SZ.v width) (input (SZ.v width) vh (SZ.v source)))
  ensures A.pts_to hidden (row_writes (SZ.v width) vh vw (SZ.v source) (SZ.v destination) (SZ.v weight_offset) (SZ.v width))
{
  row_zero (SZ.v width) vh vw (SZ.v source) (SZ.v destination) (SZ.v weight_offset);
  let mut i = 0sz;
  while (!i <^ width)
    invariant exists* (j:sz{SZ.v j <= SZ.v width}).
      i |-> j ** A.pts_to hidden (row_writes (SZ.v width) vh vw
        (SZ.v source) (SZ.v destination) (SZ.v weight_offset) (SZ.v j))
    invariant gpu ** A.pts_to weights #fraction vw
    decreases (SZ.v width - SZ.v !i)
  {
    let j = !i;
    row_step (SZ.v width) vh vw (SZ.v source) (SZ.v destination) (SZ.v weight_offset) (SZ.v j);
    let value = hidden.(SZ.add source j);
    let gain = weights.(SZ.add weight_offset j);
    let x = F.mul (Cast.cast_bf16_to_f32 value) value_scale;
    hidden.(SZ.add destination j) <- Cast.cast_f32_to_bf16
      (F.mul x (F.add (F.one #FP.t) (Cast.cast_bf16_to_f32 gain)));
    i := SZ.add j 1sz;
  }
}

inline_for_extraction noextract
fn apply_row
  (width:sz{SZ.v width = 128 \/ SZ.v width = 256 \/ SZ.v width = 1024})
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000) (partial:larray FP.t 256)
  (source:sz{SZ.v source + SZ.v width <= 16000})
  (destination:sz{SZ.v destination + SZ.v width <= 16000 /\
    (SZ.v destination <= SZ.v source \/ SZ.v source + SZ.v width <= SZ.v destination)})
  (weight_offset:sz{SZ.v weight_offset + SZ.v width <= 752393024})
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch)
  (#temporary:erased (lseq FP.t 256)) (#fraction:perm)
  preserves gpu ** A.pts_to weights #fraction vw
  requires A.pts_to hidden vh ** partial |-> temporary
  ensures exists* (after:lseq FP.t 256). partial |-> after **
    A.pts_to hidden (row_writes (SZ.v width) vh vw (SZ.v source) (SZ.v destination) (SZ.v weight_offset) (SZ.v width))
{
  let value_scale = scale width hidden source partial;
  write_row width weights hidden source destination weight_offset value_scale;
}

inline_for_extraction noextract
fn source_row
  (width:sz{SZ.v width = 128 \/ SZ.v width = 256 \/ SZ.v width = 1024})
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000)
  (source:sz{SZ.v source + SZ.v width <= 16000})
  (destination:sz{SZ.v destination + SZ.v width <= 16000 /\
    (SZ.v destination <= SZ.v source \/ SZ.v source + SZ.v width <= SZ.v destination)})
  (weight_offset:sz{SZ.v weight_offset + SZ.v width <= 752393024})
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch) (#fraction:perm)
  preserves gpu ** A.pts_to weights #fraction vw
  requires A.pts_to hidden vh
  ensures A.pts_to hidden (row_writes (SZ.v width) vh vw (SZ.v source) (SZ.v destination) (SZ.v weight_offset) (SZ.v width))
{
  let value_scale = source_scale width hidden source;
  write_row width weights hidden source destination weight_offset value_scale;
}
