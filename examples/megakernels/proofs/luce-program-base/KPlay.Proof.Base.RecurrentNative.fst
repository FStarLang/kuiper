module KPlay.Proof.Base.RecurrentNative

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
module Neg = Kuiper.Float32.NativeNeg
module Cast = Kuiper.Float.Casts.Base
module Softplus = Luce.Softplus
module P = KPlay.Proof.Shared.Projection
module Scalar = KPlay.Proof.Base.RecurrentScalars
module L2 = KPlay.Proof.Base.L2Native
module Transfer = KPlay.Proof.Base.Transfer
module Local = KPlay.Proof.Base.LocalArray
module Init = KPlay.Proof.Base.Initialization

inline_for_extraction noextract
let prefix (before:lseq FP.t 128) (next:P.vector FP.t 128) (count:nat{count <= 128})
  : GTot (lseq FP.t 128) =
  Seq.init_ghost 128 (fun i -> if i < count then next i else Seq.index before i)

inline_for_extraction noextract
let prefix_zero (before:lseq FP.t 128) (next:P.vector FP.t 128)
  : Lemma (prefix before next 0 == before) =
  assert (Seq.equal (prefix before next 0) before);
  Seq.lemma_eq_elim (prefix before next 0) before

inline_for_extraction noextract
let prefix_step (before:lseq FP.t 128) (next:P.vector FP.t 128) (i:P.index 128)
  : Lemma (Seq.upd (prefix before next i) i (next i) == prefix before next (i + 1)) =
  assert (Seq.equal (Seq.upd (prefix before next i) i (next i)) (prefix before next (i + 1)));
  Seq.lemma_eq_elim (Seq.upd (prefix before next i) i (next i)) (prefix before next (i + 1))

inline_for_extraction noextract
let query_values (hidden:Transfer.scratch) (head:P.index 16) : GTot (lseq FP.t 128) =
  Seq.init_ghost 128 (Scalar.query hidden head)

inline_for_extraction noextract
let key_values (hidden:Transfer.scratch) (head:P.index 16) : GTot (lseq FP.t 128) =
  Seq.init_ghost 128 (Scalar.key hidden head)

inline_for_extraction noextract
fn prepare_qk
  (hidden:larray BF.t 16000) (query key:larray FP.t 128) (partial:larray FP.t 16)
  (head:sz{SZ.v head < 16})
  (#vh:erased Transfer.scratch) (#vq #vk:erased (lseq FP.t 128)) (#vp:erased (lseq FP.t 16))
  (#fraction:perm)
  preserves gpu ** A.pts_to hidden #fraction vh
  requires query |-> vq ** key |-> vk ** partial |-> vp
  ensures exists* (temporary:lseq FP.t 16). partial |-> temporary **
    query |-> prefix vq (Scalar.query vh (SZ.v head)) 128 **
    key |-> prefix vk (Scalar.key vh (SZ.v head)) 128
{
  let qoffset = SZ.add 2048sz (SZ.mul head 128sz);
  let koffset = SZ.add 4096sz (SZ.mul head 128sz);
  let qscale = L2.scale hidden qoffset partial;
  let kscale = L2.scale hidden koffset partial;
  prefix_zero vq (Scalar.query vh (SZ.v head));
  prefix_zero vk (Scalar.key vh (SZ.v head));
  let mut i = 0sz;
  while (!i <^ 128sz)
    invariant exists* (j:sz{SZ.v j <= 128}).
      i |-> j ** query |-> prefix vq (Scalar.query vh (SZ.v head)) (SZ.v j) **
      key |-> prefix vk (Scalar.key vh (SZ.v head)) (SZ.v j)
    invariant gpu ** A.pts_to hidden #fraction vh
    decreases (128 - SZ.v !i)
  {
    let j = !i;
    let qraw = hidden.(SZ.add qoffset j);
    let q = Cast.cast_bf16_to_f32 (Cast.cast_f32_to_bf16 (F.mul (Cast.cast_bf16_to_f32 qraw) qscale));
    prefix_step vq (Scalar.query vh (SZ.v head)) (SZ.v j);
    query.(j) <- F.mul q Scalar.query_coefficient;
    let kraw = hidden.(SZ.add koffset j);
    let k = Cast.cast_bf16_to_f32 (Cast.cast_f32_to_bf16 (F.mul (Cast.cast_bf16_to_f32 kraw) kscale));
    prefix_step vk (Scalar.key vh (SZ.v head)) (SZ.v j);
    key.(j) <- k;
    i := SZ.add j 1sz;
  }
}

inline_for_extraction noextract
fn source_prepare_qk
  (hidden:larray BF.t 16000) (query key:larray FP.t 128) (head:sz{SZ.v head < 16})
  (#vh:erased Transfer.scratch) (#vq #vk:erased (Local.snapshot 128)) (#fraction:perm)
  preserves gpu ** A.pts_to hidden #fraction vh
  requires A.pts_to_mask query vq Init.all_cells ** A.pts_to_mask key vk Init.all_cells
  ensures query |-> query_values vh (SZ.v head) ** key |-> key_values vh (SZ.v head)
{
  let qoffset = SZ.add 2048sz (SZ.mul head 128sz);
  let koffset = SZ.add 4096sz (SZ.mul head 128sz);
  let qscale = L2.source_scale hidden qoffset;
  let kscale = L2.source_scale hidden koffset;
  Local.written_zero #128 #128 vq (Scalar.query vh (SZ.v head));
  Local.written_zero #128 #128 vk (Scalar.key vh (SZ.v head));
  let mut i = 0sz;
  while (!i <^ 128sz)
    invariant exists* (j:sz{SZ.v j <= 128}).
      i |-> j **
      A.pts_to_mask query (Local.written #128 #128 vq (Scalar.query vh (SZ.v head)) (SZ.v j)) Init.all_cells **
      A.pts_to_mask key (Local.written #128 #128 vk (Scalar.key vh (SZ.v head)) (SZ.v j)) Init.all_cells
    invariant gpu ** A.pts_to hidden #fraction vh
    decreases (128 - SZ.v !i)
  {
    let j = !i;
    let qraw = hidden.(SZ.add qoffset j);
    let q = Cast.cast_bf16_to_f32 (Cast.cast_f32_to_bf16 (F.mul (Cast.cast_bf16_to_f32 qraw) qscale));
    Local.written_step #128 #128 vq (Scalar.query vh (SZ.v head)) (SZ.v j);
    A.mask_write query j (F.mul q Scalar.query_coefficient);
    let kraw = hidden.(SZ.add koffset j);
    let k = Cast.cast_bf16_to_f32 (Cast.cast_f32_to_bf16 (F.mul (Cast.cast_bf16_to_f32 kraw) kscale));
    Local.written_step #128 #128 vk (Scalar.key vh (SZ.v head)) (SZ.v j);
    A.mask_write key j k;
    i := SZ.add j 1sz;
  };
  Local.written_ready #128 #128 vq (Scalar.query vh (SZ.v head)) 128;
  Local.written_ready #128 #128 vk (Scalar.key vh (SZ.v head)) 128;
  A.from_mask query;
  A.from_mask key;
  with actual_query actual_key. assert query |-> actual_query ** key |-> actual_key;
  assert pure (Seq.equal actual_query (query_values vh (SZ.v head)));
  assert pure (Seq.equal actual_key (key_values vh (SZ.v head)));
  Seq.lemma_eq_elim actual_query (query_values vh (SZ.v head));
  Seq.lemma_eq_elim actual_key (key_values vh (SZ.v head));
}

inline_for_extraction noextract
fn controls
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000)
  (base:sz{SZ.v base + 19458080 <= 752393024}) (head:sz{SZ.v head < 16})
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch) (#weight_fraction #hidden_fraction:perm)
  preserves gpu ** A.pts_to weights #weight_fraction vw ** A.pts_to hidden #hidden_fraction vh
  returns result:(FP.t & FP.t)
  ensures pure (
    result._1 == Scalar.beta (Seq.index vh (10240 + SZ.v head)) /\
    result._2 == Scalar.decay (Seq.index vh (10256 + SZ.v head))
      (Seq.index vw (SZ.v base + 19458064 + SZ.v head))
      (Seq.index vw (SZ.v base + 19458048 + SZ.v head)))
{
  let b = hidden.(SZ.add 10240sz head);
  let beta = Scalar.beta b;
  let a = hidden.(SZ.add 10256sz head);
  let bias = weights.(SZ.add (SZ.add base (SZ.uint_to_t 19458064)) head);
  let combined = F.add (Cast.cast_bf16_to_f32 a) (Cast.cast_bf16_to_f32 bias);
  let softplus = Softplus.softplus20 combined;
  let log = weights.(SZ.add (SZ.add base (SZ.uint_to_t 19458048)) head);
  let decay = F.fexp (F.mul (Neg.neg (F.fexp (Cast.cast_bf16_to_f32 log))) softplus);
  (beta, decay)
}

inline_for_extraction noextract
let decayed (before:lseq FP.t 4718592) (offset:nat{offset + 16384 <= 4718592})
  (gain:FP.t) (count:nat{count <= 16384}) : GTot (lseq FP.t 4718592) =
  Seq.init_ghost 4718592 (fun i ->
    if offset <= i && i < offset + count then F.mul (Seq.index before i) gain else Seq.index before i)

inline_for_extraction noextract
let decayed_zero (before:lseq FP.t 4718592) (offset:nat{offset + 16384 <= 4718592}) (gain:FP.t)
  : Lemma (decayed before offset gain 0 == before) =
  assert (Seq.equal (decayed before offset gain 0) before);
  Seq.lemma_eq_elim (decayed before offset gain 0) before

inline_for_extraction noextract
let decayed_step (before:lseq FP.t 4718592) (offset:nat{offset + 16384 <= 4718592})
  (gain:FP.t) (i:P.index 16384)
  : Lemma (
      Seq.index (decayed before offset gain i) (offset + i) == Seq.index before (offset + i) /\
      Seq.upd (decayed before offset gain i) (offset + i) (F.mul (Seq.index before (offset + i)) gain) ==
      decayed before offset gain (i + 1)) =
  assert (Seq.equal
    (Seq.upd (decayed before offset gain i) (offset + i) (F.mul (Seq.index before (offset + i)) gain))
    (decayed before offset gain (i + 1)));
  Seq.lemma_eq_elim
    (Seq.upd (decayed before offset gain i) (offset + i) (F.mul (Seq.index before (offset + i)) gain))
    (decayed before offset gain (i + 1))

inline_for_extraction noextract
fn decay_state
  (state:larray FP.t 4718592) (offset:sz{SZ.v offset + 16384 <= 4718592}) (gain:FP.t)
  (#before:erased (lseq FP.t 4718592))
  preserves gpu
  requires state |-> before
  ensures state |-> decayed before (SZ.v offset) gain 16384
{
  decayed_zero before (SZ.v offset) gain;
  let mut i = 0sz;
  while (!i <^ 16384sz)
    invariant exists* (j:sz{SZ.v j <= 16384}).
      i |-> j ** state |-> decayed before (SZ.v offset) gain (SZ.v j)
    invariant gpu
    decreases (16384 - SZ.v !i)
  {
    let j = !i;
    decayed_step before (SZ.v offset) gain (SZ.v j);
    let old = state.(SZ.add offset j);
    state.(SZ.add offset j) <- F.mul old gain;
    i := SZ.add j 1sz;
  }
}

inline_for_extraction noextract
let updated_column (before:lseq FP.t 4718592) (key:lseq FP.t 128)
  (offset:nat{offset + 16384 <= 4718592}) (column:P.index 128) (delta:FP.t)
  (count:nat{count <= 128}) : GTot (lseq FP.t 4718592) =
  Seq.init_ghost 4718592 (fun i ->
    if offset <= i && i < offset + 16384 && (i - offset) % 128 = column && (i - offset) / 128 < count
    then F.add (Seq.index before i) (F.mul (Seq.index key ((i - offset) / 128)) delta)
    else Seq.index before i)

inline_for_extraction noextract
let updated_zero (before:lseq FP.t 4718592) (key:lseq FP.t 128)
  (offset:nat{offset + 16384 <= 4718592}) (column:P.index 128) (delta:FP.t)
  : Lemma (updated_column before key offset column delta 0 == before) =
  assert (Seq.equal (updated_column before key offset column delta 0) before);
  Seq.lemma_eq_elim (updated_column before key offset column delta 0) before

inline_for_extraction noextract
let updated_step (before:lseq FP.t 4718592) (key:lseq FP.t 128)
  (offset:nat{offset + 16384 <= 4718592}) (column:P.index 128) (delta:FP.t) (i:P.index 128)
  : Lemma (
      Seq.index (updated_column before key offset column delta i) (offset + i * 128 + column) ==
        Seq.index before (offset + i * 128 + column) /\
      Seq.upd (updated_column before key offset column delta i) (offset + i * 128 + column)
        (F.add (Seq.index before (offset + i * 128 + column)) (F.mul (Seq.index key i) delta)) ==
      updated_column before key offset column delta (i + 1)) =
  assert (Seq.equal
    (Seq.upd (updated_column before key offset column delta i) (offset + i * 128 + column)
      (F.add (Seq.index before (offset + i * 128 + column)) (F.mul (Seq.index key i) delta)))
    (updated_column before key offset column delta (i + 1)));
  Seq.lemma_eq_elim
    (Seq.upd (updated_column before key offset column delta i) (offset + i * 128 + column)
      (F.add (Seq.index before (offset + i * 128 + column)) (F.mul (Seq.index key i) delta)))
    (updated_column before key offset column delta (i + 1))

inline_for_extraction noextract
fn update_column
  (state:larray FP.t 4718592) (key:larray FP.t 128)
  (offset:sz{SZ.v offset + 16384 <= 4718592}) (column:sz{SZ.v column < 128}) (delta:FP.t)
  (#before:erased (lseq FP.t 4718592)) (#keys:erased (lseq FP.t 128)) (#fraction:perm)
  preserves gpu ** key |-> Frac fraction keys
  requires state |-> before
  ensures state |-> updated_column before keys (SZ.v offset) (SZ.v column) delta 128
{
  updated_zero before keys (SZ.v offset) (SZ.v column) delta;
  let mut i = 0sz;
  while (!i <^ 128sz)
    invariant exists* (j:sz{SZ.v j <= 128}).
      i |-> j ** state |-> updated_column before keys (SZ.v offset) (SZ.v column) delta (SZ.v j)
    invariant gpu ** key |-> Frac fraction keys
    decreases (128 - SZ.v !i)
  {
    let j = !i;
    updated_step before keys (SZ.v offset) (SZ.v column) delta (SZ.v j);
    let address = SZ.add (SZ.add offset (SZ.mul j 128sz)) column;
    let old = state.(address);
    let coefficient = key.(j);
    state.(address) <- F.add old (F.mul coefficient delta);
    i := SZ.add j 1sz;
  }
}
