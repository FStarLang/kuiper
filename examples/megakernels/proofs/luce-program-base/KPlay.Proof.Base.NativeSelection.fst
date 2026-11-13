module KPlay.Proof.Base.NativeSelection

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Array
open FStar.Ghost

module S = FStar.Seq
module A = Pulse.Lib.Array
module FP = Kuiper.Float32
module BF = Kuiper.BFloat16
module F = Kuiper.Floating
module Cast = Kuiper.Float.Casts.Base
module SZ = Kuiper.SizeT
module I64 = FStar.Int64
module IC = FStar.Int.Cast
module C = Kuiper.Chest
module P = KPlay.Proof.Shared.Projection
module D = KPlay.Proof.Shared.Selection
module Order = KPlay.Proof.Base.Order
module Transfer = KPlay.Proof.Base.Transfer
module Native = KPlay.Proof.Base.NativeProjection
module Scratch = KPlay.Proof.Base.Scratch
module Vocabulary = KPlay.Proof.Base.Vocabulary
module Inputs = KPlay.Proof.Base.Inputs

inline_for_extraction noextract
let replace_best (candidate current:FP.t)
  : result:bool{result == not (Order.priority_le candidate current)} =
  Order.native_order candidate current;
  F.eq_spec candidate candidate;
  F.eq_spec current current;
  F.lt current candidate ||
    (not (F.ieee_eq candidate candidate) && F.ieee_eq current current)

inline_for_extraction noextract
let replacement_agrees (before:Order.selection) (candidate:FP.t) (index:nat)
  : Lemma (
      Order.update before candidate index ==
        (if replace_best candidate before.Order.best
         then {Order.best=candidate; Order.token=index} else before)) =
  F.eq_spec candidate candidate;
  F.eq_spec before.Order.best before.Order.best

inline_for_extraction noextract
let bounded_selection = before:Order.selection{before.Order.token < 248320}

inline_for_extraction noextract
let owns (best:larray FP.t 1) (next:larray I64.t 1) (value:Order.selection) : slprop =
  A.pts_to best (S.create 1 value.Order.best) **
  (exists* (token:I64.t).
    A.pts_to next (S.create 1 token) ** pure (I64.v token == value.Order.token))

inline_for_extraction noextract
fn apply_candidate
  (best:larray FP.t 1) (next:larray I64.t 1) (candidate:FP.t)
  (index:sz{SZ.v index < 248320}) (#before:erased bounded_selection)
  preserves gpu
  requires owns best next before
  ensures owns best next (Order.update before candidate (SZ.v index))
{
  unfold owns best next before;
  with token. assert A.pts_to next (S.create 1 token);
  let current = best.(0sz);
  assert pure (current == (reveal before).Order.best);
  replacement_agrees before candidate (SZ.v index);
  if (replace_best candidate current) {
    let value = IC.uint32_to_int64 (SZ.sizet_to_u32 index);
    assert pure (I64.v value == SZ.v index);
    best.(0sz) <- candidate;
    next.(0sz) <- value;
    with best_values. assert A.pts_to best best_values;
    with token_values. assert A.pts_to next token_values;
    assert pure (S.equal best_values (S.create 1 candidate));
    S.lemma_eq_elim best_values (S.create 1 candidate);
    assert pure (S.equal token_values (S.create 1 value));
    S.lemma_eq_elim token_values (S.create 1 value);
    rewrite A.pts_to best best_values as A.pts_to best (S.create 1 candidate);
    rewrite A.pts_to next token_values as A.pts_to next (S.create 1 value);
    fold owns best next (Order.update before candidate (SZ.v index));
  } else {
    fold owns best next (Order.update before candidate (SZ.v index));
  }
}

inline_for_extraction noextract
let source_values (hidden:Transfer.scratch) : GTot (D.vec FP.t 16) =
  S.init_ghost 16 (fun i -> Cast.cast_bf16_to_f32 (S.index hidden (12500 + i)))

inline_for_extraction noextract
let source_value_at (hidden:Transfer.scratch) (i:P.index 16)
  : Lemma (
      S.index (source_values hidden) i ==
      Cast.cast_bf16_to_f32 (S.index hidden (12500 + i))) = ()

inline_for_extraction noextract
let scan_bound (values:D.vec FP.t 16) (row:Transfer.row_block) (before:bounded_selection)
  (count:nat{count <= 16})
  : Lemma ((Order.scan values row before count).Order.token < 248320) =
  Order.scan_dominates values row before count

inline_for_extraction noextract
let scan_step (values:D.vec FP.t 16) (row:Transfer.row_block) (before:bounded_selection)
  (count:nat{count < 16})
  : Lemma (
      Order.scan values row before (count + 1) ==
        Order.update (Order.scan values row before count) (S.index values count) (row + count)) =
  norm_spec [delta_once ["KPlay.Proof.Base.Order.scan"]] (Order.scan values row before (count + 1))

inline_for_extraction noextract
fn scan_source_block
  (hidden:larray BF.t 16000) (best:larray FP.t 1) (next:larray I64.t 1)
  (row:sz{SZ.v row % 16 == 0 /\ SZ.v row + 16 <= 248320})
  (#values:erased Transfer.scratch) (#before:erased bounded_selection) (#fraction:perm)
  preserves gpu ** A.pts_to hidden #fraction values
  requires owns best next before
  ensures owns best next (Order.scan (source_values values) (SZ.v row) before 16)
{
  let mut i = 0sz;
  while (!i <^ 16sz)
    invariant exists* (count:sz{SZ.v count <= 16}).
      i |-> count ** owns best next (Order.scan (source_values values) (SZ.v row) before (SZ.v count))
    invariant gpu ** A.pts_to hidden #fraction values
    decreases (16 - SZ.v !i)
  {
    let count = !i;
    let stored = hidden.(SZ.add 12500sz count);
    let value = Cast.cast_bf16_to_f32 stored;
    source_value_at values (SZ.v count);
    assert pure (value == S.index (source_values values) (SZ.v count));
    scan_bound (source_values values) (SZ.v row) before (SZ.v count);
    apply_candidate best next value (SZ.add row count);
    scan_step (source_values values) (SZ.v row) before (SZ.v count);
    rewrite owns best next
      (Order.update (Order.scan (source_values values) (SZ.v row) before (SZ.v count))
        value (SZ.v row + SZ.v count))
      as owns best next (Order.scan (source_values values) (SZ.v row) before (SZ.v count + 1));
    let next_count = SZ.add count 1sz;
    assert pure (SZ.v next_count == SZ.v count + 1);
    rewrite owns best next (Order.scan (source_values values) (SZ.v row) before (SZ.v count + 1))
      as owns best next (Order.scan (source_values values) (SZ.v row) before (SZ.v next_count));
    i := next_count;
  };
  let count = !i;
  assert pure (SZ.v count == 16);
  rewrite owns best next (Order.scan (source_values values) (SZ.v row) before (SZ.v count))
    as owns best next (Order.scan (source_values values) (SZ.v row) before 16);
}

inline_for_extraction noextract
let producer_consumer_binding
  (hidden:C.chest2 BF.t 16000 1) (produced:C.chest2 FP.t 16 8)
  : Lemma (
      source_values (Native.hidden_values
        (Scratch.replace_window hidden 12500 16 (Native.stored_column produced))) ==
      Transfer.consumed_scores produced) =
  Native.consumer_binding produced;
  let at_index (i:P.index 16)
    : Lemma (
        S.index (source_values (Native.hidden_values
          (Scratch.replace_window hidden 12500 16 (Native.stored_column produced)))) i ==
        S.index (Transfer.consumed_scores produced) i) =
    source_value_at (Native.hidden_values
      (Scratch.replace_window hidden 12500 16 (Native.stored_column produced))) i
  in
  FStar.Classical.forall_intro at_index;
  assert (S.equal
    (source_values (Native.hidden_values
      (Scratch.replace_window hidden 12500 16 (Native.stored_column produced))))
    (Transfer.consumed_scores produced));
  S.lemma_eq_elim
    (source_values (Native.hidden_values
      (Scratch.replace_window hidden 12500 16 (Native.stored_column produced))))
    (Transfer.consumed_scores produced)

inline_for_extraction noextract
fn initialize
  (best:larray FP.t 1) (next:larray I64.t 1)
  (#old_best:erased (lseq FP.t 1)) (#old_next:erased (lseq I64.t 1))
  preserves gpu
  requires A.pts_to best old_best ** A.pts_to next old_next
  ensures owns best next Vocabulary.seed
{
  best.(0sz) <- Vocabulary.negative_infinity;
  next.(0sz) <- 0L;
  with best_values. assert A.pts_to best best_values;
  with token_values. assert A.pts_to next token_values;
  assert pure (S.equal best_values (S.create 1 Vocabulary.negative_infinity));
  S.lemma_eq_elim best_values (S.create 1 Vocabulary.negative_infinity);
  assert pure (S.equal token_values (S.create 1 0L));
  S.lemma_eq_elim token_values (S.create 1 0L);
  rewrite A.pts_to best best_values as A.pts_to best (S.create 1 Vocabulary.negative_infinity);
  rewrite A.pts_to next token_values as A.pts_to next (S.create 1 0L);
  fold owns best next Vocabulary.seed;
}

inline_for_extraction noextract
fn selected_token
  (best:larray FP.t 1) (next:larray I64.t 1) (#before:erased bounded_selection)
  preserves gpu ** owns best next before
  returns token:Inputs.token_value
  ensures pure (I64.v token == (reveal before).Order.token)
{
  unfold owns best next before;
  with value. assert A.pts_to next (S.create 1 value);
  let token = next.(0sz);
  fold owns best next before;
  token
}
