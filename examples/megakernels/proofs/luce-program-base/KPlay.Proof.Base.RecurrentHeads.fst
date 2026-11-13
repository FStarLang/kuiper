module KPlay.Proof.Base.RecurrentHeads

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
module Approx = Kuiper.Approximates.Base
module P = KPlay.Proof.Shared.Projection
module H = KPlay.Proof.Base.RecurrentHeadModel
module Head = KPlay.Proof.Base.RecurrentHead
module State = KPlay.Proof.Base.RecurrentStateMath
module Dot = KPlay.Proof.Base.RecurrentDotMath
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
let models = P.index 16 -> GTot H.inputs

inline_for_extraction noextract
let at (model:erased models) (head:P.index 16) : GTot H.inputs =
  (reveal #models model) head

inline_for_extraction noextract
let offset (slot:nat{slot < 18}) (head:P.index 16) : (i:nat{i + 16384 <= 4718592}) =
  slot * 262144 + head * 16384

inline_for_extraction noextract
let state_frame (before current:lseq FP.t 4718592) (slot:nat{slot < 18}) (count:nat{count <= 16}) : prop =
  forall (i:P.index 4718592).
    ~(slot * 262144 <= i /\ i < slot * 262144 + count * 16384) ==>
      Seq.index current i == Seq.index before i

inline_for_extraction noextract
let hidden_frame (before current:Transfer.scratch) (count:nat{count <= 16}) : prop =
  forall (i:P.index 16000).
    ~(10400 <= i /\ i < 10400 + count * 128) ==> Seq.index current i == Seq.index before i

inline_for_extraction noextract
let completed (state:lseq FP.t 4718592) (hidden:Transfer.scratch)
  (slot:nat{slot < 18}) (model:models) (count:nat{count <= 16}) : prop =
  forall (head:nat{head < count}).
    State.columns state (offset slot head) (model head) 128 /\
    (forall (i:P.index 128).
      Approx.v_approximates (Seq.index hidden (10400 + head * 128 + i)) (H.gated (model head) i))

inline_for_extraction noextract
let source_inputs (scope:prop) (weights:Transfer.packed) (initial_hidden current_hidden:Transfer.scratch)
  (initial_state current_state:lseq FP.t 4718592)
  (base:nat{base + 19458208 <= 752393024}) (slot:nat{slot < 18}) (model:models) (head:P.index 16)
  : Lemma
      (requires
        (scope ==> forall (h:P.index 16). H.represented weights initial_hidden initial_state base (offset slot h) h (model h)) /\
        state_frame initial_state current_state slot head /\ hidden_frame initial_hidden current_hidden head)
      (ensures scope ==> H.represented weights current_hidden current_state base (offset slot head) head (model head)) =
  let at_cell (i j:P.index 128)
    : Lemma (
        Seq.index current_state (offset slot head + i * 128 + j) ==
        Seq.index initial_state (offset slot head + i * 128 + j)) = ()
  in
  FStar.Classical.forall_intro_2 at_cell

inline_for_extraction noextract
let completed_frame (before current:lseq FP.t 4718592) (old_hidden new_hidden:Transfer.scratch)
  (slot:nat{slot < 18}) (model:models) (head:P.index 16)
  : Lemma
      (requires completed before old_hidden slot model head /\
        State.state_frame before current (offset slot head) /\ State.hidden_frame old_hidden new_hidden head 128)
      (ensures completed current new_hidden slot model head) =
  let at_head (h:nat{h < head})
    : Lemma (
        State.columns current (offset slot h) (model h) 128 /\
        (forall (i:P.index 128).
          Approx.v_approximates (Seq.index new_hidden (10400 + h * 128 + i)) (H.gated (model h) i))) =
    let at_cell (i j:P.index 128)
      : Lemma (Seq.index current (offset slot h + i * 128 + j) ==
          Seq.index before (offset slot h + i * 128 + j)) = ()
    in
    FStar.Classical.forall_intro_2 at_cell;
    let hidden_cell (i:P.index 128)
      : Lemma (Seq.index new_hidden (10400 + h * 128 + i) == Seq.index old_hidden (10400 + h * 128 + i)) = ()
    in
    FStar.Classical.forall_intro hidden_cell
  in
  FStar.Classical.forall_intro at_head

inline_for_extraction noextract
let step (scope:prop) (initial_state before current:lseq FP.t 4718592)
  (initial_hidden old_hidden new_hidden:Transfer.scratch)
  (slot:nat{slot < 18}) (model:models) (head:P.index 16)
  : Lemma
      (requires
        state_frame initial_state before slot head /\ hidden_frame initial_hidden old_hidden head /\
        State.state_frame before current (offset slot head) /\
        State.hidden_frame old_hidden new_hidden head 128 /\
        (scope ==> completed before old_hidden slot model head /\
          State.columns current (offset slot head) (model head) 128 /\
          (forall (i:P.index 128).
            Approx.v_approximates (Seq.index new_hidden (10400 + head * 128 + i)) (H.gated (model head) i))))
      (ensures state_frame initial_state current slot (head + 1) /\ hidden_frame initial_hidden new_hidden (head + 1) /\
        (scope ==> completed current new_hidden slot model (head + 1))) =
  let under (_:scope)
    : Lemma (completed current new_hidden slot model (head + 1)) =
    completed_frame before current old_hidden new_hidden slot model head;
    let at_head (h:nat{h < head + 1})
      : Lemma (
          State.columns current (offset slot h) (model h) 128 /\
          (forall (i:P.index 128).
            Approx.v_approximates (Seq.index new_hidden (10400 + h * 128 + i)) (H.gated (model h) i))) =
      if h < head then
        assert (State.columns current (offset slot h) (model h) 128)
      else assert (h == head)
    in
    FStar.Classical.forall_intro at_head
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
fn apply
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000) (state:larray FP.t 4718592)
  (base:sz{SZ.v base + 19458208 <= 752393024}) (slot:sz{SZ.v slot < 18})
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch) (#ss:erased (lseq FP.t 4718592))
  (#fraction:perm) (#model:erased models) (#scope:prop)
  preserves gpu ** A.pts_to weights #fraction vw
  requires A.pts_to hidden vh ** state |-> ss
  requires pure (scope ==> forall (head:P.index 16).
    H.represented vw vh ss (SZ.v base) (offset (SZ.v slot) head) head (at model head))
  ensures exists* (result:Transfer.scratch) (updated:lseq FP.t 4718592).
    A.pts_to hidden result ** state |-> updated **
    pure (state_frame ss updated (SZ.v slot) 16 /\ hidden_frame vh result 16 /\
      (scope ==> completed updated result (SZ.v slot) model 16))
{
  let mut head = 0sz;
  while (!head <^ 16sz)
    invariant exists* (h:sz{SZ.v h <= 16}) (current_hidden:Transfer.scratch) (current_state:lseq FP.t 4718592).
      head |-> h ** A.pts_to hidden current_hidden ** state |-> current_state **
      pure (state_frame ss current_state (SZ.v slot) (SZ.v h) /\ hidden_frame vh current_hidden (SZ.v h) /\
        (scope ==> completed current_state current_hidden (SZ.v slot) model (SZ.v h)))
    invariant gpu ** A.pts_to weights #fraction vw
    decreases (16 - SZ.v !head)
  {
    let h = !head;
    with old_hidden old_state. assert A.pts_to hidden old_hidden ** state |-> old_state;
    source_inputs scope vw vh old_hidden ss old_state (SZ.v base) (SZ.v slot) model (SZ.v h);
    let state_offset = SZ.add (SZ.mul slot (SZ.uint_to_t 262144)) (SZ.mul h 16384sz);
    Head.apply weights hidden state base state_offset h
      #vw #old_hidden #old_state #fraction #(at model (SZ.v h)) #scope;
    with next_hidden next_state. assert A.pts_to hidden next_hidden ** state |-> next_state;
    step scope ss old_state next_state vh old_hidden next_hidden (SZ.v slot) model (SZ.v h);
    head := SZ.add h 1sz;
  }
}
