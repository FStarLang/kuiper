module KPlay.Proof.Base.RecurrentHead

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
module State = KPlay.Proof.Base.RecurrentStateMath
module Core = KPlay.Proof.Base.RecurrentCore
module Gating = KPlay.Proof.Base.RecurrentGating
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
let gating_inputs (scope:prop) (weights:Transfer.packed) (before current:Transfer.scratch) (state:lseq FP.t 4718592)
  (base:nat{base + 19458208 <= 752393024}) (offset:nat{offset + 16384 <= 4718592})
  (head:P.index 16) (x:H.inputs)
  : Lemma
      (requires State.hidden_frame before current head 128 /\
        (scope ==> H.represented weights before state base offset head x /\ State.core_prefix current head x 128))
      (ensures scope ==> Gating.inputs current weights base head x) =
  let at_index (i:P.index 128)
    : Lemma (Seq.index current (8192 + head * 128 + i) == Seq.index before (8192 + head * 128 + i)) = ()
  in
  FStar.Classical.forall_intro at_index

inline_for_extraction noextract
fn apply
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000) (state:larray FP.t 4718592)
  (base:sz{SZ.v base + 19458208 <= 752393024})
  (offset:sz{SZ.v offset + 16384 <= 4718592}) (head:sz{SZ.v head < 16})
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch) (#ss:erased (lseq FP.t 4718592))
  (#fraction:perm) (#x:erased H.inputs) (#scope:prop)
  preserves gpu ** A.pts_to weights #fraction vw
  requires A.pts_to hidden vh ** state |-> ss
  requires pure (scope ==> H.represented vw vh ss (SZ.v base) (SZ.v offset) (SZ.v head) x)
  ensures exists* (result:Transfer.scratch) (updated:lseq FP.t 4718592).
    A.pts_to hidden result ** state |-> updated **
    pure (State.state_frame ss updated (SZ.v offset) /\
      State.hidden_frame vh result (SZ.v head) 128 /\
      (scope ==> State.columns updated (SZ.v offset) x 128 /\
        (forall (i:P.index 128).
          Approx.v_approximates (Seq.index result (10400 + SZ.v head * 128 + i)) (H.gated x i))))
{
  let mut query : A.array FP.t = [| 128sz |];
  let mut key : A.array FP.t = [| 128sz |];
  Core.apply weights hidden state query key base offset head
    #vw #vh #ss #_ #_ #fraction #x #scope;
  with core_hidden updated. assert A.pts_to hidden core_hidden ** state |-> updated;
  gating_inputs scope vw vh core_hidden ss (SZ.v base) (SZ.v offset) (SZ.v head) x;
  Gating.source_apply weights hidden base head;
  Gating.complete scope core_hidden vw (SZ.v base) (SZ.v head) x;
  A.to_mask query;
  A.to_mask key;
  ()
}
