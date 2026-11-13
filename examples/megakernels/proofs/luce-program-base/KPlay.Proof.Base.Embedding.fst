module KPlay.Proof.Base.Embedding

#lang-pulse

open Kuiper
open Pulse.Lib.Array
open FStar.Ghost

module S = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module I64 = FStar.Int64
module BF = Kuiper.BFloat16
module Shared = KPlay.Proof.Shared.Embedding
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module Rotary = KPlay.Proof.Base.Rotary
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
let embedding_mask (i:nat) : prop = i < 1024

inline_for_extraction noextract
let row (weights:Transfer.packed) (token:D.token 248320)
  : Shared.hidden_vector BF.t =
  S.slice weights (token * 1024) ((token + 1) * 1024)

inline_for_extraction noextract
let writes (before:Rotary.snapshot) (weights:Transfer.packed)
  (token:D.token 248320) (count:nat{count <= 1024}) : GTot Rotary.snapshot =
  S.init_ghost 16000 (fun i ->
    if i < count then Some (S.index weights (token * 1024 + i)) else S.index before i)

inline_for_extraction noextract
let writes_zero (before:Rotary.snapshot) (weights:Transfer.packed)
  (token:D.token 248320) : Lemma (writes before weights token 0 == before) =
  assert (S.equal (writes before weights token 0) before);
  S.lemma_eq_elim (writes before weights token 0) before

inline_for_extraction noextract
let writes_step (before:Rotary.snapshot) (weights:Transfer.packed)
  (token:D.token 248320) (i:nat{i < 1024})
  : Lemma (
      S.upd (writes before weights token i) i (Some (S.index weights (token * 1024 + i))) ==
      writes before weights token (i + 1)) =
  assert (S.equal
    (S.upd (writes before weights token i) i (Some (S.index weights (token * 1024 + i))))
    (writes before weights token (i + 1)));
  S.lemma_eq_elim
    (S.upd (writes before weights token i) i (Some (S.index weights (token * 1024 + i))))
    (writes before weights token (i + 1))

inline_for_extraction noextract
let copied_row (before:Rotary.snapshot) (weights:Transfer.packed)
  (token:D.token 248320)
  : Lemma (
      (forall (i:nat{i < 1024}).
        S.index (writes before weights token 1024) i == Some (S.index (row weights token) i)) /\
      (forall (i:nat{1024 <= i /\ i < 16000}).
        S.index (writes before weights token 1024) i == S.index before i)) = ()

inline_for_extraction noextract
fn copy_source_row
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000)
  (token:sz{SZ.v token < 248320})
  (#values:erased Transfer.packed) (#before:erased Rotary.snapshot) (#fraction:perm)
  preserves gpu ** A.pts_to weights #fraction values
  requires A.pts_to_mask hidden before embedding_mask
  ensures A.pts_to_mask hidden (writes before values (SZ.v token) 1024) embedding_mask
  ensures pure (
    forall (i:nat{i < 1024}).
      S.index (writes before values (SZ.v token) 1024) i ==
      Some (S.index (row values (SZ.v token)) i))
{
  writes_zero before values (SZ.v token);
  let mut i = 0sz;
  while (!i <^ 1024sz)
    invariant exists* (current:sz{SZ.v current <= 1024}).
      i |-> current ** A.pts_to_mask hidden (writes before values (SZ.v token) (SZ.v current)) embedding_mask
    invariant gpu ** A.pts_to weights #fraction values
    decreases (1024 - SZ.v !i)
  {
    let current = !i;
    let value = weights.(SZ.add (SZ.mul token 1024sz) current);
    writes_step before values (SZ.v token) (SZ.v current);
    A.mask_write hidden current value;
    i := SZ.add current 1sz;
  };
  copied_row before values (SZ.v token);
}

inline_for_extraction noextract
let prompt_valid (call:Model.invocation) (values:lseq I64.t call.prompt_length) : prop =
  forall (i:nat{i < call.prompt_length}).
    0 <= I64.v (S.index values i) /\ I64.v (S.index values i) < 248320

inline_for_extraction noextract
let source_token (call:Model.invocation) (position:nat{position < Model.calls call})
  (prompt:lseq I64.t call.prompt_length) (next:I64.t) : I64.t =
  if position < call.prompt_length then S.index prompt position else next

inline_for_extraction noextract
fn read_source_token
  (call:Model.invocation) (position:sz{SZ.v position < Model.calls call})
  (prompt_length:sz{SZ.v prompt_length == call.prompt_length})
  (prompt:larray I64.t call.prompt_length) (next:I64.t)
  (#values:lseq I64.t call.prompt_length) (#fraction:perm)
  preserves gpu ** A.pts_to prompt #fraction values
  requires pure (prompt_valid call values /\
    (SZ.v position < call.prompt_length \/ (0 <= I64.v next /\ I64.v next < 248320)))
  returns token:I64.t
  ensures pure (token == source_token call (SZ.v position) values next /\
    0 <= I64.v token /\ I64.v token < 248320)
{
  if (position <^ prompt_length) {
    prompt.(position)
  } else {
    next
  }
}
