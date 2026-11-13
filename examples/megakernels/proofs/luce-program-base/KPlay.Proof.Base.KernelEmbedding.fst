module KPlay.Proof.Base.KernelEmbedding

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Array
open FStar.Ghost

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module BF = Kuiper.BFloat16
module I64 = FStar.Int64
module SZ = Kuiper.SizeT
module IC = FStar.Int.Cast
module P = KPlay.Proof.Shared.Projection
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module Embedding = KPlay.Proof.Base.Embedding
module Rotary = KPlay.Proof.Base.Rotary
module Source = KPlay.Proof.RoPE.Source
module Init = KPlay.Proof.Base.Initialization
module Transfer = KPlay.Proof.Base.Transfer
module Stages = KPlay.Proof.Base.AttentionStages

inline_for_extraction noextract
let outside_embedding (i:nat) : prop = 1024 <= i

inline_for_extraction noextract
fn copy_row
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000) (token:sz{SZ.v token < 248320})
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch) (#fraction:perm)
  preserves gpu ** A.pts_to weights #fraction vw
  requires A.pts_to hidden vh
  ensures exists* (after:Transfer.scratch). A.pts_to hidden after **
    pure ((forall (i:P.index 1024). Seq.index after i == Seq.index (Embedding.row vw (SZ.v token)) i) /\
      (forall (i:P.index 16000). 1024 <= i ==> Seq.index after i == Seq.index vh i))
{
  A.to_mask hidden;
  with masked. assert A.pts_to_mask hidden masked (fun _ -> True);
  let original : erased Rotary.snapshot = hide #Rotary.snapshot masked;
  A.mask_mext hidden #1.0R #masked #(fun _ -> True) Init.all_cells;
  rewrite A.pts_to_mask hidden #1.0R masked Init.all_cells
    as A.pts_to_mask hidden #1.0R original Init.all_cells;
  A.split_mask hidden Embedding.embedding_mask;
  A.mask_mext hidden #1.0R #original #(A.mask_isect Init.all_cells Embedding.embedding_mask) Embedding.embedding_mask;
  A.mask_mext hidden #1.0R #original #(A.mask_diff Init.all_cells Embedding.embedding_mask) outside_embedding;
  Embedding.copy_source_row weights hidden token #vw #original #fraction;
  Embedding.copied_row original vw (SZ.v token);
  A.mask_vext hidden #1.0R #original (Embedding.writes original vw (SZ.v token) 1024) #outside_embedding;
  A.join_mask' hidden #1.0R #(Embedding.writes original vw (SZ.v token) 1024) #Embedding.embedding_mask #outside_embedding;
  A.mask_mext hidden #1.0R #(Embedding.writes original vw (SZ.v token) 1024)
    #(fun i -> Embedding.embedding_mask i \/ outside_embedding i) Init.all_cells;
  A.from_mask hidden;
}

inline_for_extraction noextract
let rotary_initialized (before:Rotary.snapshot) (position:Source.position)
  : Lemma
      (requires forall (i:P.index 16000). Some? (Seq.index before i))
      (ensures forall (i:P.index 16000). Some? (Seq.index (Rotary.source_writes before position 32) i)) =
  let at_index (i:P.index 16000)
    : Lemma (Some? (Seq.index (Rotary.source_writes before position 32) i)) =
    Rotary.source_write_cell before position 32 i
  in
  FStar.Classical.forall_intro at_index

inline_for_extraction noextract
fn position_embedding
  (hidden:larray BF.t 16000) (position:Source.position) (#before:erased Transfer.scratch)
  preserves gpu
  requires A.pts_to hidden before
  ensures exists* (after:Transfer.scratch). A.pts_to hidden after **
    pure (Stages.rope (I64.v position) after /\
      (forall (i:P.index 16000). ~(Rotary.rotary_mask i) ==> Seq.index after i == Seq.index before i))
{
  A.to_mask hidden;
  with masked. assert A.pts_to_mask hidden masked (fun _ -> True);
  let original : erased Rotary.snapshot = hide #Rotary.snapshot masked;
  A.mask_mext hidden #1.0R #masked #(fun _ -> True) Init.all_cells;
  rewrite A.pts_to_mask hidden #1.0R masked Init.all_cells
    as A.pts_to_mask hidden #1.0R original Init.all_cells;
  Init.isolate_rotary_window hidden #original #1.0R;
  Rotary.position_embedding hidden position #original;
  Rotary.source_connection original position;
  Rotary.source_initialized original position;
  rotary_initialized original position;
  A.mask_vext hidden #1.0R #original (Rotary.source_writes original position 32) #Init.outside_rotary;
  A.join_mask' hidden #1.0R #(Rotary.source_writes original position 32) #Rotary.rotary_mask #Init.outside_rotary;
  A.mask_mext hidden #1.0R #(Rotary.source_writes original position 32)
    #(fun i -> Rotary.rotary_mask i \/ Init.outside_rotary i) Init.all_cells;
  A.from_mask hidden;
  with after. assert A.pts_to hidden after;
  let cells : erased (lseq BF.t 64) = hide #(lseq BF.t 64) (Rotary.produced_cells original position);
  assert pure (forall (i:P.index 64). Seq.index after (15600 + i) == Seq.index cells i);
  ()
}

inline_for_extraction noextract
fn read_token
  (call:Model.invocation) (position:sz{SZ.v position < Model.calls call})
  (prompt_length:sz{SZ.v prompt_length == call.Model.prompt_length})
  (prompt:larray I64.t call.Model.prompt_length) (next:larray I64.t 1)
  (#sp:erased (lseq I64.t call.Model.prompt_length)) (#sn:erased (lseq I64.t 1)) (#fraction:perm)
  preserves gpu ** A.pts_to prompt #fraction sp ** next |-> sn
  requires pure (Embedding.prompt_valid call sp /\
    (SZ.v position < call.Model.prompt_length \/ (0 <= I64.v (Seq.index sn 0) /\ I64.v (Seq.index sn 0) < 248320)))
  returns word:I64.t
  ensures pure (word == Embedding.source_token call (SZ.v position) sp (Seq.index sn 0) /\ 0 <= I64.v word /\ I64.v word < 248320)
{
  if (position <^ prompt_length) { prompt.(position) } else { next.(0sz) }
}

inline_for_extraction noextract
fn apply
  (call:Model.invocation) (position:sz{SZ.v position < Model.calls call})
  (prompt_length:sz{SZ.v prompt_length == call.Model.prompt_length})
  (weights:larray BF.t 752393024) (prompt:larray I64.t call.Model.prompt_length)
  (hidden:larray BF.t 16000) (next:larray I64.t 1)
  (#vw:erased Transfer.packed) (#sp:erased (lseq I64.t call.Model.prompt_length))
  (#vh:erased Transfer.scratch) (#sn:erased (lseq I64.t 1)) (#wf #pf:perm)
  preserves gpu ** A.pts_to weights #wf vw ** A.pts_to prompt #pf sp ** next |-> sn
  requires A.pts_to hidden vh
  requires pure (Embedding.prompt_valid call sp /\
    (SZ.v position < call.Model.prompt_length \/ (0 <= I64.v (Seq.index sn 0) /\ I64.v (Seq.index sn 0) < 248320)))
  returns word:I64.t
  ensures exists* (after:Transfer.scratch). A.pts_to hidden after **
    pure (word == Embedding.source_token call (SZ.v position) sp (Seq.index sn 0) /\ 0 <= I64.v word /\ I64.v word < 248320 /\
      Stages.rope (SZ.v position) after /\
      (forall (i:P.index 1024). Seq.index after i == Seq.index (Embedding.row vw (I64.v word)) i))
{
  let word = read_token call position prompt_length prompt next;
  let token = SZ.uint32_to_sizet (IC.int64_to_uint32 word);
  copy_row weights hidden token;
  let p = IC.uint32_to_int64 (SZ.sizet_to_u32 position);
  position_embedding hidden p;
  word
}
