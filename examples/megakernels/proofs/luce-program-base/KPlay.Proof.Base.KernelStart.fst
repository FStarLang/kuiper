module KPlay.Proof.Base.KernelStart

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor
open Pulse.Lib.Array
open FStar.Ghost

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module GR = Pulse.Lib.GhostReference
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module I64 = FStar.Int64
module P = KPlay.Proof.Shared.Projection
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module K = KPlay.Proof.Base.KernelState
module Trace = KPlay.Proof.Base.Trace
module Entry = KPlay.Proof.Base.Entry
module Owned = KPlay.Proof.Base.KernelOwnership
module Book = KPlay.Proof.Base.WarpBook
module Transfer = KPlay.Proof.Base.Transfer
module Native = KPlay.Proof.Base.NativeProjection
module Hidden = KPlay.Proof.Base.WarpHidden
module Cache = KPlay.Proof.Base.CacheNative
module Global = KPlay.Proof.Base.GlobalCache
module Init = KPlay.Proof.Base.Initialization

inline_for_extraction noextract
let initial (call:Model.invocation)
  (tokens:lseq (option I64.t) call.Model.decode_steps)
  (history:lseq (option BF.t) (call.Model.decode_steps * 1024)) (best:FP.t)
  : K.metadata call = {
  K.completed = 0; K.history = Trace.empty; K.phase = K.Between;
  K.memory = {
    K.state = Init.state_values; K.convolution = Init.convolution_values;
    K.tokens = tokens; K.hidden = history; K.next = 0L; K.best = best};
}

inline_for_extraction noextract
let initial_valid (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length)
  (tokens:lseq (option I64.t) call.Model.decode_steps)
  (history:lseq (option BF.t) (call.Model.decode_steps * 1024)) (best:FP.t)
  (h:Hidden.snapshot) (kv:Cache.snapshot)
  : Lemma
      (requires Global.available_contents kv)
      (ensures K.ready call weights prompt 0 (initial call tokens history best) h kv) =
  Global.initial K.configuration kv;
  norm_spec [delta_once ["KPlay.Proof.Base.KernelState.ready"]]
    (K.ready call weights prompt 0 (initial call tokens history best) h kv)

inline_for_extraction noextract
let pending (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (hidden:array2 BF.t Native.hidden_layout) (best:larray FP.t 1) (next:larray I64.t 1) : slprop =
  exists* (state:lseq (option FP.t) 4718592) (conv:lseq (option BF.t) 442368)
    (kv:Cache.snapshot) (tokens:lseq (option I64.t) call.Model.decode_steps)
    (history:lseq (option BF.t) (call.Model.decode_steps * 1024))
    (h:Hidden.snapshot) (old_best:lseq FP.t 1) (old_next:lseq I64.t 1).
    A.pts_to_mask args.Entry.state state Init.all_cells **
    A.pts_to_mask args.Entry.convolution conv Init.all_cells **
    A.pts_to_mask args.Entry.cache kv Init.all_cells **
    A.pts_to_mask args.Entry.tokens tokens Init.all_cells **
    A.pts_to_mask args.Entry.hidden history Init.all_cells **
    hidden |-> h ** best |-> old_best ** next |-> old_next **
    Owned.parked refs.Owned.global ** Owned.parked refs.Owned.linear ** Owned.parked refs.Owned.attention **
    pure (Global.available_contents kv)

inline_for_extraction noextract
fn initialize
  (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (hidden:array2 BF.t Native.hidden_layout) (best:larray FP.t 1) (next:larray I64.t 1)
  (#weights:erased Transfer.packed) (#prompt:erased (D.vec (D.token 248320) call.Model.prompt_length))
  preserves gpu
  requires pending call refs args hidden best next
  ensures Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.ready call weights prompt 0)
{
  unfold pending call refs args hidden best next;
  with state conv kv tokens history h old_best old_next.
    assert A.pts_to_mask args.Entry.state state Init.all_cells **
      A.pts_to_mask args.Entry.convolution conv Init.all_cells **
      A.pts_to_mask args.Entry.cache kv Init.all_cells **
      A.pts_to_mask args.Entry.tokens tokens Init.all_cells **
      A.pts_to_mask args.Entry.hidden history Init.all_cells **
      hidden |-> h ** best |-> old_best ** next |-> old_next;
  Init.initialize_caches args.Entry.state args.Entry.convolution;
  next.(0sz) <- 0L;
  with next_values. assert next |-> next_values;
  assert pure (Seq.equal next_values (Seq.create 1 0L));
  Seq.lemma_eq_elim next_values (Seq.create 1 0L);
  rewrite next |-> next_values as next |-> Seq.create 1 0L;
  assert pure (Seq.equal old_best (Seq.create 1 (Seq.index old_best 0)));
  Seq.lemma_eq_elim old_best (Seq.create 1 (Seq.index old_best 0));
  rewrite best |-> old_best as best |-> Seq.create 1 (Seq.index old_best 0);
  let fresh : erased (K.metadata call) = hide #(K.metadata call) (initial call tokens history (Seq.index old_best 0));
  initial_valid call weights prompt tokens history (Seq.index old_best 0) h kv;
  unfold Owned.parked refs.Owned.global;
  with previous. assert GR.pts_to refs.Owned.global previous;
  GR.write refs.Owned.global fresh;
  fold Owned.other_memory call args best next (reveal #(K.metadata call) fresh).K.memory;
  fold Owned.memory call args best next (reveal #(K.metadata call) fresh).K.memory;
  fold Owned.local_owners call refs args best next fresh;
  fold Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.ready call weights prompt 0);
}
