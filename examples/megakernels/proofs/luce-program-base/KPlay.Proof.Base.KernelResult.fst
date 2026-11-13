module KPlay.Proof.Base.KernelResult

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
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Shared = KPlay.SharedSpec
module K = KPlay.Proof.Base.KernelState
module Position = KPlay.Proof.Base.KernelPosition
module Trace = KPlay.Proof.Base.Trace
module Entry = KPlay.Proof.Base.Entry
module Owned = KPlay.Proof.Base.KernelOwnership
module Controls = KPlay.Proof.Base.KernelControls
module Book = KPlay.Proof.Base.WarpBook
module Native = KPlay.Proof.Base.NativeProjection
module Inputs = KPlay.Proof.Base.Inputs
module Transfer = KPlay.Proof.Base.Transfer
module Cache = KPlay.Proof.Base.CacheNative
module Global = KPlay.Proof.Base.GlobalCache
module Init = KPlay.Proof.Base.Initialization
module Hidden = KPlay.Proof.Base.WarpHidden

inline_for_extraction noextract
let complete (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (m:K.metadata call) (kv:Cache.snapshot) : prop =
  m.K.completed == Model.calls call /\ m.K.phase == K.Between /\
  K.common call weights prompt m /\ K.previous_selection call m /\
  Global.represents (K.caller call weights prompt m).D.model_cache m.K.memory.K.state m.K.memory.K.convolution kv

inline_for_extraction noextract
let output (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length)
  (m:K.metadata call{m.K.completed == Model.calls call})
  : GTot (D.histories 248320 1024 BF.t call.Model.decode_steps) =
  (K.caller call weights prompt m).D.returned

inline_for_extraction noextract
let run (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length)
  (m:K.metadata call{m.K.completed == Model.calls call})
  : GTot (Shared.execution call O.cache_configuration BF.t BF.t FP.t BF.t BF.t FP.t) =
  Trace.full call (Inputs.parameter_values weights) prompt K.configuration m.K.history

inline_for_extraction noextract
let refines (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (m:K.metadata call) (kv:Cache.snapshot)
  : Lemma
      (requires complete call weights prompt m kv)
      (ensures m.K.completed == Model.calls call /\
        Shared.postcondition call (fun x -> x) (Inputs.parameter_values weights) prompt (run call weights prompt m)
          (output call weights prompt m) (K.caller call weights prompt m)) =
  K.configuration_ok ();
  Trace.full_postcondition call (Inputs.parameter_values weights) prompt K.configuration m.K.history

inline_for_extraction noextract
let output_cells (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (m:K.metadata call) (kv:Cache.snapshot)
  : Lemma
      (requires complete call weights prompt m kv)
      (ensures
        (forall (i:P.index call.Model.decode_steps). Some? (Seq.index m.K.memory.K.tokens i)) /\
        (forall (i:P.index (call.Model.decode_steps * 1024)). Some? (Seq.index m.K.memory.K.hidden i)) /\
        Global.available_contents kv) =
  let hidden (i:P.index (call.Model.decode_steps * 1024))
    : Lemma (Some? (Seq.index m.K.memory.K.hidden i)) =
    assert (i / 1024 < call.Model.decode_steps /\ i % 1024 < 1024);
    assert (i == (i / 1024) * 1024 + i % 1024)
  in
  FStar.Classical.forall_intro hidden

inline_for_extraction noextract
let post (call:Model.invocation) (args:Entry.arguments call) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) : slprop =
  exists* (m:K.metadata call) (kv:Cache.snapshot).
    args.Entry.state |-> m.K.memory.K.state ** A.pts_to args.Entry.convolution m.K.memory.K.convolution **
    A.pts_to_mask args.Entry.cache kv Init.all_cells **
    A.pts_to_mask args.Entry.tokens m.K.memory.K.tokens Init.all_cells **
    A.pts_to_mask args.Entry.hidden m.K.memory.K.hidden Init.all_cells **
    pure (complete call weights prompt m kv)

inline_for_extraction noextract
let shared_owners (hidden:array2 BF.t Native.hidden_layout) (best:larray FP.t 1) (next:larray I64.t 1) : slprop =
  exists* (h:Hidden.snapshot) (b:lseq FP.t 1) (n:lseq I64.t 1).
    hidden |-> h ** best |-> b ** next |-> n

ghost
fn detach
  (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (hidden:array2 BF.t Native.hidden_layout) (best:larray FP.t 1) (next:larray I64.t 1)
  (weights:Transfer.packed) (prompt:D.vec (D.token 248320) call.Model.prompt_length)
  requires Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.ready call weights prompt (Model.calls call))
  ensures shared_owners hidden best next ** post call args weights prompt
{
  unfold Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.ready call weights prompt (Model.calls call));
  with m h kv. assert GR.pts_to refs.Owned.global m ** hidden |-> h ** A.pts_to_mask args.Entry.cache kv Init.all_cells;
  Position.ready_facts call weights prompt (Model.calls call) m h kv;
  unfold Owned.local_owners call refs args best next m;
  unfold Owned.memory call args best next (reveal #(K.metadata call) m).K.memory;
  unfold Owned.other_memory call args best next (reveal #(K.metadata call) m).K.memory;
  fold shared_owners hidden best next;
  fold Owned.parked refs.Owned.global;
  fold Controls.parked call refs;
  Controls.free call refs;
  fold post call args weights prompt;
}
