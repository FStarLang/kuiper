module KPlay.Proof.Optimized.Kernel

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
module S = FStar.Seq
module A = Pulse.Lib.Array
module G = FStar.Ghost
module GR = Pulse.Lib.GhostReference
module SZ = Kuiper.SizeT
module I32 = FStar.Int32
module I64 = FStar.Int64
module IC = FStar.Int.Cast
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module Entry = KPlay.Proof.Optimized.Entry
module State = KPlay.Proof.Optimized.State
module Memory = KPlay.Proof.Optimized.Memory
module Init = KPlay.Proof.Optimized.Initialization
module Embedding = KPlay.Proof.Optimized.Embedding
module Step = KPlay.Proof.Optimized.Step
module Execution = KPlay.Proof.Optimized.Execution
module Layout = KPlay.Proof.Optimized.Layout

inline_for_extraction noextract
let shared_cells (hidden normalized:larray bf16 1024)
  (projected:larray f32 8192) (activation:larray f32 2048) : slprop =
  (exists* (h:lseq bf16 1024). hidden |-> h) **
  (exists* (n:lseq bf16 1024). normalized |-> n) **
  (exists* (p:lseq f32 8192). projected |-> p) **
  (exists* (a:lseq f32 2048). activation |-> a)

inline_for_extraction noextract
let allocations (call:Model.invocation) (args:Entry.arguments call) : slprop =
  (exists* (s:lseq f32 4718592). args.Entry.states |-> s) **
  (exists* (c:lseq f32 442368). args.Entry.convolution |-> c) **
  (exists* (k:lseq bf16 6291456). args.Entry.keys |-> k) **
  (exists* (v:lseq bf16 6291456). args.Entry.values |-> v) **
  (exists* (t:lseq I64.t call.decode_steps). args.Entry.tokens |-> t) **
  (exists* (h:lseq bf16 (call.decode_steps * 1024)). args.Entry.hidden |-> h)

inline_for_extraction noextract
let readers (call:Model.invocation) (args:Entry.arguments call) (w:Layout.packed)
  (physical:lseq I64.t call.prompt_length) (permission:perm) : slprop =
  A.pts_to args.Entry.weights #permission w ** args.Entry.prompt |-> Frac permission physical

inline_for_extraction noextract
let pre (call:Model.invocation) (args:Entry.arguments call) (w:Layout.packed)
  (physical:lseq I64.t call.prompt_length)
  (prompt:D.vec (D.token 248320) call.prompt_length) (permission:perm) : slprop =
  readers call args w physical permission ** allocations call args **
  pure (forall (i:nat{i < call.prompt_length}). I64.v (S.index physical i) == S.index prompt i)

inline_for_extraction noextract
let result (call:Model.invocation) (args:Entry.arguments call) (w:Layout.packed)
  (prompt:D.vec (D.token 248320) call.prompt_length) : slprop =
  exists* (p:Execution.prefix call) (tokens:lseq (option I64.t) call.decode_steps)
          (history:lseq (option bf16) (call.decode_steps * 1024)).
    A.pts_to args.Entry.states (Execution.physical p).State.states **
    A.pts_to args.Entry.convolution (Execution.physical p).State.convolution **
    A.pts_to args.Entry.keys (Execution.physical p).State.keys **
    A.pts_to args.Entry.values (Execution.physical p).State.values **
    A.pts_to_mask args.Entry.tokens tokens Memory.all_cells **
    A.pts_to_mask args.Entry.hidden history Memory.all_cells **
    pure (p.Execution.count == Model.calls call /\ Execution.well_formed w prompt p /\
      Execution.outputs_agree w prompt p tokens history)

inline_for_extraction noextract
let post (call:Model.invocation) (args:Entry.arguments call) (w:Layout.packed)
  (physical:lseq I64.t call.prompt_length)
  (prompt:D.vec (D.token 248320) call.prompt_length) (permission:perm) : slprop =
  readers call args w physical permission ** result call args w prompt **
  pure (forall (i:nat{i < call.prompt_length}). I64.v (S.index physical i) == S.index prompt i)

noextract
fn source (call:Model.invocation) (args:Entry.arguments call)
  (hidden normalized:larray bf16 1024) (projected:larray f32 8192) (activation:larray f32 2048)
  (#w:erased Layout.packed) (#physical:erased (lseq I64.t call.prompt_length))
  (#prompt:erased (D.vec (D.token 248320) call.prompt_length)) (#permission:perm)
  preserves gpu
  requires pre call args w physical prompt permission ** shared_cells hidden normalized projected activation
  ensures post call args w physical prompt permission ** shared_cells hidden normalized projected activation
{
  unfold pre call args w physical prompt permission;
  unfold readers call args w physical permission;
  unfold allocations call args;
  with sv cv kv vv tv hv. assert args.Entry.states |-> sv ** args.Entry.convolution |-> cv **
    args.Entry.keys |-> kv ** args.Entry.values |-> vv ** args.Entry.tokens |-> tv ** args.Entry.hidden |-> hv;
  A.to_mask args.Entry.states;
  A.to_mask args.Entry.convolution;
  A.to_mask args.Entry.keys;
  A.to_mask args.Entry.values;
  A.to_mask args.Entry.tokens;
  A.to_mask args.Entry.hidden;
  Init.source args.Entry.states args.Entry.convolution args.Entry.keys args.Entry.values;
  unfold shared_cells hidden normalized projected activation;
  with hs ns ps ac. assert hidden |-> hs ** normalized |-> ns ** projected |-> ps ** activation |-> ac;
  A.to_mask hidden;
  A.to_mask normalized;
  A.to_mask projected;
  A.to_mask activation;
  Model.full_profile call;
  let length = SZ.uint32_to_sizet (IC.int64_to_uint32 args.Entry.prompt_length);
  assert pure (SZ.v length == call.prompt_length);
  let steps = SZ.uint32_to_sizet (IC.int64_to_uint32 args.Entry.steps);
  assert pure (SZ.v steps == call.decode_steps);
  let count = length +^ steps -^ 1sz;
  assert pure (SZ.v count == Model.calls call);
  let first = args.Entry.prompt.(0sz);
  let mut token:Embedding.token = Embedding.narrow first;
  let trace = GR.alloc #(Execution.prefix call) (Execution.empty call);
  let mut position = 0sz;
  while (!position <^ count)
    invariant exists* (t:sz{SZ.v t <= Model.calls call}) (carried:Embedding.token)
                     (p:Execution.prefix call)
                     (tokens:lseq (option I64.t) call.decode_steps)
                     (history:lseq (option bf16) (call.decode_steps * 1024))
                     (hh nn:lseq (option bf16) 1024)
                     (pp:lseq (option f32) 8192) (aa:lseq (option f32) 2048).
      position |-> t ** token |-> carried ** trace |-> p **
      A.pts_to args.Entry.states (Execution.physical p).State.states **
      A.pts_to args.Entry.convolution (Execution.physical p).State.convolution **
      A.pts_to args.Entry.keys (Execution.physical p).State.keys **
      A.pts_to args.Entry.values (Execution.physical p).State.values **
      A.pts_to_mask args.Entry.tokens tokens Memory.all_cells **
      A.pts_to_mask args.Entry.hidden history Memory.all_cells **
      A.pts_to_mask hidden hh Memory.all_cells ** A.pts_to_mask normalized nn Memory.all_cells **
      A.pts_to_mask projected pp Memory.all_cells ** A.pts_to_mask activation aa Memory.all_cells **
      pure (p.Execution.count == SZ.v t /\ Execution.well_formed w prompt p /\
        Execution.outputs_agree w prompt p tokens history /\
        (p.Execution.count > 0 ==>
          I32.v carried == I32.v (Execution.data_at p (p.Execution.count - 1)).Step.chosen) /\
        (SZ.v t > 0 ==> Memory.ready hh 0 1024 /\ Memory.ready nn 0 1024 /\
          Memory.ready pp 0 8192 /\ Memory.ready aa 0 2048))
    decreases (SZ.v count - SZ.v !position)
  {
    let t = !position;
    let p = GR.read trace;
    Execution.state_represents w prompt p;
    Execution.input_token w prompt physical p (!token);
    let selected = Embedding.select_input call args.Entry.prompt t length (!token);
    token := selected;
    with old_tokens. assert A.pts_to_mask args.Entry.tokens old_tokens Memory.all_cells;
    with old_history. assert A.pts_to_mask args.Entry.hidden old_history Memory.all_cells;
    let produced = Step.source call args.Entry.weights hidden normalized projected activation
      args.Entry.states args.Entry.convolution args.Entry.keys args.Entry.values
      args.Entry.tokens args.Entry.hidden t length selected
      #(G.hide #(lseq bf16 752393024) (G.reveal #Layout.packed w))
      #(G.hide (Execution.cache p)) #(G.hide (Execution.physical p))
      #_ #_ #_ #_ #old_tokens #old_history #permission;
    let data = snd produced;
    token := fst produced;
    Execution.append_well_formed w prompt p data;
    Execution.outputs_append w prompt p data old_tokens old_history;
    Execution.common_append p data;
    let next:erased (Execution.prefix call) = G.hide (Execution.append p data);
    GR.write trace next;
    A.to_mask hidden;
    position := t +^ 1sz;
  };
  let end_position = !position;
  assert pure (SZ.v end_position == Model.calls call);
  let completed = GR.read trace;
  assert pure (completed.Execution.count == Model.calls call);
  GR.free trace;
  A.from_mask hidden;
  A.from_mask normalized;
  A.from_mask projected;
  A.from_mask activation;
  fold shared_cells hidden normalized projected activation;
  fold result call args w prompt;
  fold readers call args w physical permission;
  fold post call args w physical prompt permission;
}
