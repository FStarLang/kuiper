module KPlay.Proof.Optimized.Queue

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
open Pulse.Lib.Send
open FStar.Tactics.Typeclasses { solve }
module A = Pulse.Lib.Array
module I64 = FStar.Int64
module RT = Kuiper.Tensor.Runtime
module Loc = Kuiper.Locs
module Stream = Kuiper.Kernel.Stream
module Epoch = Kuiper.Epoch
module Pledge = Pulse.Lib.Pledge
module NativeInput = KPlay.Proof.NativeInput
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module Layout = KPlay.Proof.Optimized.Layout
module Caller = KPlay.Proof.Optimized.Caller
module Entry = KPlay.Proof.Optimized.Entry
module Kernel = KPlay.Proof.Optimized.Kernel
module Return = KPlay.Proof.Optimized.Return

inline_for_extraction noextract
instance contents_placeless (#a:Type0) (value:RT.tensor a) : placeless (RT.contents value) = solve

inline_for_extraction noextract
let source_input (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (w:Layout.packed) (physical:lseq I64.t call.prompt_length)
  (prompt:D.vec (D.token 248320) call.prompt_length) (permission:perm) : slprop =
  on Loc.gpu_loc (Return.input_cells call inputs w physical prompt permission)

inline_for_extraction noextract
let result_cells (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (tokens:RT.tensor I64.t) (hidden:RT.tensor bf16) (w:Layout.packed)
  (physical:lseq I64.t call.prompt_length) (prompt:D.vec (D.token 248320) call.prompt_length)
  (permission:perm) : slprop =
  exists* (args:Entry.arguments call).
    pure (args.Entry.tokens == RT.data tokens /\ args.Entry.hidden == RT.data hidden) **
    Return.exported call inputs args w physical prompt permission

ghost
fn prepare_on (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (b:Caller.buffers call) (args:Entry.arguments call{Entry.bound call inputs b args})
  (w:Layout.packed) (physical:lseq I64.t call.prompt_length)
  (prompt:D.vec (D.token 248320) call.prompt_length) (permission:perm) ()
  requires Caller.pending_contents call b (source_input call inputs w physical prompt permission)
  ensures on Loc.gpu_loc (Kernel.pre call args w physical prompt permission)
{
  ghost_impersonate #emp_inames Loc.gpu_loc
    (Caller.pending_contents call b (source_input call inputs w physical prompt permission))
    (on Loc.gpu_loc (Kernel.pre call args w physical prompt permission))
    fn () {
      unfold Caller.pending_contents call b (source_input call inputs w physical prompt permission);
      unfold RT.contents b.Caller.states;
      unfold RT.contents b.Caller.convolution;
      unfold RT.contents b.Caller.keys;
      unfold RT.contents b.Caller.values;
      unfold RT.contents b.Caller.tokens;
      unfold RT.contents b.Caller.hidden;
      with states conv keys values tokens hidden.
        assert on Loc.gpu_loc (A.pts_to (RT.data b.Caller.states) states) **
          on Loc.gpu_loc (A.pts_to (RT.data b.Caller.convolution) conv) **
          on Loc.gpu_loc (A.pts_to (RT.data b.Caller.keys) keys) **
          on Loc.gpu_loc (A.pts_to (RT.data b.Caller.values) values) **
          on Loc.gpu_loc (A.pts_to (RT.data b.Caller.tokens) tokens) **
          on Loc.gpu_loc (A.pts_to (RT.data b.Caller.hidden) hidden);
      on_elim (A.pts_to (RT.data b.Caller.states) states);
      on_elim (A.pts_to (RT.data b.Caller.convolution) conv);
      on_elim (A.pts_to (RT.data b.Caller.keys) keys);
      on_elim (A.pts_to (RT.data b.Caller.values) values);
      on_elim (A.pts_to (RT.data b.Caller.tokens) tokens);
      on_elim (A.pts_to (RT.data b.Caller.hidden) hidden);
      rewrite A.pts_to (RT.data b.Caller.states) states as A.pts_to args.Entry.states states;
      rewrite A.pts_to (RT.data b.Caller.convolution) conv as A.pts_to args.Entry.convolution conv;
      rewrite A.pts_to (RT.data b.Caller.keys) keys as A.pts_to args.Entry.keys keys;
      rewrite A.pts_to (RT.data b.Caller.values) values as A.pts_to args.Entry.values values;
      rewrite A.pts_to (RT.data b.Caller.tokens) tokens as A.pts_to args.Entry.tokens tokens;
      rewrite A.pts_to (RT.data b.Caller.hidden) hidden as A.pts_to args.Entry.hidden hidden;
      A.pts_to_len args.Entry.states;
      A.pts_to_len args.Entry.convolution;
      A.pts_to_len args.Entry.keys;
      A.pts_to_len args.Entry.values;
      A.pts_to_len args.Entry.tokens;
      A.pts_to_len args.Entry.hidden;
      fold Kernel.allocations call args;
      unfold source_input call inputs w physical prompt permission;
      on_elim (Return.input_cells call inputs w physical prompt permission);
      Return.prepare call inputs args w physical prompt permission ();
      on_intro (Kernel.pre call args w physical prompt permission);
    };
}

ghost
fn export_on (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (b:Caller.buffers call) (args:Entry.arguments call{Entry.bound call inputs b args})
  (w:Layout.packed) (physical:lseq I64.t call.prompt_length)
  (prompt:D.vec (D.token 248320) call.prompt_length) (permission:perm) ()
  requires on Loc.gpu_loc (Kernel.post call args w physical prompt permission)
  ensures Entry.private_contents call b
    (on Loc.gpu_loc (result_cells call inputs b.Caller.tokens b.Caller.hidden w physical prompt permission))
{
  ghost_impersonate #emp_inames Loc.gpu_loc
    (on Loc.gpu_loc (Kernel.post call args w physical prompt permission))
    (Entry.private_contents call b
      (on Loc.gpu_loc (result_cells call inputs b.Caller.tokens b.Caller.hidden w physical prompt permission)))
    fn () {
      on_elim (Kernel.post call args w physical prompt permission);
      Return.export call inputs args w physical prompt permission ();
      fold result_cells call inputs b.Caller.tokens b.Caller.hidden w physical prompt permission;
      on_intro (result_cells call inputs b.Caller.tokens b.Caller.hidden w physical prompt permission);
      unfold Return.private_arrays call args;
      with values keys conv states.
        assert args.Entry.values |-> values ** args.Entry.keys |-> keys **
          args.Entry.convolution |-> conv ** args.Entry.states |-> states;
      rewrite args.Entry.values |-> values as A.pts_to (RT.data b.Caller.values) values;
      rewrite args.Entry.keys |-> keys as A.pts_to (RT.data b.Caller.keys) keys;
      rewrite args.Entry.convolution |-> conv as A.pts_to (RT.data b.Caller.convolution) conv;
      rewrite args.Entry.states |-> states as A.pts_to (RT.data b.Caller.states) states;
      on_intro (A.pts_to (RT.data b.Caller.values) values);
      on_intro (A.pts_to (RT.data b.Caller.keys) keys);
      on_intro (A.pts_to (RT.data b.Caller.convolution) conv);
      on_intro (A.pts_to (RT.data b.Caller.states) states);
      fold RT.contents b.Caller.values;
      fold RT.contents b.Caller.keys;
      fold RT.contents b.Caller.convolution;
      fold RT.contents b.Caller.states;
      fold Entry.private_contents call b
        (on Loc.gpu_loc (result_cells call inputs b.Caller.tokens b.Caller.hidden w physical prompt permission));
    };
}

ghost
fn prepare_pledge (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (b:Caller.buffers call) (args:Entry.arguments call{Entry.bound call inputs b args})
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (w:Layout.packed) (physical:lseq I64.t call.prompt_length)
  (prompt:D.vec (D.token 248320) call.prompt_length) (permission:perm) ()
  requires Pledge.pledge0 (Epoch.epoch_done stream epoch)
    (Caller.pending_contents call b (source_input call inputs w physical prompt permission))
  ensures Pledge.pledge0 (Epoch.epoch_done stream epoch)
    (on Loc.gpu_loc (Kernel.pre call args w physical prompt permission))
{
  Pledge.rewrite_pledge
    (Caller.pending_contents call b (source_input call inputs w physical prompt permission))
    (on Loc.gpu_loc (Kernel.pre call args w physical prompt permission))
    (prepare_on call inputs b args w physical prompt permission);
}

ghost
fn export_pledge (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (b:Caller.buffers call) (args:Entry.arguments call{Entry.bound call inputs b args})
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (w:Layout.packed) (physical:lseq I64.t call.prompt_length)
  (prompt:D.vec (D.token 248320) call.prompt_length) (permission:perm) ()
  requires Pledge.pledge0 (Epoch.epoch_done stream epoch)
    (on Loc.gpu_loc (Kernel.post call args w physical prompt permission))
  ensures Pledge.pledge0 (Epoch.epoch_done stream epoch)
    (Entry.private_contents call b
      (on Loc.gpu_loc (result_cells call inputs b.Caller.tokens b.Caller.hidden w physical prompt permission)))
{
  Pledge.rewrite_pledge
    (on Loc.gpu_loc (Kernel.post call args w physical prompt permission))
    (Entry.private_contents call b
      (on Loc.gpu_loc (result_cells call inputs b.Caller.tokens b.Caller.hidden w physical prompt permission)))
    (export_on call inputs b args w physical prompt permission);
}
