module KPlay.Proof.Base.KernelQueue

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Array
open Pulse.Lib.Send
open FStar.Ghost
open FStar.Tactics.Typeclasses { solve }

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module BF = Kuiper.BFloat16
module I64 = FStar.Int64
module RT = Kuiper.Tensor.Runtime
module Loc = Kuiper.Locs
module Stream = Kuiper.Kernel.Stream
module Epoch = Kuiper.Epoch
module Pledge = Pulse.Lib.Pledge
module Input = Kuiper.Tensor.Input
module NativeInput = KPlay.Proof.NativeInput
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module Caller = KPlay.Proof.Base.Caller
module Entry = KPlay.Proof.Base.Entry
module Owned = KPlay.Proof.Base.KernelOwnership
module Controls = KPlay.Proof.Base.KernelControls
module Layout = KPlay.Proof.Base.KernelSetup
module Storage = KPlay.Proof.Base.KernelStorage
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
instance contents_placeless (#a:Type0) (value:RT.tensor a) : placeless (RT.contents value) = solve

inline_for_extraction noextract
instance controls_placeless (call:Model.invocation) (refs:Owned.controls call)
  : placeless (Controls.parked call refs) = solve

inline_for_extraction noextract
let source_input (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (values:Transfer.packed) (physical:lseq I64.t call.Model.prompt_length)
  (logical:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm) : slprop =
  on Loc.gpu_loc (Storage.input_cells call inputs values physical logical fraction)

inline_for_extraction noextract
let result_cells (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (tokens:RT.tensor I64.t) (hidden:RT.tensor BF.t)
  (values:Transfer.packed) (physical:lseq I64.t call.Model.prompt_length)
  (logical:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm) : slprop =
  exists* (args:Entry.arguments call).
    pure (args.Entry.tokens == RT.data tokens /\ args.Entry.hidden == RT.data hidden) **
    Storage.exported call inputs args values physical logical fraction

ghost
fn prepare_on
  (call:Model.invocation) (refs:Owned.controls call) (inputs:NativeInput.arguments call.Model.prompt_length)
  (b:Caller.buffers call)
  (args:Entry.arguments call{Entry.bound call inputs b args})
  (values:Transfer.packed) (physical:lseq I64.t call.Model.prompt_length)
  (logical:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm) ()
  requires Controls.parked call refs **
    Caller.pending_contents call b (source_input call inputs values physical logical fraction)
  ensures on Loc.gpu_loc (Layout.pre call refs args (Storage.tensor values) physical logical fraction)
{
  ghost_impersonate #emp_inames Loc.gpu_loc
    (Controls.parked call refs ** Caller.pending_contents call b (source_input call inputs values physical logical fraction))
    (on Loc.gpu_loc (Layout.pre call refs args (Storage.tensor values) physical logical fraction))
    fn () {
      unfold Caller.pending_contents call b (source_input call inputs values physical logical fraction);
      unfold RT.contents b.Caller.state;
      unfold RT.contents b.Caller.convolution;
      unfold RT.contents b.Caller.cache;
      unfold RT.contents b.Caller.tokens;
      unfold RT.contents b.Caller.hidden;
      with state conv kv tokens hidden.
        assert on Loc.gpu_loc (A.pts_to (RT.data b.Caller.state) state) **
          on Loc.gpu_loc (A.pts_to (RT.data b.Caller.convolution) conv) **
          on Loc.gpu_loc (A.pts_to (RT.data b.Caller.cache) kv) **
          on Loc.gpu_loc (A.pts_to (RT.data b.Caller.tokens) tokens) **
          on Loc.gpu_loc (A.pts_to (RT.data b.Caller.hidden) hidden);
      on_elim (A.pts_to (RT.data b.Caller.state) state);
      on_elim (A.pts_to (RT.data b.Caller.convolution) conv);
      on_elim (A.pts_to (RT.data b.Caller.cache) kv);
      on_elim (A.pts_to (RT.data b.Caller.tokens) tokens);
      on_elim (A.pts_to (RT.data b.Caller.hidden) hidden);
      rewrite A.pts_to (RT.data b.Caller.state) state as A.pts_to args.Entry.state state;
      rewrite A.pts_to (RT.data b.Caller.convolution) conv as A.pts_to args.Entry.convolution conv;
      rewrite A.pts_to (RT.data b.Caller.cache) kv as A.pts_to args.Entry.cache kv;
      rewrite A.pts_to (RT.data b.Caller.tokens) tokens as A.pts_to args.Entry.tokens tokens;
      rewrite A.pts_to (RT.data b.Caller.hidden) hidden as A.pts_to args.Entry.hidden hidden;
      A.pts_to_len args.Entry.state;
      A.pts_to_len args.Entry.convolution;
      A.pts_to_len args.Entry.cache;
      A.pts_to_len args.Entry.tokens;
      A.pts_to_len args.Entry.hidden;
      fold Layout.allocation_contents call args;
      unfold source_input call inputs values physical logical fraction;
      on_elim (Storage.input_cells call inputs values physical logical fraction);
      Storage.prepare call refs inputs args values physical logical fraction ();
      on_intro (Layout.pre call refs args (Storage.tensor values) physical logical fraction);
    };
}

ghost
fn export_on
  (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (b:Caller.buffers call) (args:Entry.arguments call{Entry.bound call inputs b args})
  (values:Transfer.packed) (physical:lseq I64.t call.Model.prompt_length)
  (logical:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm) ()
  requires on Loc.gpu_loc (Layout.post call args (Storage.tensor values) physical logical fraction)
  ensures Entry.private_contents call b
    (on Loc.gpu_loc (result_cells call inputs b.Caller.tokens b.Caller.hidden values physical logical fraction))
{
  ghost_impersonate #emp_inames Loc.gpu_loc
    (on Loc.gpu_loc (Layout.post call args (Storage.tensor values) physical logical fraction))
    (Entry.private_contents call b
      (on Loc.gpu_loc (result_cells call inputs b.Caller.tokens b.Caller.hidden values physical logical fraction)))
    fn () {
      on_elim (Layout.post call args (Storage.tensor values) physical logical fraction);
      Storage.export call inputs args values physical logical fraction ();
      fold result_cells call inputs b.Caller.tokens b.Caller.hidden values physical logical fraction;
      on_intro (result_cells call inputs b.Caller.tokens b.Caller.hidden values physical logical fraction);
      unfold Storage.private_arrays call args;
      with kv conv state. assert args.Entry.cache |-> kv ** args.Entry.convolution |-> conv ** args.Entry.state |-> state;
      rewrite args.Entry.cache |-> kv as A.pts_to (RT.data b.Caller.cache) kv;
      rewrite args.Entry.convolution |-> conv as A.pts_to (RT.data b.Caller.convolution) conv;
      rewrite args.Entry.state |-> state as A.pts_to (RT.data b.Caller.state) state;
      on_intro (A.pts_to (RT.data b.Caller.cache) kv);
      on_intro (A.pts_to (RT.data b.Caller.convolution) conv);
      on_intro (A.pts_to (RT.data b.Caller.state) state);
      fold RT.contents b.Caller.cache;
      fold RT.contents b.Caller.convolution;
      fold RT.contents b.Caller.state;
      fold Entry.private_contents call b
        (on Loc.gpu_loc (result_cells call inputs b.Caller.tokens b.Caller.hidden values physical logical fraction));
    };
}

ghost
fn prepare_pledge
  (call:Model.invocation) (refs:Owned.controls call) (inputs:NativeInput.arguments call.Model.prompt_length)
  (b:Caller.buffers call) (args:Entry.arguments call{Entry.bound call inputs b args})
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (values:Transfer.packed) (physical:lseq I64.t call.Model.prompt_length)
  (logical:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm) ()
  requires Controls.parked call refs **
    Pledge.pledge0 (Epoch.epoch_done stream epoch)
      (Caller.pending_contents call b (source_input call inputs values physical logical fraction))
  ensures Pledge.pledge0 (Epoch.epoch_done stream epoch)
    (on Loc.gpu_loc (Layout.pre call refs args (Storage.tensor values) physical logical fraction))
{
  Pledge.return_pledge (Epoch.epoch_done stream epoch) (Controls.parked call refs)
    #(solve #(is_send (Controls.parked call refs)));
  Pledge.join_pledge (Controls.parked call refs)
    (Caller.pending_contents call b (source_input call inputs values physical logical fraction));
  Pledge.rewrite_pledge
    (Controls.parked call refs ** Caller.pending_contents call b (source_input call inputs values physical logical fraction))
    (on Loc.gpu_loc (Layout.pre call refs args (Storage.tensor values) physical logical fraction))
    (prepare_on call refs inputs b args values physical logical fraction);
}

ghost
fn export_pledge
  (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (b:Caller.buffers call) (args:Entry.arguments call{Entry.bound call inputs b args})
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (values:Transfer.packed) (physical:lseq I64.t call.Model.prompt_length)
  (logical:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm) ()
  requires Pledge.pledge0 (Epoch.epoch_done stream epoch)
    (on Loc.gpu_loc (Layout.post call args (Storage.tensor values) physical logical fraction))
  ensures Pledge.pledge0 (Epoch.epoch_done stream epoch)
    (Entry.private_contents call b
      (on Loc.gpu_loc (result_cells call inputs b.Caller.tokens b.Caller.hidden values physical logical fraction)))
{
  Pledge.rewrite_pledge
    (on Loc.gpu_loc (Layout.post call args (Storage.tensor values) physical logical fraction))
    (Entry.private_contents call b
      (on Loc.gpu_loc (result_cells call inputs b.Caller.tokens b.Caller.hidden values physical logical fraction)))
    (export_on call inputs b args values physical logical fraction);
}
