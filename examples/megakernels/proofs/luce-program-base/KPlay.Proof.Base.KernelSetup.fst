module KPlay.Proof.Base.KernelSetup

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor
open Kuiper.ForEvery
open Pulse.Lib.Array
open Pulse.Lib.Send
open FStar.Ghost

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module C = Kuiper.Chest
module T = Kuiper.Tensor
module Mem = Kuiper.Array.Core
module Extra = Kuiper.Array.Extra
module Every = Kuiper.ForEvery
module SH = Kuiper.SHMem
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module I64 = FStar.Int64
module D = KPlay.Proof.Shared.Selection
module P = KPlay.Proof.Shared.Projection
module Model = KPlay.Proof.Shared.Model
module K = KPlay.Proof.Base.KernelState
module Entry = KPlay.Proof.Base.Entry
module Owned = KPlay.Proof.Base.KernelOwnership
module Controls = KPlay.Proof.Base.KernelControls
module Book = KPlay.Proof.Base.WarpBook
module Head = KPlay.Proof.Base.Head
module Native = KPlay.Proof.Base.NativeProjection
module Transfer = KPlay.Proof.Base.Transfer
module Start = KPlay.Proof.Base.KernelStart
module Result = KPlay.Proof.Base.KernelResult
module Init = KPlay.Proof.Base.Initialization
module Global = KPlay.Proof.Base.GlobalCache
module Hidden = KPlay.Proof.Base.WarpHidden

// Native int64_t width; the descriptor default performs no shared-memory store.
inline_for_extraction noextract
instance shared_int64_size : Kuiper.Sized.sized I64.t = {size = 8sz; default = 0L}

inline_for_extraction noextract
let declarations : list SH.shmem_desc = [SH.SHArray BF.t 16000sz; SH.SHArray I64.t 1sz; SH.SHArray FP.t 1sz]

inline_for_extraction noextract
let shared = SH.c_shmems declarations

inline_for_extraction noextract
let hidden (sh:shared) = T.from_array Native.hidden_layout sh._1

inline_for_extraction noextract
let next (sh:shared) = sh._2._1

inline_for_extraction noextract
let best (sh:shared) = sh._2._2._1

inline_for_extraction noextract
let weight (call:Model.invocation) (args:Entry.arguments call) = T.from_array Native.weight_layout args.Entry.weight

inline_for_extraction noextract
let device_arguments (call:Model.invocation) (args:Entry.arguments call) : prop =
  Mem.is_global_array args.Entry.weight /\ Mem.is_global_array args.Entry.prompt /\
  Mem.is_global_array args.Entry.state /\ Mem.is_global_array args.Entry.convolution /\
  Mem.is_global_array args.Entry.cache /\ Mem.is_global_array args.Entry.tokens /\ Mem.is_global_array args.Entry.hidden

inline_for_extraction noextract
let readers (call:Model.invocation) (args:Entry.arguments call)
  (vw:C.chest2 BF.t 752393024 1) (sp:lseq I64.t call.Model.prompt_length) (fraction:perm) : slprop =
  weight call args |-> Frac fraction vw ** A.pts_to args.Entry.prompt #fraction sp

inline_for_extraction noextract
let allocation_contents (call:Model.invocation) (args:Entry.arguments call) : slprop =
  (exists* (s:lseq FP.t 4718592). args.Entry.state |-> s) **
  (exists* (s:lseq BF.t 442368). args.Entry.convolution |-> s) **
  (exists* (s:lseq BF.t 12582912). args.Entry.cache |-> s) **
  (exists* (s:lseq I64.t call.Model.decode_steps). args.Entry.tokens |-> s) **
  (exists* (s:lseq BF.t (call.Model.decode_steps * 1024)). args.Entry.hidden |-> s)

inline_for_extraction noextract
let pre (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (vw:C.chest2 BF.t 752393024 1) (sp:lseq I64.t call.Model.prompt_length)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm) : slprop =
  readers call args vw sp fraction ** allocation_contents call args ** Controls.parked call refs **
  pure (forall (i:P.index call.Model.prompt_length). I64.v (Seq.index sp i) == Seq.index prompt i)

inline_for_extraction noextract
let post (call:Model.invocation) (args:Entry.arguments call)
  (vw:C.chest2 BF.t 752393024 1) (sp:lseq I64.t call.Model.prompt_length)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm) : slprop =
  readers call args vw sp fraction ** Result.post call args (Native.packed_values vw) prompt **
  pure (forall (i:P.index call.Model.prompt_length). I64.v (Seq.index sp i) == Seq.index prompt i)

inline_for_extraction noextract
let kpre (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (vw:C.chest2 BF.t 752393024 1) (sp:lseq I64.t call.Model.prompt_length)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm) (sh:shared) (i:Head.lane) : slprop =
  readers call args vw sp (fraction /. 32.0R) **
  Head.lane_only i (Start.pending call refs args (hidden sh) (best sh) (next sh)) **
  pure (forall (j:P.index call.Model.prompt_length). I64.v (Seq.index sp j) == Seq.index prompt j)

inline_for_extraction noextract
let kpost (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (vw:C.chest2 BF.t 752393024 1) (sp:lseq I64.t call.Model.prompt_length)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm) (sh:shared) (i:Head.lane) : slprop =
  readers call args vw sp (fraction /. 32.0R) **
  Head.lane_only i
    (Book.full refs.Owned.global (hidden sh) args.Entry.cache (Owned.local_owners call refs args (best sh) (next sh))
      (K.ready call (Native.packed_values vw) prompt (Model.calls call))) **
  pure (forall (j:P.index call.Model.prompt_length). I64.v (Seq.index sp j) == Seq.index prompt j)

ghost
fn shared_open (sh:shared)
  requires SH.live_c_shmems sh
  ensures Result.shared_owners (hidden sh) (best sh) (next sh)
{
  SH.unfold_live_c_shmems_cons sh #1.0R;
  SH.unfold_live_c_shmems_cons sh._2 #1.0R;
  SH.unfold_live_c_shmems_cons sh._2._2 #1.0R;
  SH.unfold_live_c_shmems_nil sh._2._2._2 #1.0R;
  SH.unfold_live_c_shmem #(SH.SHArray BF.t 16000sz) sh._1 #1.0R;
  SH.unfold_live_c_shmem #(SH.SHArray I64.t 1sz) (next sh) #1.0R;
  SH.unfold_live_c_shmem #(SH.SHArray FP.t 1sz) (best sh) #1.0R;
  with h n b. assert sh._1 |-> h ** next sh |-> n ** best sh |-> b;
  A.pts_to_len sh._1;
  A.pts_to_len (next sh);
  A.pts_to_len (best sh);
  T.to_from Native.hidden_layout h;
  T.tensor_abs Native.hidden_layout sh._1 #1.0R #(T.from_seq Native.hidden_layout h);
  fold Result.shared_owners (hidden sh) (best sh) (next sh);
}

ghost
fn shared_close (sh:shared)
  requires Result.shared_owners (hidden sh) (best sh) (next sh)
  ensures SH.live_c_shmems sh
{
  unfold Result.shared_owners (hidden sh) (best sh) (next sh);
  with h b n. assert hidden sh |-> h ** best sh |-> b ** next sh |-> n;
  T.tensor_concr (hidden sh);
  T.lem_from_array_core #BF.t #2 #_ #Native.hidden_layout sh._1;
  rewrite A.pts_to (T.core (hidden sh)) (T.to_seq Native.hidden_layout h)
    as A.pts_to sh._1 (T.to_seq Native.hidden_layout h);
  SH.fold_live_c_shmem #(SH.SHArray BF.t 16000sz) sh._1 #1.0R;
  SH.fold_live_c_shmem #(SH.SHArray I64.t 1sz) (next sh) #1.0R;
  SH.fold_live_c_shmem #(SH.SHArray FP.t 1sz) (best sh) #1.0R;
  SH.fold_live_c_shmems_nil sh._2._2._2 #1.0R;
  SH.fold_live_c_shmems_cons sh._2._2 #1.0R;
  SH.fold_live_c_shmems_cons sh._2 #1.0R;
  SH.fold_live_c_shmems_cons sh #1.0R;
}

ghost
fn block_setup
  (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (vw:C.chest2 BF.t 752393024 1) (sp:lseq I64.t call.Model.prompt_length)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm)
  (frame:slprop{frame == emp})
  (sh:shared) (bid:Kuiper.Common.natlt 1) ()
  requires SH.live_c_shmems sh ** pre call refs args vw sp prompt fraction
  ensures (forall+ (i:Head.lane). kpre call refs args vw sp prompt fraction sh i) ** frame
{
  unfold pre call refs args vw sp prompt fraction;
  unfold readers call args vw sp fraction;
  unfold allocation_contents call args;
  with state conv kv tokens history.
    assert args.Entry.state |-> state ** args.Entry.convolution |-> conv **
      args.Entry.cache |-> kv ** args.Entry.tokens |-> tokens ** args.Entry.hidden |-> history;
  A.to_mask args.Entry.state;
  A.to_mask args.Entry.convolution;
  A.to_mask args.Entry.cache;
  A.to_mask args.Entry.tokens;
  A.to_mask args.Entry.hidden;
  with state_mask conv_mask kv_mask token_mask history_mask.
    assert A.pts_to_mask args.Entry.state state_mask Init.all_cells **
      A.pts_to_mask args.Entry.convolution conv_mask Init.all_cells **
      A.pts_to_mask args.Entry.cache kv_mask Init.all_cells **
      A.pts_to_mask args.Entry.tokens token_mask Init.all_cells **
      A.pts_to_mask args.Entry.hidden history_mask Init.all_cells;
  assert pure (Global.available_contents kv_mask);
  shared_open sh;
  unfold Result.shared_owners (hidden sh) (best sh) (next sh);
  with h b n. assert hidden sh |-> h ** best sh |-> b ** next sh |-> n;
  unfold Controls.parked call refs;
  fold Start.pending call refs args (hidden sh) (best sh) (next sh);
  Head.give_lane_zero (Start.pending call refs args (hidden sh) (best sh) (next sh)) ();
  T.tensor_share_n (weight call args) 32;
  Extra.array_share args.Entry.prompt #sp #fraction 32;
  Every.forevery_zip3 #Head.lane
    (fun _ -> weight call args |-> Frac (fraction /. 32.0R) vw)
    (fun _ -> A.pts_to args.Entry.prompt #(fraction /. 32.0R) sp)
    (fun i -> Head.lane_only i (Start.pending call refs args (hidden sh) (best sh) (next sh)));
  Every.forevery_map #Head.lane
    (fun i -> weight call args |-> Frac (fraction /. 32.0R) vw **
      A.pts_to args.Entry.prompt #(fraction /. 32.0R) sp **
      Head.lane_only i (Start.pending call refs args (hidden sh) (best sh) (next sh)))
    (kpre call refs args vw sp prompt fraction sh)
    fn i {
      fold readers call args vw sp (fraction /. 32.0R);
      fold kpre call refs args vw sp prompt fraction sh i;
    };
  rewrite emp as frame;
}

ghost
fn block_teardown
  (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (vw:C.chest2 BF.t 752393024 1) (sp:lseq I64.t call.Model.prompt_length)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm)
  (frame:slprop{frame == emp})
  (sh:shared) (bid:Kuiper.Common.natlt 1) ()
  requires (forall+ (i:Head.lane). kpost call refs args vw sp prompt fraction sh i) ** frame
  ensures SH.live_c_shmems sh ** post call args vw sp prompt fraction
{
  rewrite frame as emp;
  Every.forevery_extract_if_eqtype #Head.lane 0 (kpost call refs args vw sp prompt fraction sh);
  unfold kpost call refs args vw sp prompt fraction sh 0;
  assert pure (forall (j:P.index call.Model.prompt_length). I64.v (Seq.index sp j) == Seq.index prompt j);
  fold kpost call refs args vw sp prompt fraction sh 0;
  Every.forevery_unextract_if_eqtype #Head.lane 0 (kpost call refs args vw sp prompt fraction sh);
  Every.forevery_map #Head.lane
    (kpost call refs args vw sp prompt fraction sh)
    (fun i -> readers call args vw sp (fraction /. 32.0R) **
      Head.lane_only i
        (Book.full refs.Owned.global (hidden sh) args.Entry.cache (Owned.local_owners call refs args (best sh) (next sh))
          (K.ready call (Native.packed_values vw) prompt (Model.calls call))))
    fn i { unfold kpost call refs args vw sp prompt fraction sh i; };
  Every.forevery_unzip #Head.lane
    (fun _ -> readers call args vw sp (fraction /. 32.0R))
    (fun i -> Head.lane_only i
      (Book.full refs.Owned.global (hidden sh) args.Entry.cache (Owned.local_owners call refs args (best sh) (next sh))
        (K.ready call (Native.packed_values vw) prompt (Model.calls call))));
  Every.forevery_map #Head.lane
    (fun _ -> readers call args vw sp (fraction /. 32.0R))
    (fun _ -> weight call args |-> Frac (fraction /. 32.0R) vw ** A.pts_to args.Entry.prompt #(fraction /. 32.0R) sp)
    fn _ { unfold readers call args vw sp (fraction /. 32.0R); };
  Every.forevery_unzip #Head.lane
    (fun _ -> weight call args |-> Frac (fraction /. 32.0R) vw)
    (fun _ -> A.pts_to args.Entry.prompt #(fraction /. 32.0R) sp);
  T.tensor_gather_n (weight call args) 32;
  Extra.array_gather args.Entry.prompt #sp #fraction 32;
  fold readers call args vw sp fraction;
  Head.take_lane_zero
    (Book.full refs.Owned.global (hidden sh) args.Entry.cache (Owned.local_owners call refs args (best sh) (next sh))
      (K.ready call (Native.packed_values vw) prompt (Model.calls call))) ();
  Result.detach call refs args (hidden sh) (best sh) (next sh) (Native.packed_values vw) prompt;
  shared_close sh;
  fold post call args vw sp prompt fraction;
}
