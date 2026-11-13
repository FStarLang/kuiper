module KPlay.Proof.Optimized.Descriptor

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
open Pulse.Lib.Send
open FStar.Tactics.Typeclasses { solve }
module A = Pulse.Lib.Array
module S = FStar.Seq
module Mem = Kuiper.Array.Core
module Loc = Kuiper.Locs
module SH = Kuiper.SHMem
module B = Kuiper.Barrier
module K = Kuiper.Kernel
module Every = Kuiper.ForEvery
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module I64 = FStar.Int64
module Entry = KPlay.Proof.Optimized.Entry
module Layout = KPlay.Proof.Optimized.Layout
module Source = KPlay.Proof.Optimized.Kernel

inline_for_extraction noextract
let declarations : list SH.shmem_desc = [
  SH.SHArray bf16 1024sz; SH.SHArray bf16 1024sz;
  SH.SHArray f32 8192sz; SH.SHArray f32 2048sz
]

inline_for_extraction noextract
let shared = SH.c_shmems declarations

inline_for_extraction noextract
let hidden (sh:shared) = sh._1
inline_for_extraction noextract
let normalized (sh:shared) = sh._2._1
inline_for_extraction noextract
let projected (sh:shared) = sh._2._2._1
inline_for_extraction noextract
let activation (sh:shared) = sh._2._2._2._1

inline_for_extraction noextract
let device_arguments (call:Model.invocation) (args:Entry.arguments call) : prop =
  Mem.is_global_array args.Entry.weights /\ Mem.is_global_array args.Entry.prompt /\
  Mem.is_global_array args.Entry.states /\ Mem.is_global_array args.Entry.convolution /\
  Mem.is_global_array args.Entry.keys /\ Mem.is_global_array args.Entry.values /\
  Mem.is_global_array args.Entry.tokens /\ Mem.is_global_array args.Entry.hidden

inline_for_extraction noextract
let shared_owned (sh:shared) : slprop =
  Source.shared_cells (hidden sh) (normalized sh) (projected sh) (activation sh)

inline_for_extraction noextract
let thread_pre (call:Model.invocation) (args:Entry.arguments call) (w:Layout.packed)
  (physical:lseq I64.t call.prompt_length) (prompt:D.vec (D.token 248320) call.prompt_length)
  (permission:perm) (sh:shared) : slprop =
  Source.pre call args w physical prompt permission ** shared_owned sh

inline_for_extraction noextract
let thread_post (call:Model.invocation) (args:Entry.arguments call) (w:Layout.packed)
  (physical:lseq I64.t call.prompt_length) (prompt:D.vec (D.token 248320) call.prompt_length)
  (permission:perm) (sh:shared) : slprop =
  Source.post call args w physical prompt permission ** shared_owned sh

ghost
fn shared_open (sh:shared)
  requires SH.live_c_shmems sh
  ensures shared_owned sh
{
  SH.unfold_live_c_shmems_cons sh #1.0R;
  SH.unfold_live_c_shmems_cons sh._2 #1.0R;
  SH.unfold_live_c_shmems_cons sh._2._2 #1.0R;
  SH.unfold_live_c_shmems_cons sh._2._2._2 #1.0R;
  SH.unfold_live_c_shmems_nil sh._2._2._2._2 #1.0R;
  SH.unfold_live_c_shmem #(SH.SHArray bf16 1024sz) (hidden sh) #1.0R;
  SH.unfold_live_c_shmem #(SH.SHArray bf16 1024sz) (normalized sh) #1.0R;
  SH.unfold_live_c_shmem #(SH.SHArray f32 8192sz) (projected sh) #1.0R;
  SH.unfold_live_c_shmem #(SH.SHArray f32 2048sz) (activation sh) #1.0R;
  with h n p a. assert hidden sh |-> h ** normalized sh |-> n ** projected sh |-> p ** activation sh |-> a;
  A.pts_to_len (hidden sh);
  A.pts_to_len (normalized sh);
  A.pts_to_len (projected sh);
  A.pts_to_len (activation sh);
  fold Source.shared_cells (hidden sh) (normalized sh) (projected sh) (activation sh);
  fold shared_owned sh;
}

ghost
fn shared_close (sh:shared)
  requires shared_owned sh
  ensures SH.live_c_shmems sh
{
  unfold shared_owned sh;
  unfold Source.shared_cells (hidden sh) (normalized sh) (projected sh) (activation sh);
  SH.fold_live_c_shmem #(SH.SHArray bf16 1024sz) (hidden sh) #1.0R;
  SH.fold_live_c_shmem #(SH.SHArray bf16 1024sz) (normalized sh) #1.0R;
  SH.fold_live_c_shmem #(SH.SHArray f32 8192sz) (projected sh) #1.0R;
  SH.fold_live_c_shmem #(SH.SHArray f32 2048sz) (activation sh) #1.0R;
  SH.fold_live_c_shmems_nil sh._2._2._2._2 #1.0R;
  SH.fold_live_c_shmems_cons sh._2._2._2 #1.0R;
  SH.fold_live_c_shmems_cons sh._2._2 #1.0R;
  SH.fold_live_c_shmems_cons sh._2 #1.0R;
  SH.fold_live_c_shmems_cons sh #1.0R;
}

ghost
fn singleton_intro (p:slprop) (frame:slprop{frame == emp}) ()
  requires p
  ensures Every.forevery (Kuiper.Common.natlt 1) (fun _ -> p) ** frame
{
  Every.forevery_singleton_intro' #(Kuiper.Common.natlt 1) (fun _ -> p) 0;
  rewrite emp as frame;
}

ghost
fn singleton_elim (p:slprop) (frame:slprop{frame == emp}) ()
  requires Every.forevery (Kuiper.Common.natlt 1) (fun _ -> p) ** frame
  ensures p
{
  rewrite frame as emp;
  Every.forevery_singleton_elim' #(Kuiper.Common.natlt 1) (fun _ -> p) 0;
}

ghost
fn block_setup (p:slprop) (frame:slprop{frame == emp}) (sh:shared) (_:Kuiper.Common.natlt 1) ()
  requires SH.live_c_shmems sh ** p
  ensures Every.forevery (Kuiper.Common.natlt 1) (fun _ -> p ** shared_owned sh) ** frame
{
  shared_open sh;
  singleton_intro (p ** shared_owned sh) frame ();
}

ghost
fn block_teardown (p:slprop) (frame:slprop{frame == emp}) (sh:shared) (_:Kuiper.Common.natlt 1) ()
  requires Every.forevery (Kuiper.Common.natlt 1) (fun _ -> p ** shared_owned sh) ** frame
  ensures SH.live_c_shmems sh ** p
{
  singleton_elim (p ** shared_owned sh) frame ();
  shared_close sh;
}

inline_for_extraction noextract
instance global_array (#a:Type0) (array:A.array a{Mem.is_global_array array})
  (#fraction:perm) (values:S.seq a) : is_send_across Loc.gpu_of (A.pts_to array #fraction values) =
  Mem.is_send_pts_to array #fraction values

inline_for_extraction noextract
instance global_array_block (#a:Type0) (array:A.array a{Mem.is_global_array array})
  (#fraction:perm) (values:S.seq a) : is_send_across Loc.block_of (A.pts_to array #fraction values) =
  Loc.send_across_if_send_across_gpu (A.pts_to array #fraction values) (global_array array #fraction values)

inline_for_extraction noextract
instance global_mask_block (#a:Type0) (array:A.array a{Mem.is_global_array array})
  (#fraction:perm) (values:S.seq (option a)) (mask:nat -> prop)
  : is_send_across Loc.block_of (A.pts_to_mask array #fraction values mask) =
  Loc.send_across_if_send_across_gpu (A.pts_to_mask array #fraction values mask)
    (Mem.is_send_across_global_array_mask array #fraction values mask)

let shared_locations (sh:shared)
  : Lemma
      (requires SH.c_shmems_inv sh)
      (ensures SH.is_block_array (hidden sh) /\ SH.is_block_array (normalized sh) /\
        SH.is_block_array (projected sh) /\ SH.is_block_array (activation sh)) =
  FStar.Pervasives.norm_spec [delta_only [
    `%SH.c_shmems_inv; `%SH.c_shmems_block_inv; `%SH.c_shmem_inv; `%declarations]; zeta; iota]
    (SH.c_shmems_inv sh)

inline_for_extraction noextract
let join_block (#p #q:slprop)
  (left:is_send_across Loc.block_of p) (right:is_send_across Loc.block_of q)
  : is_send_across Loc.block_of (p ** q) =
  is_send_across_star p q #left #right

inline_for_extraction noextract
let quantify_block (#a:Type0) (#p:a -> slprop)
  (move:(x:a -> is_send_across Loc.block_of (p x)))
  : is_send_across Loc.block_of (exists* (x:a). p x) =
  is_send_across_exists p #move

inline_for_extraction noextract
let send_shared (sh:shared) (proof:squash (SH.c_shmems_inv sh))
  : is_send_across Loc.block_of (shared_owned sh) =
  shared_locations sh;
  let h = fun (v:lseq bf16 1024) -> SH.is_send_across_block_array (hidden sh) #1.0R #v in
  let n = fun (v:lseq bf16 1024) -> SH.is_send_across_block_array (normalized sh) #1.0R #v in
  let p = fun (v:lseq f32 8192) -> SH.is_send_across_block_array (projected sh) #1.0R #v in
  let a = fun (v:lseq f32 2048) -> SH.is_send_across_block_array (activation sh) #1.0R #v in
  join_block (quantify_block h)
    (join_block (quantify_block n) (join_block (quantify_block p) (quantify_block a)))

inline_for_extraction noextract
fn thread (call:Model.invocation) (args:Entry.arguments call)
  (#w:erased Layout.packed) (#physical:erased (lseq I64.t call.prompt_length))
  (#prompt:erased (D.vec (D.token 248320) call.prompt_length)) (#permission:perm)
  (sh:shared) (bid:sz{bid < 1}) (tid:sz{tid < 1}) ()
  requires gpu ** pure (SH.c_shmems_inv sh) **
    thread_pre call args w physical prompt permission sh **
    thread_id 1 tid ** block_id 1 bid ** B.barrier_tok (B.empty_contract 1) ** B.barrier_state 0
  ensures gpu ** thread_post call args w physical prompt permission sh **
    thread_id 1 tid ** block_id 1 bid ** B.barrier_tok (B.empty_contract 1) ** B.barrier_state 0
{
  unfold thread_pre call args w physical prompt permission sh;
  unfold shared_owned sh;
  Source.source call args (hidden sh) (normalized sh) (projected sh) (activation sh)
    #w #physical #prompt #permission;
  fold shared_owned sh;
  fold thread_post call args w physical prompt permission sh;
}

inline_for_extraction noextract
let descriptor (call:Model.invocation) (args:Entry.arguments call{device_arguments call args})
  (#w:erased Layout.packed) (#physical:erased (lseq I64.t call.prompt_length))
  (#prompt:erased (D.vec (D.token 248320) call.prompt_length)) (#permission:perm)
  : K.kernel_desc (Source.pre call args w physical prompt permission)
      (Source.post call args w physical prompt permission) = {
  K.nblk = 1sz;
  K.nthr = 1sz;
  K.shmems_desc = declarations;
  K.kpre = (fun sh _ _ -> thread_pre call args w physical prompt permission sh);
  K.kpost = (fun sh _ _ -> thread_post call args w physical prompt permission sh);
  K.barrier_contract = (fun _ _ -> B.empty_contract 1);
  K.barrier_count = (fun _ -> 0);
  K.barrier_ok = (fun _ _ -> B.empty_barrier_transform 1);
  K.f = thread call args #w #physical #prompt #permission;
  K.frame = emp;
  K.block_pre = (fun _ -> Source.pre call args w physical prompt permission);
  K.block_post = (fun _ -> Source.post call args w physical prompt permission);
  K.setup = singleton_intro (Source.pre call args w physical prompt permission) emp;
  K.teardown = singleton_elim (Source.post call args w physical prompt permission) emp;
  K.block_frame = (fun _ _ -> emp);
  K.block_setup = block_setup (Source.pre call args w physical prompt permission) emp;
  K.block_teardown = block_teardown (Source.post call args w physical prompt permission) emp;
  K.block_pre_sendable = (fun _ -> solve);
  K.block_post_sendable = (fun _ -> solve);
  K.kpre_sendable = (fun sh proof _ _ ->
    let inputs : is_send_across Loc.block_of (Source.pre call args w physical prompt permission) = solve in
    join_block inputs (send_shared sh proof));
  K.kpost_sendable = (fun sh proof _ _ ->
    let outputs : is_send_across Loc.block_of (Source.post call args w physical prompt permission) = solve in
    join_block outputs (send_shared sh proof));
}
