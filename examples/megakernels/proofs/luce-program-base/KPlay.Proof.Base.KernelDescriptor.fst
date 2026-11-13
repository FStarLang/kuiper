module KPlay.Proof.Base.KernelDescriptor

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor
open Kuiper.ForEvery
open FStar.Ghost

module C = Kuiper.Chest
module BF = Kuiper.BFloat16
module I64 = FStar.Int64
module SZ = Kuiper.SizeT
module SH = Kuiper.SHMem
module B = Kuiper.Barrier
module Kernel = Kuiper.Kernel
module Every = Kuiper.ForEvery
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module Entry = KPlay.Proof.Base.Entry
module Owned = KPlay.Proof.Base.KernelOwnership
module Layout = KPlay.Proof.Base.KernelSetup
module Move = KPlay.Proof.Base.KernelSend
module Native = KPlay.Proof.Base.KernelNative

ghost
fn setup (p:slprop) (frame:slprop{frame == emp}) ()
  requires p
  ensures (forall+ (_:Kuiper.Common.natlt 1). p) ** frame
{
  Every.forevery_singleton_intro' #(Kuiper.Common.natlt 1) (fun _ -> p) 0;
  rewrite emp as frame;
}

ghost
fn teardown (p:slprop) (frame:slprop{frame == emp}) ()
  requires (forall+ (_:Kuiper.Common.natlt 1). p) ** frame
  ensures p
{
  rewrite frame as emp;
  Every.forevery_singleton_elim' #(Kuiper.Common.natlt 1) (fun _ -> p) 0;
}

inline_for_extraction noextract
fn thread
  (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (#vw:C.chest2 BF.t 752393024 1) (#sp:erased (lseq I64.t call.Model.prompt_length))
  (#prompt:erased (D.vec (D.token 248320) call.Model.prompt_length)) (#fraction:perm)
  (sh:Layout.shared) (bid:sz{SZ.v bid < 1}) (tid:sz{SZ.v tid < 32}) ()
  requires gpu ** pure (SH.c_shmems_inv sh) ** Layout.kpre call refs args vw sp prompt fraction sh (SZ.v tid) **
    thread_id 32 (SZ.v tid) ** block_id 1 (SZ.v bid) ** B.barrier_tok (B.empty_contract 32) ** B.barrier_state 0
  ensures gpu ** Layout.kpost call refs args vw sp prompt fraction sh (SZ.v tid) **
    thread_id 32 (SZ.v tid) ** block_id 1 (SZ.v bid) ** B.barrier_tok (B.empty_contract 32) ** B.barrier_state 0
{
  unfold Layout.kpre call refs args vw sp prompt fraction sh (SZ.v tid);
  unfold Layout.readers call args vw sp (fraction /. 32.0R);
  Native.apply call refs args (Layout.weight call args) (Layout.hidden sh) (Layout.best sh) (Layout.next sh) tid
    #prompt #sp #vw #(fraction /. 32.0R) #(fraction /. 32.0R);
  fold Layout.readers call args vw sp (fraction /. 32.0R);
  fold Layout.kpost call refs args vw sp prompt fraction sh (SZ.v tid);
}

inline_for_extraction noextract
let descriptor
  (call:Model.invocation) (refs:Owned.controls call)
  (args:Entry.arguments call{Layout.device_arguments call args})
  (#vw:C.chest2 BF.t 752393024 1) (#sp:erased (lseq I64.t call.Model.prompt_length))
  (#prompt:erased (D.vec (D.token 248320) call.Model.prompt_length)) (#fraction:perm)
  : Kernel.kernel_desc
      (Layout.pre call refs args vw sp prompt fraction)
      (Layout.post call args vw sp prompt fraction) = {
  Kernel.nblk = 1sz;
  Kernel.nthr = 32sz;
  Kernel.shmems_desc = Layout.declarations;
  Kernel.kpre = (fun sh _ i -> Layout.kpre call refs args vw sp prompt fraction sh i);
  Kernel.kpost = (fun sh _ i -> Layout.kpost call refs args vw sp prompt fraction sh i);
  Kernel.barrier_contract = (fun _ _ -> B.empty_contract 32);
  Kernel.barrier_count = (fun _ -> 0);
  Kernel.barrier_ok = (fun _ _ -> B.empty_barrier_transform 32);
  Kernel.f = thread call refs args #vw #sp #prompt #fraction;
  Kernel.frame = emp;
  Kernel.block_pre = (fun _ -> Layout.pre call refs args vw sp prompt fraction);
  Kernel.block_post = (fun _ -> Layout.post call args vw sp prompt fraction);
  Kernel.setup = setup (Layout.pre call refs args vw sp prompt fraction) emp;
  Kernel.teardown = teardown (Layout.post call args vw sp prompt fraction) emp;
  Kernel.block_frame = (fun _ _ -> emp);
  Kernel.block_setup = Layout.block_setup call refs args vw sp prompt fraction emp;
  Kernel.block_teardown = Layout.block_teardown call refs args vw sp prompt fraction emp;
  Kernel.block_pre_sendable = (fun _ -> Move.pre call refs args vw sp prompt fraction);
  Kernel.block_post_sendable = (fun _ -> Move.post call args vw sp prompt fraction);
  Kernel.kpre_sendable = (fun sh proof _ i -> Move.kpre call refs args vw sp prompt fraction sh #proof i);
  Kernel.kpost_sendable = (fun sh proof _ i -> Move.kpost call refs args vw sp prompt fraction sh #proof i);
}
