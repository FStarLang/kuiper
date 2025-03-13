module Kuiper.Kernel.Desc
#lang-pulse

open FStar.Ghost
open Pulse.Lib.Core
open Kuiper.Common
open Kuiper.ForEvery
open Kuiper.Array
open Kuiper.Base
open Kuiper.SizeT
module SZ = FStar.SizeT

val shmem_tok
  (#a:Type u#0)
  {| Kuiper.Sized.sized a |}
  (#sz:nat)
  (ar:gpu_array a sz)
: slprop

noeq
inline_for_extraction noextract
type kernel_desc (full_pre : slprop) (full_post : slprop) = {
  nblk : (x : SZ.t { 0 < x /\ x <= max_blocks });
  nthr : (x : SZ.t { 0 < x /\ x <= max_threads });

  shmem_type : Type u#0;
  shmem_type_is_sized : Kuiper.Sized.sized shmem_type;
  shmem_sz : sz;

  (* This is used to split up the array for the threads in the block.
     It should definitely be generalized to be more uniform with
     other levels of the hierarchy, e.g. by having a teardown. Currently
     we just drop the block_post. *)
  block_pre  : gpu_array shmem_type shmem_sz -> natlt nblk -> natlt nthr -> slprop;
  block_post : gpu_array shmem_type shmem_sz -> natlt nblk -> natlt nthr -> slprop;
  block_setup : (
    (ar: gpu_array shmem_type shmem_sz) ->
    (bid: natlt nblk) ->
    stt_ghost unit emp_inames
      (requires
        block_setup nthr **
        (exists* v. gpu_pts_to_array #shmem_type #shmem_sz ar #1.0R v))
      (ensures fun _ ->
        block_setup nthr **
        (forall+ (i : natlt nthr). block_pre ar bid i))
  );

  kpre : natlt nblk -> natlt nthr -> slprop;
  kpost : natlt nblk -> natlt nthr -> slprop;

  f : (
    eshmem : erased (gpu_array shmem_type shmem_sz) ->
    ebid : enatlt nblk ->
    etid : enatlt nthr ->
    stt unit
      (requires
         gpu **
         kpre ebid etid **
         thread_id nthr etid **
         block_id nblk ebid **
         shmem_tok eshmem **
         block_pre eshmem ebid etid
      )
      (ensures fun _ ->
         gpu **
         kpost ebid etid **
         thread_id nthr etid **
         block_id nblk ebid **
         shmem_tok eshmem **
         block_post eshmem ebid etid)
  );
  setup : (
    unit ->
    stt_ghost unit emp_inames
      (requires full_pre)
      (ensures  fun _ -> forall+ (bid : natlt nblk) (tid : natlt nthr). kpre bid tid)
  );
  teardown : (
    unit ->
    stt_ghost unit emp_inames
      (requires forall+ (bid : natlt nblk) (tid : natlt nthr). kpost bid tid)
      (ensures  fun _ -> full_post)
  );
}
