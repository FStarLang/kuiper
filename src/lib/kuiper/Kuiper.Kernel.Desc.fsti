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

  frame : slprop;

  block_pre  : natlt nblk -> slprop;
  block_post : natlt nblk -> slprop;

  setup : (
    unit ->
    stt_ghost unit emp_inames
      (requires full_pre)
      (ensures fun _ ->
        (forall+ (bid : natlt nblk). block_pre bid) **
        frame)
  );
  teardown : (
    unit ->
    stt_ghost unit emp_inames
      (requires
        (forall+ (bid : natlt nblk). block_post bid) **
        frame)
      (ensures  fun _ -> full_post)
  );

  kpre :
    gpu_array shmem_type shmem_sz ->
    natlt nblk ->
    natlt nthr ->
    slprop;
  kpost :
    gpu_array shmem_type shmem_sz ->
    natlt nblk ->
    natlt nthr ->
    slprop;

  block_frame : gpu_array shmem_type shmem_sz -> natlt nblk -> slprop;

  block_setup : (
    (ar: gpu_array shmem_type shmem_sz) ->
    (bid: natlt nblk) ->
    unit ->
    stt_ghost unit emp_inames
      (requires
        block_setup_tok nthr **
        (exists* v. gpu_pts_to_array #shmem_type #shmem_sz ar #1.0R v) **
        block_pre bid)
      (ensures fun _ ->
        block_setup_tok nthr **
        (forall+ (i : natlt nthr). kpre ar bid i) **
        block_frame ar bid)
  );

  block_teardown : (
    (ar: gpu_array shmem_type shmem_sz) ->
    (bid: natlt nblk) ->
    unit ->
    stt_ghost unit emp_inames
      (requires
        (forall+ (i : natlt nthr). kpost ar bid i) **
        block_frame ar bid)
      (ensures fun _ ->
        (exists* v. gpu_pts_to_array #shmem_type #shmem_sz ar #1.0R v) **
        block_post bid)
  );

  f : (
    ear : erased (gpu_array shmem_type shmem_sz) ->
    bid : szlt nblk ->
    tid : szlt nthr ->
    unit ->
    stt unit
      (* This seems to be missing shmem_tok ear *)
      (requires
         gpu **
         kpre ear bid tid **
         thread_id nthr tid **
         block_id nblk bid)
      (ensures fun _ ->
         gpu **
         kpost ear bid tid **
         thread_id nthr tid **
         block_id nblk bid)
  );
}
