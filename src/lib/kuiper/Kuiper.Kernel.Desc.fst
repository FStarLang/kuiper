module Kuiper.Kernel.Desc
#lang-pulse

open FStar.Ghost
open Pulse.Lib.Core
open Kuiper.Common
open Kuiper.ForEvery
open Kuiper.Array
open Kuiper.Base
open Kuiper.SizeT
open Kuiper.SHMem
module SZ = Kuiper.SizeT

(* A full kernel description, in its most general form. There are
simpler version in the Kuiper.Kernel.Casts module. *)
noeq
inline_for_extraction noextract
type kernel_desc (full_pre full_post : slprop) = {
  nblk : (x : SZ.t { x <= max_blocks });
  nthr : (x : SZ.t { x <= max_threads });

  shmems_desc : list shmem_desc;

  kpre :
    bid:natlt nblk ->
    natlt nthr ->
    c_shmems shmems_desc #0 bid -> //why do have I to put #0 here?
    slprop;

  kpost :
    bid:natlt nblk ->
    natlt nthr ->
    c_shmems shmems_desc bid ->
    slprop;

  f : (
    bid : szlt nblk ->
    tid : szlt nthr ->
    sh : c_shmems shmems_desc bid ->
    unit ->
    stt unit
      (requires
         gpu **
         kpre bid tid sh **
         thread_id nthr bid tid **
         block_id nblk bid)
      (ensures fun _ ->
         gpu **
         kpost bid tid sh **
         thread_id nthr bid tid **
         block_id nblk bid)
  );

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
  block_frame :
    bid:natlt nblk -> 
    c_shmems shmems_desc bid ->
    slprop;

  block_setup : (
    (bid: natlt nblk) ->
    (sh : c_shmems shmems_desc bid) ->
    unit ->
    stt_ghost unit emp_inames
      (requires
        can_create_barrier nthr **
        live_c_shmems sh **
        block_pre bid)
      (ensures fun _ ->
        consumed_can_create_barrier **
        (forall+ (i : natlt nthr). kpre bid i sh) **
        block_frame bid sh)
  );

  block_teardown : (
    (bid: natlt nblk) ->
    (sh : c_shmems shmems_desc bid) ->
    unit ->
    stt_ghost unit emp_inames
      (requires
        (forall+ (i : natlt nthr). kpost bid i sh) **
        block_frame bid sh)
      (ensures fun _ ->
        live_c_shmems sh **
        block_post bid)
  );
}
