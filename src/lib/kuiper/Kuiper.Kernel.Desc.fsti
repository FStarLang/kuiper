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
module SZ = FStar.SizeT

(* A full kernel description, in its most general form. There are
simpler version in the Kuiper.Kernel.Casts module. *)
noeq
inline_for_extraction noextract
type kernel_desc (full_pre full_post : slprop) = {
  nblk : (x : SZ.t { x <= max_blocks });
  nthr : (x : SZ.t { x <= max_threads });

  shmems_desc : list shmem_desc;

  kpre :
    c_shmems shmems_desc ->
    natlt nblk ->
    natlt nthr ->
    slprop;
  kpost :
    c_shmems shmems_desc ->
    natlt nblk ->
    natlt nthr ->
    slprop;

  f : (
    sh : c_shmems shmems_desc ->
    bid : szlt nblk ->
    tid : szlt nthr ->
    unit ->
    stt unit
      (requires
         gpu **
         kpre sh bid tid **
         thread_id nthr tid **
         block_id nblk bid)
      (ensures fun _ ->
         gpu **
         kpost sh bid tid **
         thread_id nthr tid **
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
    c_shmems shmems_desc ->
    natlt nblk -> slprop;

  block_setup : (
    (sh : c_shmems shmems_desc) ->
    (bid: natlt nblk) ->
    unit ->
    stt_ghost unit emp_inames
      (requires
        block_setup_tok nthr **
        live_c_shmems sh **
        block_pre bid)
      (ensures fun _ ->
        block_setup_tok nthr **
        (forall+ (i : natlt nthr). kpre sh bid i) **
        block_frame sh bid)
  );

  block_teardown : (
    (sh : c_shmems shmems_desc) ->
    (bid: natlt nblk) ->
    unit ->
    stt_ghost unit emp_inames
      (requires
        (forall+ (i : natlt nthr). kpost sh bid i) **
        block_frame sh bid)
      (ensures fun _ ->
        live_c_shmems sh **
        block_post bid)
  );
}
