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
    natlt nblk -> 
    slprop;

  block_setup : (
    (sh : c_shmems shmems_desc) ->
    (bid: natlt nblk) ->
    unit ->
    stt_ghost unit emp_inames
      (requires
        can_create_barrier nthr **
        live_c_shmems sh **
        block_pre bid)
      (ensures fun _ ->
        consumed_can_create_barrier **
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

  (* 
    setup: consumes full_post and produces on gpu_loc (forall+ i. block_pre i))
     
    Then, at location (block_id_loc i), 
    we consume (block_pre i) (which can be send to block_id_loc i)
    and run block_setup to produce
    on (block_id_loc i) (forall+ j. kpre i j)
     
    Finally, we run a thread at (thread_id_loc i j) 
    and (kpre i j) to (thread_id_loc i j) and obtain
    (kpost i j) at (thread_id_loc i j)

    We can send (kpost i j) to block (block_id_loc i)
    and use block_teardown to (block_post i)
    and then send that to gpu_loc to run teardown
    and obtain full_post
  *)
  block_pre_sendable: (i:natlt nblk -> is_send_across gpu_of (block_pre i));

  block_post_sendable: (i:natlt nblk -> is_send_across gpu_of (block_post i));
  
  kpre_sendable: (sh:c_shmems shmems_desc ->
                  _:squash (c_shmems_inv sh) ->
                 i:natlt nblk -> j:natlt nthr -> is_send_across block_of (kpre sh i j));
  
  kpost_sendable: (sh:c_shmems shmems_desc ->
                  _:squash (c_shmems_inv sh) -> i:natlt nblk -> j:natlt nthr -> is_send_across block_of (kpost sh i j));  
}
