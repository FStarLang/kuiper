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

(* Description of one shared memory array "request" *)
noeq
inline_for_extraction
type shmem_desc =
  | SHArray :
    (ty : Type0) ->
    {| sized : Kuiper.Sized.sized ty |} ->
    len    : SZ.t ->
    shmem_desc

inline_for_extraction unfold
let c_shmem (d : shmem_desc) : Type0 =
  match d with
  | SHArray ty len -> gpu_array ty len

inline_for_extraction
let rec c_shmems (d : list shmem_desc) : Type0 =
  match d with
  | [] -> int
  | d :: ds ->
    c_shmem d & c_shmems ds

let live_c_shmem #d (c : c_shmem d) : slprop =
  exists* v. gpu_pts_to_array #d.ty #d.len c #1.0R v

let rec live_c_shmems #ds (c : c_shmems ds) : slprop =
  match ds with
  | [] -> emp
  | d :: ds ->
    let c : c_shmem d & c_shmems ds = c in (* coerce *)
    live_c_shmem #d (fst c) ** live_c_shmems #ds (snd c)

noeq
inline_for_extraction noextract
type kernel_desc (full_pre : slprop) (full_post : slprop) = {
  nblk : (x : SZ.t { 0 < x /\ x <= max_blocks });
  nthr : (x : SZ.t { 0 < x /\ x <= max_threads });

  shmems_desc : list shmem_desc;

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
    c_shmems shmems_desc ->
    natlt nblk ->
    natlt nthr ->
    slprop;
  kpost :
    c_shmems shmems_desc ->
    natlt nblk ->
    natlt nthr ->
    slprop;

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
}
