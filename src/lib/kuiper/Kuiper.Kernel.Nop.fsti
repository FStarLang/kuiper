module Kuiper.Kernel.Nop
#lang-pulse

open Kuiper.Common
open Pulse.Lib.Core
open FStar.Ghost
open Kuiper.IntAliases
open Kuiper.Array
open Kuiper.Base
module SZ = FStar.SizeT
open Kuiper.ForEvery

open Kuiper.Kernel.Desc

ghost
fn nop_block_setup (#nblk #nthr : nat) (ar : gpu_array u32 0sz) (bid : natlt nblk)
  requires
    block_setup nthr ** (exists* v. gpu_pts_to_array ar v)
  ensures
    block_setup nthr ** (forall+ (i : natlt nthr). emp)
{
  drop_ (exists* v. gpu_pts_to_array ar v);
  assume (forall+ (i : natlt nthr). emp); (* trivial to prove *)
}

ghost
fn ghost_setup (#f:slprop) ()
  requires f
  ensures forall+ (ebid:natlt 1) (etid:natlt 1). f
{
  forevery_singleton_intro #(natlt 1) _;
  forevery_singleton_intro #(natlt 1) _;
}

ghost
fn ghost_teardown (#f:slprop) ()
  requires forall+ (ebid:natlt 1) (etid:natlt 1). f
  ensures f
{
  forevery_singleton_elim #(natlt 1) _;
  forevery_singleton_elim #(natlt 1) _;
}


inline_for_extraction noextract
fn kernel_nop (eshmem : erased (gpu_array u32 0sz)) (ebid : enatlt 1sz) (etid : enatlt 1sz)
  requires
    gpu **
    emp ** (* kpre *)
    thread_id 1sz etid **
    block_id 1sz ebid **
    shmem_tok eshmem **
    emp (* block pre *)
  ensures
    gpu **
    emp ** (* kpost *)
    thread_id 1sz etid **
    block_id 1sz ebid **
    shmem_tok eshmem **
    emp (* block post *)
{
  ()
}

inline_for_extraction noextract
let nop_desc : kernel_desc emp emp = {
  nblk = 1sz;
  nthr = 1sz;

  shmem_type = u32;
  shmem_type_is_sized = solve;
  shmem_sz = 0sz;

  block_pre  = (fun _ _ _ -> emp);
  block_post = (fun _ _ _ -> emp);
  block_setup = nop_block_setup;

  kpre = (fun _ _ -> emp);
  kpost = (fun _ _ -> emp);
  f = kernel_nop;

  setup = ghost_setup;
  teardown = ghost_teardown;
}

inline_for_extraction noextract
fn kf_no_shmem
  (#nblk : (x : SZ.t { 0 < x /\ x <= max_blocks }))
  (#nthr : (x : SZ.t { 0 < x /\ x <= max_threads }))
  (#shmem_type : Type0)
  {| shmem_type_is_sized : Kuiper.Sized.sized shmem_type |}
  (#shmem_sz : SZ.t)
  (#kpre  : natlt nblk -> natlt nthr -> slprop)
  (#kpost : natlt nblk -> natlt nthr -> slprop)
  (kf : (
    ebid : enatlt nblk ->
    etid : enatlt nthr ->
    stt unit
      (requires
         gpu **
         kpre ebid etid **
         thread_id nthr etid **
         block_id nblk ebid
      )
      (ensures fun _ ->
         gpu **
         kpost ebid etid **
         thread_id nthr etid **
         block_id nblk ebid)
  ))
  (eshmem : erased (gpu_array shmem_type shmem_sz))
  (ebid : enatlt nblk)
  (etid : enatlt nthr)
  requires
    gpu **
    kpre ebid etid **
    thread_id nthr etid **
    block_id nblk ebid **
    shmem_tok eshmem **
    emp
  ensures
    gpu **
    kpost ebid etid **
    thread_id nthr etid **
    block_id nblk ebid **
    shmem_tok eshmem **
    emp
{
  kf ebid etid
}
