module Kuiper.Kernel
#lang-pulse

open Kuiper.Common
open Pulse.Lib.Core
open FStar.Ghost
open Kuiper.ForEvery
open Kuiper.IntAliases
open Kuiper.SizeT
open Kuiper.Array
open Kuiper.Base
open Kuiper.Barrier.RPM
open Kuiper.Epoch
open FStar.Mul
module SZ = FStar.SizeT
open Pulse.Lib.Pledge
include Kuiper.Kernel.Base

(* Helpers below *)

(* f<<<nblk, nthr, smem_sz>>>(...); *)
inline_for_extraction noextract
fn launch_kernel_n_m_shmem
  (nblk : szp { nblk <= max_blocks })
  (nthr : szp { nthr <= max_threads })
  (#pre #post : natlt nblk -> natlt nthr -> slprop)
  (a : Type u#0) {| Kuiper.Sized.sized a |}
  (smem_sz : SZ.t)
  (#shared_pre #shared_post : gpu_array a smem_sz -> natlt nblk -> natlt nthr -> slprop)
  (setup : (ar: gpu_array a smem_sz) -> (bid: natlt nblk) ->
    stt_ghost unit emp_inames
      (block_setup nthr ** (exists* v. gpu_pts_to_array #a #smem_sz ar #1.0R v))
      (fun _ -> block_setup nthr ** (forall+ (i : natlt nthr). shared_pre ar bid i)))
(k :
    (ar: erased (gpu_array a smem_sz)) ->
    (ebid : enatlt nblk) ->
    (etid : enatlt nthr) ->
    stt unit
      (         gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                shmem_tok ar **
                shared_pre ar ebid etid **
                pre ebid etid)
      (fun _ -> gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                // shmem_tok ar **
                shared_post ar ebid etid **
                post ebid etid)
  )
  requires
    cpu **
    (forall+ (b : natlt nblk) (t : natlt nthr). pre b t)
  ensures
    cpu **
    (forall+ (b : natlt nblk) (t : natlt nthr). post b t)

inline_for_extraction noextract
fn launch_kernel_n_m_barrier_async
  (nblk : szp { nblk <= max_blocks })
  (nthr : szp { nthr <= max_threads })
  (#pre #post : natlt nblk -> natlt nthr -> slprop)
  (#p : rpm_t nthr)
  (k :
    (ebid : enatlt nblk) ->
    (etid : enatlt nthr) ->
    stt unit
      (         gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                mbarrier_tok nthr p 0 etid **
                pre ebid etid)
      (fun _ -> gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                (exists* it. mbarrier_tok nthr p it etid) **
                post ebid etid)
  )
  (#e : epoch_t)
  requires
    cpu **
    epoch_live e **
    (forall+ (b : natlt nblk) (t : natlt nthr). pre b t)
  returns
    e' : epoch_t
  ensures
    cpu **
    epoch_live e' **
    pledge0 (epoch_done e')
      (forall+ (b : natlt nblk) (t : natlt nthr). post b t) **
    pure (e' >= e)

inline_for_extraction noextract
fn launch_kernel_n_m_barrier
  (nblk : szp { nblk <= max_blocks })
  (nthr : szp { nthr <= max_threads })
  (#pre #post : natlt nblk -> natlt nthr -> slprop)
  (#p : rpm_t nthr)
  (k :
    (ebid : enatlt nblk) ->
    (etid : enatlt nthr) ->
    stt unit
      (         gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                mbarrier_tok nthr p 0 etid **
                pre ebid etid)
      (fun _ -> gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                (exists* it. mbarrier_tok nthr p it etid) **
                post ebid etid)
  )
  requires
    cpu **
    (forall+ (b : natlt nblk) (t : natlt nthr). pre b t)
  ensures
    cpu **
    (forall+ (b : natlt nblk) (t : natlt nthr). post b t)

(* f<<<nblk, nthr>>>(...); *)
inline_for_extraction noextract
fn launch_kernel_n_m
  (nblk : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (nthr : SZ.t { 0 < nthr /\ nthr <= max_threads })
  (#pre #post : natlt nblk -> natlt nthr -> slprop)
  (k :
    (ebid : enatlt nblk) ->
    (etid : enatlt nthr) ->
    stt unit
      (         gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                pre ebid etid)
      (fun _ -> gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                post ebid etid)
  )
  requires
    cpu **
    (forall+ (b : natlt nblk) (t : natlt nthr). pre b t)
  ensures
    cpu **
    (forall+ (b : natlt nblk) (t : natlt nthr). post b t)

inline_for_extraction noextract
fn launch_kernel_n_blocks_async
  (nblk  : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (#pre #post : natlt nblk -> slprop)
  (k :
    (ebid : enatlt nblk ->
    stt unit
      (         gpu **
                block_id nblk ebid **
                pre ebid)
      (fun _ -> gpu **
                block_id nblk ebid **
                post ebid)
  ))
  (#e : epoch_t)
  requires
    cpu **
    epoch_live e **
    (forall+ (b : natlt nblk). pre b)
  returns
    e' : epoch_t
  ensures
    cpu **
    epoch_live e' **
    pledge0 (epoch_done e')
      (forall+ (b : natlt nblk). post b) **
    pure (e' >= e)

inline_for_extraction noextract
fn launch_kernel_n_blocks
  (nblk : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (#pre #post : (natlt nblk -> slprop))
  (k :
    (ebid : enatlt nblk ->
    stt unit
      (         gpu **
                block_id nblk ebid **
                pre ebid)
      (fun _ -> gpu **
                block_id nblk ebid **
                post ebid)
  ))
  requires
    cpu **
    (forall+ (b : natlt nblk). pre b)
  ensures
    cpu **
    (forall+ (b : natlt nblk). post b)

inline_for_extraction noextract
fn launch_kernel_1_async
  (#pre #post : slprop)
  (k : unit ->
    stt unit (gpu ** pre) (fun _ -> gpu ** post)
  )
  (#e : epoch_t)
  requires
    cpu **
    epoch_live e **
    pre
  returns
    e' : epoch_t
  ensures
    cpu **
    epoch_live e' **
    pledge0 (epoch_done e') post **
    pure (e' >= e)

inline_for_extraction
fn launch_kernel_1
  (#pre #post : slprop)
  (k : unit ->
    stt unit (gpu ** pre) (fun _ -> gpu ** post)
  )
  requires cpu ** pre
  ensures  cpu ** post

// inline_for_extraction noextract
// fn thread_idx_all () (#n: tid_t)
//   preserves
//     thread_id n
//   requires
//     emp
//   returns
//     id : SZ.t
//   ensures
//     pure (SZ.v id == thread_index n /\ SZ.v id < max_blocks * max_threads)
