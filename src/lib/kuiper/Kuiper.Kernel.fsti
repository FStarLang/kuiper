module Kuiper.Kernel
#lang-pulse

open Pulse.Lib.Core
open FStar.Ghost
open Pulse.Lib.BigStar
open Kuiper.SizeT
open Kuiper.Array
open Kuiper.Base
open Kuiper.Barrier.RPM
open Kuiper.Epoch
open FStar.Mul
module SZ = FStar.SizeT
open Pulse.Lib.Pledge

val shmem_tok
  (#a:Type u#0)
  {| Kuiper.Sized.sized a |}
  (#sz:nat)
  (ar:gpu_array a sz)
: slprop

fn obtain_shmem
  (#a:Type u#0)
  {| Kuiper.Sized.sized a |}
  (#sz : erased nat)
  (ear : erased (gpu_array a sz))
  requires shmem_tok ear
  returns  ar : gpu_array a sz
  ensures  pure (reveal ear == ar)

fn sync () (#e:erased nat)
  requires epoch_live e
  ensures
    exists* e'.
      epoch_live e' ** epoch_done e **
      pure (e' >= e)

fn launch_kernel_n_m_shmem_async
  (#u1: erased int)
  (nblk : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (nthr : SZ.t { 0 < nthr /\ nthr <= max_threads })
  (#pre #post : (tid:nat{ 0 <= tid /\ tid < (nblk * nthr) } -> slprop))
  (a : Type u#0)
  {| Kuiper.Sized.sized a |}
  (smem_sz : SZ.t)
  (#shared_pre : (ar: gpu_array a smem_sz) -> (bid: nat { 0 <= bid /\ bid < nblk }) -> (tid: nat { 0 <= tid /\ tid < nthr } -> slprop))
  (#shared_post : (ar: gpu_array a smem_sz) -> (bid: nat { 0 <= bid /\ bid < nblk }) -> (tid: nat { 0 <= tid /\ tid < nthr } -> slprop))
  (setup : (ar: gpu_array a smem_sz) -> (bid: SZ.t { 0 <= bid /\ bid < nblk }) ->
    stt_ghost unit emp_inames
      (block_setup nthr ** (exists* v. gpu_pts_to_array #a #smem_sz ar #1.0R v))
      (fun _ -> block_setup nthr ** bigstar 0 nthr (shared_pre ar bid)))

  (k :
    (ar: erased (gpu_array a smem_sz)) -> (etid: tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr }) ->
    stt unit (         gpu ** thread_id etid ** shmem_tok ar ** shared_pre ar (bidx_x etid) (tidx_x etid) ** pre (thread_index etid))
             (fun _ -> gpu ** thread_id etid **                 shared_post ar (bidx_x etid) (tidx_x etid) ** post (thread_index etid))
  )
  (#e : erased nat)
  requires
    cpu **
    epoch_live e **
    bigstar #u1 0 (nblk * nthr) pre
  ensures
    exists* e'.
      cpu **
      epoch_live e' **
      pledge0 (epoch_done e') (bigstar #u1 0 (nblk * nthr) post) **
      pure (e' >= e)

(* f<<<nblk, nthr, smem_sz>>>(...); *)
inline_for_extraction
fn launch_kernel_n_m_shmem
  (#u1: erased int)
  (nblk : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (nthr : SZ.t { 0 < nthr /\ nthr <= max_threads })
  (#pre #post : (tid:nat{ 0 <= tid /\ tid < (nblk * nthr) } -> slprop))

  (a : Type u#0)
  {| Kuiper.Sized.sized a |}
  (smem_sz : SZ.t)
  (#shared_pre : (ar: gpu_array a smem_sz) -> (bid: nat { 0 <= bid /\ bid < nblk }) -> (tid: nat { 0 <= tid /\ tid < nthr } -> slprop))
  (#shared_post : (ar: gpu_array a smem_sz) -> (bid: nat { 0 <= bid /\ bid < nblk }) -> (tid: nat { 0 <= tid /\ tid < nthr } -> slprop))
  (setup : (ar: gpu_array a smem_sz) -> (bid: SZ.t { 0 <= bid /\ bid < nblk }) ->
    stt_ghost unit emp_inames
      (block_setup nthr ** (exists* v. gpu_pts_to_array #a #smem_sz ar #1.0R v))
      (fun _ -> block_setup nthr ** bigstar 0 nthr (shared_pre ar bid)))

  (k :
    (ar: erased (gpu_array a smem_sz)) -> (etid: tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr }) ->
    stt unit (         gpu ** thread_id etid ** shmem_tok ar ** shared_pre ar (bidx_x etid) (tidx_x etid) ** pre (thread_index etid))
             (fun _ -> gpu ** thread_id etid **                 shared_post ar (bidx_x etid) (tidx_x etid) ** post (thread_index etid))
  )
  requires cpu ** bigstar #u1 0 (nblk * nthr) pre
  ensures  cpu ** bigstar #u1 0 (nblk * nthr) post

(* f<<<nblk, nthr>>>(...); *)
fn launch_kernel_n_m_barrier
  (#u1: erased int)
  (nblk : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (nthr : SZ.t { 0 < nthr /\ nthr <= max_threads })
  (#pre #post : (tid:nat{ tid < (nblk * nthr) } -> slprop))

  (#p: (it:nat -> from: nat { 0 <= from /\ from < nthr } -> to: nat { 0 <= to /\ to < nthr } -> slprop))
  (k :
    (etid: tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr }) ->
    stt unit (         gpu ** thread_id etid ** mbarrier_tok nthr p 0 (tidx_x etid) ** pre (thread_index etid))
             (fun _ -> gpu ** thread_id etid ** (exists* it. mbarrier_tok nthr p it (tidx_x etid)) ** post (thread_index etid))
  )
  requires cpu ** bigstar #u1 0 (nblk * nthr) pre
  ensures  cpu ** bigstar #u1 0 (nblk * nthr) post

(* f<<<nblk, nthr>>>(...); *)
fn launch_kernel_n_m
  (#u1: erased int)
  (nblk : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (nthr : SZ.t { 0 < nthr /\ nthr <= max_threads })
  (#pre #post : (tid:nat{ 0 <= tid /\ tid < (nblk * nthr) } -> slprop))
  (k :
    (etid: tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr }) ->
    stt unit (         gpu ** thread_id etid ** pre (thread_index etid))
             (fun _ -> gpu ** thread_id etid ** post (thread_index etid))
  )
  requires cpu ** bigstar #u1 0 (nblk * nthr) pre
  ensures  cpu ** bigstar #u1 0 (nblk * nthr) post

fn launch_kernel_n
  (#u1: erased int)
  (nblk  : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (#pre #post : (tid:nat{ 0 <= tid /\ tid < SZ.v nblk } -> slprop))
  (k :
    (etid:tid_t { gdim_x etid == nblk /\ bdim_x etid == 1sz }) ->
    stt unit (gpu ** thread_id etid ** pre (thread_index etid))
             (fun _ -> gpu ** thread_id etid ** post (thread_index etid))
  )
  requires cpu ** bigstar #u1 0 (SZ.v nblk) pre
  ensures  cpu ** bigstar #u1 0 (SZ.v nblk) post

fn launch_kernel_1_async
  (#pre #post : slprop)
  (k : unit ->
    stt unit (gpu ** pre) (fun _ -> gpu ** post)
  )
  (#e : erased nat)
  requires cpu ** epoch_live e ** pre
  ensures
    exists* e'.
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
