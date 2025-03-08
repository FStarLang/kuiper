module Kuiper.Kernel
#lang-pulse

open Kuiper.Common
open Pulse.Lib.Core
open FStar.Ghost
open Pulse.Lib.BigStar
open Kuiper.ForEvery
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

noextract
fn obtain_shmem
  (#a:Type u#0)
  {| Kuiper.Sized.sized a |}
  (#sz : erased nat)
  (ear : erased (gpu_array a sz))
  requires shmem_tok ear
  returns  ar : gpu_array a sz
  ensures  pure (reveal ear == ar)

fn sync () (#e:erased nat)
  requires
    epoch_live e
  returns
    e' : epoch_t
  ensures
    epoch_done e **
    epoch_live e' **
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
  returns
    e' : epoch_t
  ensures
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
  (nblk : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (nthr : SZ.t { 0 < nthr /\ nthr <= max_threads })
  (#pre #post : natlt nblk -> natlt nthr -> slprop)
  (#p : rpm_t nthr)
  (k :
    (etid: tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr }) ->
    stt unit
      (         gpu ** thread_id etid ** mbarrier_tok nthr p 0 (tidx_x etid) ** pre  (bidx_x etid) (tidx_x etid))
      (fun _ -> gpu ** thread_id etid ** (exists* it. mbarrier_tok nthr p it (tidx_x etid)) ** post (bidx_x etid) (tidx_x etid))
  )
  requires
    cpu **
    (forall+ (b : natlt nblk) (t : natlt nthr). pre b t)
  ensures
    cpu **
    (forall+ (b : natlt nblk) (t : natlt nthr). post b t)

(* f<<<nblk, nthr>>>(...); *)
fn launch_kernel_n_m
  (nblk : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (nthr : SZ.t { 0 < nthr /\ nthr <= max_threads })
  (#pre #post : natlt nblk -> natlt nthr -> slprop)
  (k :
    (etid: tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr }) ->
    stt unit
      (         gpu ** thread_id etid ** pre  (bidx_x etid) (tidx_x etid))
      (fun _ -> gpu ** thread_id etid ** post (bidx_x etid) (tidx_x etid))
  )
  requires
    cpu **
    (forall+ (b : natlt nblk) (t : natlt nthr). pre b t)
  ensures
    cpu **
    (forall+ (b : natlt nblk) (t : natlt nthr). post b t)

fn launch_kernel_n_async
  (nblk  : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (#pre #post : natlt nblk -> slprop)
  (k :
    (etid:tid_t { gdim_x etid == nblk /\ bdim_x etid == 1sz }) ->
    stt unit (gpu ** thread_id etid ** pre (thread_index etid))
             (fun _ -> gpu ** thread_id etid ** post (thread_index etid))
  )
  (#e : erased nat)
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

fn launch_kernel_n
  (nblk  : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (#pre #post : (natlt nblk -> slprop))
  (k :
    (etid:tid_t { gdim_x etid == nblk /\ bdim_x etid == 1sz }) ->
    stt unit (gpu ** thread_id etid ** pre (thread_index etid))
             (fun _ -> gpu ** thread_id etid ** post (thread_index etid))
  )
  requires
    cpu **
    (forall+ (b : natlt nblk). pre b)
  ensures
    cpu **
    (forall+ (b : natlt nblk). post b)

fn launch_kernel_1_async
  (#pre #post : slprop)
  (k : unit ->
    stt unit (gpu ** pre) (fun _ -> gpu ** post)
  )
  (#e : erased nat)
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

noextract inline_for_extraction
fn thread_idx_all () (#n: tid_t)
  preserves
    thread_id n
  requires
    emp
  returns
    id : SZ.t
  ensures
    pure (SZ.v id == thread_index n /\ SZ.v id < max_blocks * max_threads)
