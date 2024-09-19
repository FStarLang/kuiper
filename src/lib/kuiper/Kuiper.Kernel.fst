module Kuiper.Kernel
#lang-pulse

open Pulse.Lib.Pervasives
open Pulse.Lib.BigStar
open Kuiper.SizeT
open Kuiper.Array
open Kuiper.Base
open Kuiper.Barrier.RPM
open FStar.Mul
module SZ = FStar.SizeT

let shmem_tok
  (#a:Type u#0)
  {| Kuiper.Sized.sized a |}
  (#sz:nat)
  (ar:gpu_array a sz)
: slprop = magic ()

fn obtain_shmem
  (#a:Type u#0)
  {| Kuiper.Sized.sized a |}
  (#sz : erased nat)
  (ear : erased (gpu_array a sz))
  requires shmem_tok ear
  returns  ar : gpu_array a sz
  ensures  pure (reveal ear == ar)
    { admit () }

(* f<<<nblk, nthr, smem_sz>>>(...); *)
fn launch_kernel_n_m_sync
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
{ admit (); }

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
{ admit (); }

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
{ admit (); }

(* f<<<nblk, 1>>>(...); *)
// Private
fn kernel_n_as_n_m
  (nblk  : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (#pre #post : (tid:nat{ 0 <= tid /\ tid < SZ.v nblk } -> slprop))
  (k :
    (etid:tid_t { gdim_x etid == nblk /\ bdim_x etid == 1sz }) ->
    stt unit (gpu ** thread_id etid ** pre (thread_index etid))
             (fun _ -> gpu ** thread_id etid ** post (thread_index etid))
  )
  (etid:tid_t { gdim_x etid == nblk /\ bdim_x etid == 1sz })
  requires gpu ** thread_id etid ** pre (thread_index etid)
  ensures  gpu ** thread_id etid ** post (thread_index etid)
{
  k etid;
}

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
{
  rewrite (bigstar #u1 0 (SZ.v nblk) pre) as (bigstar #u1 0 (SZ.v nblk * 1) pre);
  launch_kernel_n_m #u1 nblk 1sz #pre #post
    (fun etid -> kernel_n_as_n_m nblk #pre #post k etid);
  rewrite (bigstar #u1 0 (SZ.v nblk * 1) post) as (bigstar #u1 0 (SZ.v nblk) post);
}

(* f<<<1, 1>>>(...); *)
// Private
fn kernel_1_as_n
  (#pre #post : slprop)
  (k : unit ->
    stt unit (gpu ** pre) (fun _ -> gpu ** post)
  )
  (etid:tid_t { gdim_x etid == 1sz /\ bdim_x etid == 1sz })
  requires gpu ** thread_id etid ** pre
  ensures  gpu ** thread_id etid ** post
{
  k ()
}

let epoch_live (n:nat) : slprop = magic ()
let epoch_done (n:nat) : slprop = magic ()

ghost
fn get_epoch ()
  requires emp
  returns e : erased nat
  ensures epoch_live e
{ admit (); }

fn sync () (#e:erased nat)
  requires epoch_live e
  ensures
    exists* e'.
      epoch_live e' ** epoch_done e **
      pure (e' >= e)
{ admit (); }

ghost
fn done_lower (e f :nat)
  requires epoch_done e ** pure (f <= e)
  ensures  epoch_done e ** epoch_done f
{ admit (); }

fn launch_kernel_1_async
  (#pre #post : slprop)
  (k : unit ->
    stt unit (gpu ** pre) (fun _ -> gpu ** post)
  )
  (#e : erased nat)
  requires cpu ** epoch_live e ** pre
  ensures
    exists* e'.
      cpu ** epoch_live e' ** pledge0 (epoch_done e) post **
      pure (e' >= e)
{ admit (); }

// fn launch_kernel_1
//   (#pre #post : slprop)
//   (k : unit ->
//     stt unit (gpu ** pre) (fun _ -> gpu ** post)
//   )
//   requires cpu ** pre
//   ensures  cpu ** post
// {
//   bigstar_single_intro 0 (fun (i: nat { 0 <= i /\ i < 1 }) -> pre);
//   launch_kernel_n 1sz (fun etid -> kernel_1_as_n #pre #post k etid);
//   bigstar_single_elim #0;
// }

inline_for_extraction
fn launch_kernel_1
  (#pre #post : slprop)
  (k : unit ->
    stt unit (gpu ** pre) (fun _ -> gpu ** post)
  )
  requires cpu ** pre
  ensures  cpu ** post
{
  let _ = get_epoch ();
  launch_kernel_1_async #pre #post k;
  unfold pledge0;
  sync ();
  with e'. assert (epoch_done e');
  redeem_pledge emp_inames (epoch_done e') post;
  drop_ (epoch_done e');
  drop_ (epoch_live _);
}
