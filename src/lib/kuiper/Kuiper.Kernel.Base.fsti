module Kuiper.Kernel.Base
#lang-pulse

open Kuiper.Common
open Pulse.Lib.Core
open FStar.Ghost
open Kuiper.ForEvery
open Kuiper.IntAliases
open Kuiper.Array
open Kuiper.Base
open Kuiper.Epoch
module SZ = FStar.SizeT
open Kuiper.SizeT
open Pulse.Lib.Pledge

val shmem_tok
  (#a:Type u#0)
  {| Kuiper.Sized.sized a |}
  (#sz:nat)
  (ar:gpu_array a sz)
: slprop

(* This is the single primitive for launching kernels, with the most general
type and capabilities. There are many simpler versions in the Kuiper.Kernel module,
all implemented using this one and without any extra assumptions. *)
fn launch_kernel_n_m_shmem_async
  (nblk : szp { nblk <= max_blocks })
  (nthr : szp { nthr <= max_threads })
  (#pre #post : natlt nblk -> natlt nthr -> slprop)
  (a : Type u#0) {| Kuiper.Sized.sized a |}
  (smem_sz : SZ.t)
  (#shared_pre #shared_post : gpu_array a smem_sz -> natlt nblk -> natlt nthr -> slprop)
  (setup :
    (ar: gpu_array a smem_sz) ->
    (bid: natlt nblk) ->
    stt_ghost unit emp_inames
      (block_setup nthr ** (exists* v. gpu_pts_to_array #a #smem_sz ar #1.0R v))
      (fun _ -> block_setup nthr ** (forall+ (i : natlt nthr). shared_pre ar bid i)))
  (k :
    (ar: erased (gpu_array a smem_sz)) ->
    (etid: tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr }) ->
    stt unit
      (         gpu **
                thread_id etid **
                shmem_tok ar **
                shared_pre ar (bidx_x etid) (tidx_x etid) **
                pre (bidx_x etid) (tidx_x etid))
      (fun _ -> gpu **
                thread_id etid **
                // shmem_tok ar **
                shared_post ar (bidx_x etid) (tidx_x etid) **
                post (bidx_x etid) (tidx_x etid))
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

fn obtain_shmem
  (#a:Type u#0)
  {| Kuiper.Sized.sized a |}
  (#sz : erased nat)
  (ear : erased (gpu_array a sz))
  requires shmem_tok ear
  returns  ar : gpu_array a sz
  ensures  pure (reveal ear == ar)

fn sync () (#e:epoch_t)
  requires
    epoch_live e
  returns
    e' : epoch_t
  ensures
    epoch_done e **
    epoch_live e' **
    pure (e' >= e)
