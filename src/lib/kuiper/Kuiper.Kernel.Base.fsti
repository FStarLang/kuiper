module Kuiper.Kernel.Base
#lang-pulse

open Kuiper.Common
open Pulse.Lib.Core
open FStar.Ghost
open Kuiper.Base
open Kuiper.Array
open Kuiper.Epoch
open Pulse.Lib.Pledge
open Kuiper.Kernel.Desc

(* This is the single primitive for launching kernels, with the most general
type and capabilities. There are many simpler versions in the Kuiper.Kernel module,
all implemented using this one and without any extra assumptions. *)
noextract
fn launch_kernel_full
  (#full_pre : slprop)
  (#full_post : slprop)
  (k : kernel_desc full_pre full_post)
  (#e : epoch_t)
  requires
    cpu **
    epoch_live e **
    full_pre
  returns
    e' : epoch_t
  ensures
    cpu **
    epoch_live e' **
    pledge0 (epoch_done e') full_post **
    pure (e' >= e)

(* Conretize the erased pointer to shared memory. *)
noextract
fn obtain_shmem
  (#a:Type u#0)
  {| Kuiper.Sized.sized a |}
  (#sz : erased nat)
  (ear : erased (gpu_array a sz))
  requires shmem_tok ear
  returns  ar : gpu_array a sz
  ensures  pure (reveal ear == ar)

(* Sync the device: wait for all pending kernels. *)
noextract
fn sync_device () (#e:epoch_t)
  requires
    epoch_live e
  returns
    e' : epoch_t
  ensures
    epoch_done e **
    epoch_live e' **
    pure (e' >= e)


// (*** UGLY ZONE ***)


// module SZ = FStar.SizeT
// open Kuiper.ForEvery
// open Kuiper.SizeT

// (* To be removed in favor of above *)
// inline_for_extraction noextract
// fn launch_kernel_n_m_shmem_async
//   (nblk : szp { nblk <= max_blocks })
//   (nthr : szp { nthr <= max_threads })
//   (#pre #post : natlt nblk -> natlt nthr -> slprop)
//   (a : Type u#0) {| Kuiper.Sized.sized a |}
//   (smem_sz : SZ.t)
//   (#shared_pre #shared_post : gpu_array a smem_sz -> natlt nblk -> natlt nthr -> slprop)
//   (setup :
//     (ar: gpu_array a smem_sz) ->
//     (bid: natlt nblk) ->
//     stt_ghost unit emp_inames
//       (block_setup_tok nthr ** (exists* v. gpu_pts_to_array #a #smem_sz ar #1.0R v))
//       (fun _ -> block_setup_tok nthr ** (forall+ (i : natlt nthr). shared_pre ar bid i)))
//   (k :
//     (ar: erased (gpu_array a smem_sz)) ->
//     (bid : enatlt nblk) ->
//     (tid : enatlt nthr) ->
//     stt unit
//       (         gpu **
//                 block_id nblk bid **
//                 thread_id nthr tid **
//                 shmem_tok ar **
//                 shared_pre ar bid tid **
//                 pre bid tid)
//       (fun _ -> gpu **
//                 block_id nblk bid **
//                 thread_id nthr tid **
//                 // shmem_tok ar **
//                 shared_post ar bid tid **
//                 post bid tid)
//   )
//   (#e : epoch_t)
//   requires
//     cpu **
//     epoch_live e **
//     (forall+ (b : natlt nblk) (t : natlt nthr). pre b t)
//   returns
//     e' : epoch_t
//   ensures
//     cpu **
//     epoch_live e' **
//     pledge0 (epoch_done e')
//       (forall+ (b : natlt nblk) (t : natlt nthr). post b t) **
//     pure (e' >= e)
