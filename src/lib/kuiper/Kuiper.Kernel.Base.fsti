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

noeq
inline_for_extraction noextract
type kernel_desc = {
  nblk : (x : SZ.t { 0 < x /\ x <= max_blocks });
  nthr : (x : SZ.t { 0 < x /\ x <= max_threads });

  shmem_type : Type u#0;
  shmem_type_is_sized : Kuiper.Sized.sized shmem_type;
  shmem_sz : sz;

  (* This is used to split up the array for the threads in the block.
     It should definitely be generalized to be more uniform with
     other levels of the hierarchy, e.g. by having a teardown. Currently
     we just drop the block_post. *)
  block_pre  : gpu_array shmem_type shmem_sz -> natlt nblk -> natlt nthr -> slprop;
  block_post : gpu_array shmem_type shmem_sz -> natlt nblk -> natlt nthr -> slprop;
  block_setup : (
    (ar: gpu_array shmem_type shmem_sz) ->
    (bid: natlt nblk) ->
    stt_ghost unit emp_inames
      (requires
        block_setup nthr **
        (exists* v. gpu_pts_to_array #shmem_type #shmem_sz ar #1.0R v))
      (ensures fun _ ->
        block_setup nthr **
        (forall+ (i : natlt nthr). block_pre ar bid i))
  );

  kpre : natlt nblk -> natlt nthr -> slprop;
  kpost : natlt nblk -> natlt nthr -> slprop;

  f : (
    eshmem : erased (gpu_array shmem_type shmem_sz) ->
    ebid : enatlt nblk ->
    etid : enatlt nthr ->
    stt unit
      (requires
         gpu **
         kpre ebid etid **
         thread_id nthr etid **
         block_id nblk ebid **
         shmem_tok eshmem **
         block_pre eshmem ebid etid
      )
      (ensures fun _ ->
         gpu **
         kpost ebid etid **
         thread_id nthr etid **
         block_id nblk ebid **
         shmem_tok eshmem **
         block_post eshmem ebid etid)
  );
  full_pre : slprop;
  setup : (
    unit ->
    stt_ghost unit emp_inames
      (requires full_pre)
      (ensures  fun _ -> forall+ (bid : natlt nblk) (tid : natlt nthr). kpre bid tid)
  );
  full_post : slprop;
  teardown : (
    unit ->
    stt_ghost unit emp_inames
      (requires forall+ (bid : natlt nblk) (tid : natlt nthr). kpost bid tid)
      (ensures  fun _ -> full_post)
  );
}

(* This is the single primitive for launching kernels, with the most general
type and capabilities. There are many simpler versions in the Kuiper.Kernel module,
all implemented using this one and without any extra assumptions. *)
fn launch_kernel (k : kernel_desc)
  (#e : epoch_t)
  requires
    cpu **
    epoch_live e **
    k.full_pre
  returns
    e' : epoch_t
  ensures
    cpu **
    epoch_live e' **
    pledge0 (epoch_done e') k.full_post **
    pure (e' >= e)

inline_for_extraction noextract
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
