module Kuiper.Kernel.Sync
open Pulse.Lib.Core
open Kuiper.Common
open Kuiper.ForEvery
open Kuiper.Base
open Kuiper.SizeT
include Kuiper.Kernel.Base
include Kuiper.Kernel.Desc
friend Kuiper.Array.Core // for gpu_array_alloc_vis, gpu_array_free_gen
friend Kuiper.Locs // construct launch tokens at the modeled thread locations

#lang-pulse

open Pulse.Lib.Pervasives

module Par = Pulse.Lib.Par

(* A model for launch_kernel_sync *)

module SH = Kuiper.SHMem

module A = Pulse.Lib.Array.Core
module L = Kuiper.Locs.Base

(* All blocks and threads of this modeled launch use the same immutable
   configuration. The exact block location also fixes the thread count,
   which is not an argument of the public block_id token. *)
let launch_context_for (#pre #post:slprop) (k:kernel_desc pre post) =
  ctx:L.launch_context {
    ctx.launch_gpu == 0 /\ ctx.launch_nblk == k.nblk /\ ctx.launch_nthr == k.nthr
  }

let c_shmem_full (#d : SH.shmem_desc) (c : SH.c_shmem d) : prop =
  match d with
  | SH.SHArray ty len -> A.is_full_array #ty c

let rec c_shmems_full (#ds : list SH.shmem_desc) (c : SH.c_shmems ds) : prop =
  match ds with
  | [] -> True
  | d :: ds ->
    let c : SH.c_shmem d & SH.c_shmems ds = c in
    c_shmem_full #d (fst c) /\
    c_shmems_full #ds (snd c)

(* Destructing a [shmem_desc] and projecting its fields agree.  F* no longer
   gets these facts for free (the [match] on [d] does not reduce when [d] is a
   variable), so we state them as an explicit lemma. *)
let c_shmem_eq (d : SH.shmem_desc)
  : Lemma (SH.c_shmem d == larray d.ty d.len /\
           SH.d_ty d == d.ty /\ SH.d_len d == d.len)
  = let SH.SHArray ty len = d in ()

// Models the allocation of per-block shared memory by the GPU runtime.
noextract
fn rec alloc_c_shmems
  (block_loc: loc_id)
  (d: list SH.shmem_desc)
  preserves loc block_loc
  returns res: SH.c_shmems d
  ensures SH.live_c_shmems res
  ensures pure (SH.c_shmems_block_inv res /\ c_shmems_full res)
  decreases d
{
  match d {
    norewrite
    Nil -> {
      let res : SH.c_shmems d = 0;
      SH.fold_live_c_shmems_nil res #1.0R;
      rewrite SH.live_c_shmems #[] res
        as SH.live_c_shmems res;
      res
    }
    norewrite
    Cons a q -> {
      let resq = alloc_c_shmems block_loc q;
      let resa' = Kuiper.Array.Core.gpu_array_alloc_vis #a.ty #a.sized a.len block_loc block_of;
      c_shmem_eq a;
      let resa : larray a.ty a.len = resa';
      on_elim _;
      rewrite each resa' as resa;
      SH.fold_live_c_shmem #a resa;
      let res : SH.c_shmems d = (resa, resq);
      rewrite each resa as fst #(SH.c_shmem a) #(SH.c_shmems q) res;
      rewrite each resq as snd #(SH.c_shmem a) #(SH.c_shmems q) res;
      SH.fold_live_c_shmems_cons #a #q res #1.0R;
      rewrite each (a :: q) as d;
      res
    }
  }
}

// Models the liberation of per-block shared memory by the GPU runtime.
noextract
fn rec free_c_shmems
  (block_loc: loc_id)
  (d: list SH.shmem_desc)
  (res: SH.c_shmems d)
  preserves loc block_loc
  requires SH.live_c_shmems res
  requires pure (SH.c_shmems_block_inv res /\ c_shmems_full res)
  decreases d
{
  match d {
    Nil -> {
      SH.unfold_live_c_shmems_nil res #1.0R;
    }
    Cons a q -> {
      let res' : SH.c_shmems (a :: q) = res;
      rewrite each res as res';
      SH.unfold_live_c_shmems_cons #a #q res' #1.0R;
      let resq : SH.c_shmems q = snd res';
      rewrite each (snd res') as resq;
      free_c_shmems block_loc q resq;
      let resa : SH.c_shmem a = fst res';
      rewrite each (fst res') as resa;
      SH.unfold_live_c_shmem resa;
      let resa' : larray a.ty a.len = resa;
      rewrite each (resa <: larray a.ty a.len) as resa';
      on_intro (resa' |-> _);
      Kuiper.Array.Core.gpu_array_free_gen resa' block_loc;
    }
  }
}

inline_for_extraction noextract
let szlt_coerce #m #n (i: szlt n { i < m }) : szlt m = i

noextract
divergent
fn rec run_block_threads
  (#full_pre #full_post : slprop)
  (k : kernel_desc full_pre full_post)
  (ctx : launch_context_for k)
  (bid: szlt k.nblk)
  (sh: SH.c_shmems k.shmems_desc {SH.c_shmems_inv sh /\ c_shmems_full sh})
  (upto: sz { upto <= k.nthr})
  preserves loc (L.block_id_loc ctx bid)
  requires
    (forall+ (i : natlt upto). k.kpre sh bid (natlt_coerce i))
  ensures
    (forall+ (i : natlt upto). k.kpost sh bid (natlt_coerce i))
  decreases SizeT.v upto
{
  if (upto = 0sz) {
    forevery_elim_empty _;
    forevery_intro_empty (fun (i : natlt upto) -> k.kpost sh bid (natlt_coerce i))
  } else {
    forevery_natlt_pop upto (fun (i: natlt upto) -> k.kpre sh bid (natlt_coerce i));
    let tid : szlt k.nthr = upto -^ 1sz;
    rewrite each (upto - 1) as tid;
    let send_pre = k.kpre_sendable sh () bid tid;
    let send_post = k.kpost_sendable sh () bid tid;
    let tloc = L.thread_id_loc ctx bid tid;
    L.thread_id_loc_lemma ctx bid tid;
    L.block_id_loc_lemma ctx bid;
    Kuiper.Kernel.Par.par
      #(k.kpre sh bid tid)
      #(k.kpost sh bid tid)
      #(loc (L.block_id_loc ctx bid) ** forall+ (i : natlt tid). k.kpre sh bid (natlt_coerce i))
      #(loc (L.block_id_loc ctx bid) ** forall+ (i : natlt tid). k.kpost sh bid (natlt_coerce i))
      block_of tloc
      fn _ {
        fold (thread_id k.nthr tid);
        fold (block_id k.nblk bid);
        fold gpu;
        // Assume the barrier state
        assume Kuiper.Barrier.barrier_state 0;
        assume Kuiper.Barrier.barrier_tok (k.barrier_contract bid sh);
        Mkkernel_desc?.f k sh bid tid ();
        drop_ gpu;
        drop_ (block_id _ _);
        drop_ (thread_id _ _);
        // Drop barrier state
        drop_ (Kuiper.Barrier.barrier_tok _);
        drop_ (Kuiper.Barrier.barrier_state _);
      }
      fn _ {
        run_block_threads k ctx bid sh tid
      };
    rewrite each (v tid) as (upto - 1);
    forevery_natlt_push upto (fun (i: natlt upto) -> k.kpost sh bid (natlt_coerce i));
  }
}

noextract
divergent
fn run_block
  (#full_pre #full_post : slprop)
  (k : kernel_desc full_pre full_post)
  (ctx : launch_context_for k)
  (bid: szlt k.nblk)
  preserves loc (L.block_id_loc ctx bid)
  requires k.block_pre bid
  ensures k.block_post bid
{
  let sh = alloc_c_shmems (L.block_id_loc ctx bid) k.shmems_desc;
  (* Each block gets a single dynamic shared memory region, 16-byte aligned,
  which the extracted code carves into the requested arrays in declaration
  order (see KPR_SHMEM/KPR_SHMEM_AT in include/kuiper.h). The model allocates
  them independently, so that layout is assumed here, as with the alignment of
  cudaMalloc'd arrays. Taking the region to start at 0 is fine: only the
  offsets within it are observable. *)
  assume pure (SH.c_shmems_at sh 0);
  let _ : unit = Mkkernel_desc?.block_setup k sh bid ();
  run_block_threads k ctx bid sh k.nthr;
  Mkkernel_desc?.block_teardown k sh bid ();
  free_c_shmems (L.block_id_loc ctx bid) _ sh;
  ()
}

noextract
divergent
fn rec run_blocks
  (#full_pre #full_post : slprop)
  (k : kernel_desc full_pre full_post)
  (ctx : launch_context_for k)
  (upto: sz { upto <= k.nblk})
preserves gpu
requires
  (forall+ (i : natlt upto). k.block_pre (natlt_coerce i))
ensures
  (forall+ (i : natlt upto). k.block_post (natlt_coerce i))
{
  if (upto = 0sz) {
    forevery_elim_empty _;
    forevery_intro_empty (fun (i : natlt upto) -> k.block_post (natlt_coerce i))
  } else {
    forevery_natlt_pop upto (fun (i: natlt upto) -> k.block_pre (natlt_coerce i));
    let bid : szlt k.nblk = upto -^ 1sz;
    rewrite each (upto - 1) as bid;
    let send_pre = k.block_pre_sendable bid;
    let send_post = k.block_post_sendable bid;
    L.block_id_loc_lemma ctx bid;
    unfold gpu;
    fold gpu;
    Kuiper.Kernel.Par.par
      #(k.block_pre bid)
      #(k.block_post bid)
      #(gpu ** forall+ (i : natlt bid). k.block_pre (natlt_coerce i))
      #(gpu ** forall+ (i : natlt bid). k.block_post (natlt_coerce i))
      gpu_of (L.block_id_loc ctx bid)
      fn _ {
        run_block k ctx bid;
      }
      fn _ {
        run_blocks k ctx bid
      };
    rewrite each (v bid) as (upto - 1);
    forevery_natlt_push upto (fun (i: natlt upto) -> k.block_post (natlt_coerce i));
  }
}

noextract
divergent
fn launch_kernel_full_sync
  (#full_pre #full_post : slprop)
  (k : kernel_desc full_pre full_post)
  requires
    cpu **
    on gpu_loc full_pre
  ensures
    cpu **
    on gpu_loc full_post
{
  let ctx : launch_context_for k = {
    L.launch_gpu = 0; L.launch_nblk = k.nblk; L.launch_nthr = k.nthr
  };
  gpu_id_loc_lemma 0;
  Kuiper.Kernel.Par.impersonate_div
    unit
    gpu_loc
    (on gpu_loc full_pre)
    (fun _ -> on gpu_loc full_post)
    fn _ {
      fold gpu;
      on_elim full_pre;
      Mkkernel_desc?.setup k ();
      run_blocks k ctx k.nblk;
      Mkkernel_desc?.teardown k ();
      on_intro full_post;
      drop_ gpu;
    };
}
