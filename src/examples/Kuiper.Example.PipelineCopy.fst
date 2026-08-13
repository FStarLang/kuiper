module Kuiper.Example.PipelineCopy

#lang-pulse

(* A deliberately small example of the single-threaded async-copy pipeline
   (Kuiper.PipelineCopy).

   One block, one thread. The kernel

     - stages [chunk float == 4] floats from a global array into *shared*
       memory with [array_vec_cpy_pipelined] (extracts to
       [__pipeline_memcpy_async]),
     - commits the batch twice ([__pipeline_commit]); the second commit is
       empty and only exists to show that a later batch can be used to
       retire an earlier one,
     - flushes with [pipeline_wait_all_prior] ([__pipeline_wait_prior(0)]),
     - lowers the resulting [batch_done] to the batch the copy was tagged
       with ([batch_done_lower]) and redeems the pledge, and
     - copies the (now readable) shared tile out to a second global array
       with an ordinary synchronous [array_vec_cpy].

   The destination of the async copy is shared memory on purpose:
   [__pipeline_memcpy_async] is a global->shared instruction on real
   hardware. Kuiper does not enforce that yet (see the LATER note in
   Kuiper.PipelineCopy), so it is a convention here, not a theorem. *)

open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.Array2.Vectorized
open Kuiper.PipelineCopy
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Pulse.Lib.Pledge

module SZ = Kuiper.SizeT
module V = Pulse.Lib.Vec
module A = Pulse.Lib.Array
module B = Kuiper.Barrier

(* A single 4-float shared tile: exactly one vectorized chunk of floats. *)
inline_for_extraction noextract
let shd : list shmem_desc = [SHArray float 4sz]

noextract
unfold
let kpre
  (src dst : array float)
  (s d : seq float)
  (sh : c_shmems shd)
  (bid : natlt 1)
  (tid : natlt 1)
  : slprop
= live_c_shmems sh **
  pts_to_slice src 0 4 s **
  pts_to_slice dst 0 4 d

noextract
unfold
let kpost
  (src dst : array float)
  (s d : seq float)
  (sh : c_shmems shd)
  (bid : natlt 1)
  (tid : natlt 1)
  : slprop
= live_c_shmems sh **
  pts_to_slice src 0 4 s **
  pts_to_slice dst 0 4 s

(* Crutches: [c_l2_row_major] is stated over a [SZ.t] column count, which does
   not unify with the literal [4] in [l2_row_major 1 4]. *)
inline_for_extraction noextract
instance _ct_1x4 : ctlayout (l2_row_major 1 4) = c_l2_row_major 1 4sz

inline_for_extraction noextract
let good (a : array float) : prop =
  is_global_array a /\ aligned 16 a /\ A.length a == 4

#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn kf
  (src : array float { good src })
  (dst : array float { good dst })
  (#s #d : erased (seq float))
  (sh : c_shmems shd)
  (bid : szlt 1)
  (tid : szlt 1)
  ()
  requires
    gpu **
    kpre src dst s d sh bid tid **
    thread_id 1 tid **
    block_id 1 bid **
    B.barrier_tok (B.empty_contract 1) **
    B.barrier_state 0
  ensures
    gpu **
    kpost src dst s d sh bid tid **
    thread_id 1 tid **
    block_id 1 bid **
    B.barrier_tok (B.empty_contract 1) **
    B.barrier_state 0
{
  unfold_c_shmems sh #1.0R (`%shd);
  let (sarr, _) = sh;
  with sv. assert A.pts_to sarr sv;
  gpu_pts_to_ref sarr;

  (* MODELING ASSUMPTION (flagged): Kuiper places all shared arrays in the
     block's dynamic shared memory segment (`extern __shared__`), whose base
     CUDA guarantees to be 16-byte aligned; this array sits at offset 0 in
     it. Kuiper's array model does not track that yet, exactly like the
     pre-existing `aligned 16 local` assumption for kernel-local arrays. *)
  assume pure (aligned 16 sarr);

  pts_to_slice_ref dst 0 4;
  pts_to_slice_ref src 0 4;
  assert pure (A.length sarr == 4);

  (* View the source as a 1x4 row-major matrix so that we can exercise the
     tensor-level wrapper [array2_vec_read_pipelined]. *)
  slice_to_array_full src;
  tensor_abs' (l2_row_major 1 4) (src <: larray float 4);
  let gm = from_array (l2_row_major 1 4) (src <: larray float 4);
  rewrite each from_array (l2_row_major 1 4) (src <: larray float 4) as gm;
  with em. assert gm |-> em;

  (* Mint a batch and issue the asynchronous global -> shared copy. *)
  let b0 = get_batch ();
  array2_vec_read_pipelined gm 0sz 0sz sarr;

  (* Two commits: the copy above lands in batch [b0]; the second commit
     closes an empty batch [b1]. *)
  let b1 = pipeline_commit #b0;
  let b2 = pipeline_commit #b1;

  (* Flushing up to [b1] retires everything committed before it, in
     particular the copy tagged [b0]. *)
  pipeline_wait_all_prior #b1;
  batch_done_lower b1 b0;

  redeem_pledge emp_inames (batch_done b0) _;

  (* Put the source back in its concrete, slice-shaped form. *)
  tensor_concr gm;
  rewrite each core gm as src;
  array_to_slice src;
  drop_ (is_full_slice src _);
  (* The wrapper hands back the tile as a [chest2] view; normalize it back to
     the plain source sequence [s] so the rest of the proof stays small. *)
  with sv0. assert pts_to sarr sv0;
  assert pure (Seq.equal sv0 s);
  rewrite (pts_to sarr sv0) as (pts_to sarr (reveal s));
  array_to_slice sarr;

  (* The tile is now readable: copy it back out synchronously. *)
  array_vec_cpy dst 0sz sarr 0sz;

  with dv. assert pts_to_slice dst 0 4 dv;
  assert pure (Seq.equal dv s);

  slice_to_array sarr;
  rewrite each sarr as fst sh;
  fold_c_shmems sh #1.0R (`%shd);

  (* Ghost tokens: [b0]'s commit permission is forfeited (we already flushed
     through [b1]) and [b2] is a live but unused batch. Both are erasable
     bookkeeping, not memory. *)
  drop_ (batch_committed b0);
  drop_ (batch_done b0);
  drop_ (batch_done b1);
  drop_ (batch_live b2);
  ();
}
#pop-options

ghost
fn setup
  (src dst : array float)
  (s d : seq float)
  ()
  norewrite
  requires
    pts_to_slice src 0 4 s ** pts_to_slice dst 0 4 d
  ensures
    (forall+ (bid : natlt 1).
       pts_to_slice src 0 4 s ** pts_to_slice dst 0 4 d) **
    emp
{
  forevery_singleton_intro #(natlt 1)
    (fun _bid -> pts_to_slice src 0 4 s ** pts_to_slice dst 0 4 d);
}

ghost
fn teardown
  (src dst : array float)
  (s : seq float)
  ()
  norewrite
  requires
    (forall+ (bid : natlt 1). pts_to_slice src 0 4 s ** pts_to_slice dst 0 4 s) **
    emp
  ensures
    pts_to_slice src 0 4 s ** pts_to_slice dst 0 4 s
{
  forevery_singleton_elim #(natlt 1) _;
}

ghost
fn block_setup
  (src dst : array float)
  (s d : seq float)
  (sh : c_shmems shd)
  (bid : natlt 1)
  ()
  norewrite
  requires
    live_c_shmems sh **
    (pts_to_slice src 0 4 s ** pts_to_slice dst 0 4 d)
  ensures
    (forall+ (tid : natlt 1). kpre src dst s d sh bid tid) ** emp
{
  forevery_singleton_intro #(natlt 1) (fun tid -> kpre src dst s d sh bid tid);
}

ghost
fn block_teardown
  (src dst : array float)
  (s d : seq float)
  (sh : c_shmems shd)
  (bid : natlt 1)
  ()
  norewrite
  requires
    (forall+ (tid : natlt 1). kpost src dst s d sh bid tid) ** emp
  ensures
    live_c_shmems sh **
    (pts_to_slice src 0 4 s ** pts_to_slice dst 0 4 s)
{
  forevery_singleton_elim #(natlt 1) (fun tid -> kpost src dst s d sh bid tid);
}

inline_for_extraction noextract
let kdesc
  (src : array float { good src })
  (dst : array float { good dst })
  (s d : erased (seq float))
  : kernel_desc
      (pts_to_slice src 0 4 s ** pts_to_slice dst 0 4 d)
      (pts_to_slice src 0 4 s ** pts_to_slice dst 0 4 s)
  = {
    nblk = 1sz;
    nthr = 1sz;

    shmems_desc = shd;

    barrier_contract = (fun _bid _sh -> B.empty_contract 1);
    barrier_count    = (fun _bid -> 0);
    barrier_ok       = (fun _bid _sh -> B.empty_barrier_transform 1);

    f = kf src dst #s #d;

    kpre  = kpre  src dst s d;
    kpost = kpost src dst s d;

    frame = emp;

    block_pre  = (fun _bid ->
      pts_to_slice src 0 4 s ** pts_to_slice dst 0 4 d);
    block_post = (fun _bid ->
      pts_to_slice src 0 4 s ** pts_to_slice dst 0 4 s);

    setup    = setup    src dst s d;
    teardown = teardown src dst s;

    block_frame    = (fun _sh _bid -> emp);
    block_setup    = block_setup    src dst s d;
    block_teardown = block_teardown src dst s d;

    block_pre_sendable  = solve;
    block_post_sendable = solve;
    kpre_sendable       = solve;
    kpost_sendable      = solve;
  }

fn hf (v : V.vec float)
  preserves cpu
  preserves exists* s. v |-> s ** pure (Seq.length s == 4)
{
  let src = gpu_array_alloc #float 4sz;
  let dst = gpu_array_alloc #float 4sz;

  gpu_memcpy_host_to_device src v 4sz;

  with ss. assert on gpu_loc (src |-> ss);
  map_loc gpu_loc
    #(pts_to src ss)
    #(pts_to_slice src 0 4 ss ** is_full_slice src (Seq.length ss))
    fn _ {};

  with ds. assert on gpu_loc (dst |-> ds);
  map_loc gpu_loc
    #(pts_to dst ds)
    #(pts_to_slice dst 0 4 ds ** is_full_slice dst (Seq.length ds))
    fn _ {};

  launch_sync (kdesc src dst ss ds);

  map_loc gpu_loc
    #(pts_to_slice src 0 4 ss ** is_full_slice src (Seq.length ss))
    #(pts_to src ss)
    fn _ {};
  map_loc gpu_loc
    #(pts_to_slice dst 0 4 ss ** is_full_slice dst (Seq.length ds))
    #(pts_to dst ss)
    fn _ {};

  gpu_memcpy_device_to_host v dst 4sz;

  gpu_array_free src;
  gpu_array_free dst;
}
