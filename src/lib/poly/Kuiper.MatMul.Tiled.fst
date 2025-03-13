module Kuiper.MatMul.Tiled

#lang-pulse

open Kuiper
module M   = Kuiper.Matrix
module M4  = Kuiper.Matrix4
module MS = Kuiper.Spec.MatMul
module MU = Kuiper.MatMul.Util
module SZ = FStar.SizeT
open Kuiper.EMatrix4
open Kuiper.Matrix.Reprs.Type

open Kuiper.Matrix4 {
  gpu_matrix as gpu_matrix4,
  gpu_matrix_pts_to as m4_pts_to,
  gpu_matrix_pts_to_cell as m4_pts_to_cell,
  mlayout4,
  clayout4
}

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols #bdim : pos)
  (#lA : mlayout4 mrows   mshared bdim bdim)
  (#lB : mlayout4 mshared mcols   bdim bdim)
  (#lC : mlayout4 mrows   mcols   bdim bdim)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (eA : ematrix4 et mrows mshared bdim bdim)
  (eB : ematrix4 et mshared mcols bdim bdim)
  (f : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (bdim * bdim))
  : slprop
  =
  m4_pts_to gA #(f /. mlayout_size lC) eA **
  m4_pts_to gB #(f /. mlayout_size lC) eB **
  (exists* v.
    m4_pts_to_cell gC #1.0R
      (bid / mcols) (bid % mcols)
      (tid / bdim) (tid % bdim) v)

(* NO FUNCTIONAL SPEC RIGHT NOW *)
unfold
let kpost
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols #bdim : pos)
  (#lA : mlayout4 mrows   mshared bdim bdim)
  (#lB : mlayout4 mshared mcols   bdim bdim)
  (#lC : mlayout4 mrows   mcols   bdim bdim)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (eA : ematrix4 et mrows mshared bdim bdim)
  (eB : ematrix4 et mshared mcols bdim bdim)
  (f : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (bdim * bdim))
  : slprop
  =
  m4_pts_to gA #(f /. mlayout_size lC) eA **
  m4_pts_to gB #(f /. mlayout_size lC) eB **
  (exists* v.
    m4_pts_to_cell gC #1.0R
      (bid / mcols) (bid % mcols)
      (tid / bdim) (tid % bdim) v)

inline_for_extraction
type kernel_fixed_ty
  (bdim : szp) (* block dim *)
  (et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : SZ.t)
  (lA : mlayout4 mrows   mshared bdim bdim)
  (lB : mlayout4 mshared mcols   bdim bdim)
  (lC : mlayout4 mrows   mcols   bdim bdim)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
: Type0
=
  (gA : gpu_matrix4 et lA) ->
  (gB : gpu_matrix4 et lB) ->
  (gC : gpu_matrix4 et lC) ->
  (#eA : ematrix4 et mrows   mshared bdim bdim) ->
  (#eB : ematrix4 et mshared mcols   bdim bdim) ->
  (#f : perm) ->
  (ebid : enatlt (mrows * mcols)) ->
  (etid : enatlt (bdim * bdim)) ->
  stt unit
  (requires
    gpu **
    block_id (mrows * mcols) ebid **
    thread_id (bdim * bdim) etid **
    kpre gA gB gC eA eB f ebid etid)
  (ensures fun _ ->
    gpu **
    block_id (mrows * mcols) ebid **
    thread_id (bdim * bdim) etid **
    kpost gA gB gC eA eB f ebid etid)

#set-options "--print_implicits"

inline_for_extraction noextract
fn kernel_fixed_f
  (bdim : szp) (* block dim *)
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : SZ.t)
  (lA : mlayout4 mrows   mshared bdim bdim)
  (lB : mlayout4 mshared mcols   bdim bdim)
  (lC : mlayout4 mrows   mcols   bdim bdim)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared bdim bdim)
  (#eB : ematrix4 et mshared mcols   bdim bdim)
  (#f : perm)
  (ebid : enatlt (mrows * mcols))
  (etid : enatlt (bdim * bdim))
  requires
    gpu **
    block_id (mrows * mcols) ebid **
    thread_id (bdim * bdim) etid **
    kpre gA gB gC eA eB f ebid etid
  ensures
    gpu **
    block_id (mrows * mcols) ebid **
    thread_id (bdim * bdim) etid **
    kpost gA gB gC eA eB f ebid etid
{
  let bid = get_bid (); rewrite each ebid as SZ.v bid;
  let tid = get_tid (); rewrite each etid as SZ.v tid;
  let id = bid *^ (bdim *^ bdim) +^ tid;

  let mrow, mcol = s_divmod mcols bid;
  let brow, bcol = s_divmod bdim  tid;

  with bi0 bj0 i0 j0 v0.
    rewrite
      m4_pts_to_cell gC #1.0R bi0 bj0 i0 j0 v0
    as
      m4_pts_to_cell gC #1.0R mrow mcol brow bcol v0;

  assert (pure (mrow < mrows));
  assert (pure (mcol < mcols));
  assert (pure (brow < bdim));
  assert (pure (bcol < bdim));

  let s = MU.matmul_tiled_dotprod gA gB mrow mcol brow bcol;
  M4.gpu_matrix_write_cell gC mrow mcol brow bcol s;

  with v'.
    rewrite
      M4.gpu_matrix_pts_to_cell gC mrow mcol brow bcol v'
    as
      M4.gpu_matrix_pts_to_cell gC
        (ebid / mcols) (ebid % mcols)
        (etid / bdim) (etid % bdim) v';

  ()
}

let kernel_fixed = kernel_fixed_f

// let mksz = SZ.uint_to_t

ghost
fn setup
  (bdim : pos) (* block dim *)
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : nat)
  (#lA : mlayout4 mrows   mshared bdim bdim)
  (#lB : mlayout4 mshared mcols   bdim bdim)
  (#lC : mlayout4 mrows   mcols   bdim bdim)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared bdim bdim)
  (#eB : ematrix4 et mshared mcols   bdim bdim)
  (#eC : ematrix4 et mrows   mcols   bdim bdim)
  requires
    (gA |-> eA) **
    (gB |-> eB) **
    (gC |-> eC)
  ensures
    forall+ (bid : natlt (mrows * mcols))
            (tid : natlt (bdim  * bdim)).
      kpost gA gB gC eA eB 1.0R bid tid
{
  admit();
}

ghost
fn teardown
  (bdim : pos) (* block dim *)
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : nat)
  (#lA : mlayout4 mrows   mshared bdim bdim)
  (#lB : mlayout4 mshared mcols   bdim bdim)
  (#lC : mlayout4 mrows   mcols   bdim bdim)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared bdim bdim)
  (#eB : ematrix4 et mshared mcols   bdim bdim)
  (#eC : ematrix4 et mrows   mcols   bdim bdim)
  requires
    forall+ (bid : natlt (mrows * mcols))
            (tid : natlt (bdim  * bdim)).
      kpre gA gB gC eA eB 1.0R bid tid
  ensures
    (gA |-> eA) **
    (gB |-> eB) **
    (gC |-> MS.matmul eA eB)
{
  admit();
}

inline_for_extraction noextract
fn matmul_gpu_fixed
  (bdim : szp) (* block dim *)
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : SZ.t)
  (lA : mlayout4 mrows   mshared bdim bdim)
  (lB : mlayout4 mshared mcols   bdim bdim)
  (lC : mlayout4 mrows   mcols   bdim bdim)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
  (kk : kernel_fixed_ty bdim et lA lB lC #_ #_ #_)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared bdim bdim)
  (#eB : ematrix4 et mshared mcols   bdim bdim)
  (#eC : ematrix4 et mrows   mcols   bdim bdim)
  preserves
    cpu **
    (gA |-> eA) **
    (gB |-> eB)
  requires
    pure (mrows * mcols <= max_blocks) **
    pure (bdim * bdim <= max_threads) **
    (gC |-> eC)
  ensures
    gC |-> MS.matmul eA eB
{
  open FStar.SizeT;
  setup bdim gA gB gC;

  let nblk = mrows *^ mcols;
  let nthr = bdim *^ bdim;

  forevery_rw_size2 (mrows * mcols) nblk
                    (bdim * bdim) nthr;

  (* FIXME: F* inference failure means we need to annotate pre/post (somewhat) *)
  (* We also need eta due to the extraction rules looking for it. *)
  launch_kernel_n_m
    nblk
    nthr
    #(kpre  _ _ _ _ _ _)
    #(kpost _ _ _ _ _ _)
    (fun ebid etid -> kk gA gB gC ebid etid);

  forevery_rw_size2 nblk (mrows * mcols)
                    nthr (bdim * bdim);

  teardown bdim gA gB gC #eA #eB #eC;
  ()
}
