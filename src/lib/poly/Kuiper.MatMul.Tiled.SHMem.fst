module Kuiper.MatMul.Tiled.SHMem

#lang-pulse

fn __synct ()
  requires emp
  ensures emp
{
  open Kuiper.Barrier.RPM;
  let p : rpm_t 1 = (fun _ _ _ -> emp);
  assume (mbarrier_tok 1 p 0 0);
  rewrite emp as row p 0 0 by tadmit ();
  mbarrier_wait ();
  rewrite col p 0 0 as emp by tadmit ();
  drop_ (mbarrier_tok 1 p 1 0);
  ()
}

open Kuiper
module M4  = Kuiper.Matrix4
module MS = Kuiper.Spec.MatMul
module MU = Kuiper.MatMul.Util
module SZ = FStar.SizeT
open Kuiper.EMatrix4
open Kuiper.Matrix.Reprs.Type
module Nop = Kuiper.Kernel.Nop

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
  (#mrows #mshared #mcols #tile : pos)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (eA : ematrix4 et mrows mshared tile tile)
  (eB : ematrix4 et mshared mcols tile tile)
  (f : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  m4_pts_to gA #(f /. mlayout_size lC) eA **
  m4_pts_to gB #(f /. mlayout_size lC) eB **
  (exists* v.
    m4_pts_to_cell gC #1.0R
      (bid / mcols) (bid % mcols)
      (tid / tile) (tid % tile) v)

(* NO FUNCTIONAL SPEC RIGHT NOW *)
unfold
let kpost
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols #tile : pos)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (eA : ematrix4 et mrows mshared tile tile)
  (eB : ematrix4 et mshared mcols tile tile)
  (f : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  m4_pts_to gA #(f /. mlayout_size lC) eA **
  m4_pts_to gB #(f /. mlayout_size lC) eB **
  (exists* v.
    m4_pts_to_cell gC #1.0R
      (bid / mcols) (bid % mcols)
      (tid / tile) (tid % tile) v)

inline_for_extraction noextract
fn kernel
  (tile : szp) (* block dim *)
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : SZ.t)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared tile tile)
  (#eB : ematrix4 et mshared mcols   tile tile)
  (#f : perm)
  (ear : erased (gpu_array et (tile *^ tile)))
  (ebid : enatlt2 mrows mcols)
  (etid : enatlt2 tile  tile)
  requires
    gpu **
    kpre gA gB gC eA eB f ebid etid **
    thread_id (tile * tile) etid **
    block_id (mrows * mcols) ebid **
    shmem_tok ear **
    gpu_pts_to_array1 ear etid
  ensures
    gpu **
    kpost gA gB gC eA eB f ebid etid **
    thread_id (tile * tile) etid **
    block_id (mrows * mcols) ebid **
    shmem_tok ear **
    gpu_pts_to_array1 ear etid
{
  let bid = get_bid (); rewrite each ebid as SZ.v bid;
  let tid = get_tid (); rewrite each etid as SZ.v tid;
  let id = bid *^ (tile *^ tile) +^ tid;

  let mrow, mcol = s_divmod mcols bid;
  let brow, bcol = s_divmod tile  tid;

  with bi0 bj0 i0 j0 v0.
    rewrite
      m4_pts_to_cell gC #1.0R bi0 bj0 i0 j0 v0
    as
      m4_pts_to_cell gC #1.0R mrow mcol brow bcol v0;

  assert (pure (mrow < mrows));
  assert (pure (mcol < mcols));
  assert (pure (brow < tile));
  assert (pure (bcol < tile));

  let s = MU.matmul_tiled_dotprod gA gB mrow mcol brow bcol;
  M4.gpu_matrix_write_cell gC mrow mcol brow bcol s;

  with v'.
    rewrite
      M4.gpu_matrix_pts_to_cell gC mrow mcol brow bcol v'
    as
      M4.gpu_matrix_pts_to_cell gC
        (ebid / mcols) (ebid % mcols)
        (etid / tile) (etid % tile) v';

  ()
}

ghost
fn setup
  (tile : SZ.t)
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : SZ.t)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared tile tile)
  (#eB : ematrix4 et mshared mcols   tile tile)
  (#eC : ematrix4 et mrows   mcols   tile tile)
  ()
  requires
    (gA |-> eA) **
    (gB |-> eB) **
    (gC |-> eC)
  ensures
    forall+ (bid : natlt2 mrows mcols)
            (tid : natlt2 tile  tile).
      kpre gA gB gC eA eB 1.0R bid tid
{
  admit();
}

ghost
fn teardown
  (tile : SZ.t)
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : SZ.t)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared tile tile)
  (#eB : ematrix4 et mshared mcols   tile tile)
  (#eC : ematrix4 et mrows   mcols   tile tile)
  ()
  requires
    forall+ (bid : natlt2 mrows mcols)
            (tid : natlt2 tile  tile).
      kpost gA gB gC eA eB 1.0R bid tid
  ensures
    (gA |-> eA) **
    (gB |-> eB) **
    (gC |-> MS.matmul eA eB)
{
  admit();
}

ghost
fn block_setup
  (tile : szp{tile * tile <= max_threads}) (* block dim *)
  (#et : Type0) {| scalar et |}
  (nblk : pos)
  (ar: gpu_array et (tile *^ tile))
  (bid: natlt nblk)
  requires
   block_setup (tile *^ tile) **
   (exists* v. gpu_pts_to_array #et #(tile *^ tile) ar #1.0R v)
  ensures
   block_setup (tile *^ tile) **
   (forall+ (tid : natlt (tile *^ tile)).
     gpu_pts_to_array1 ar tid)
{
  with v. assert gpu_pts_to_array #et #(tile *^ tile) ar #1.0R v;
  gpu_pts_to_slice_ref ar 0 (tile *^ tile);
  gpu_array_slice_1 ar;
  forevery_fromnat (tile *^ tile) (fun i ->
    gpu_pts_to_slice ar i (i+1) seq![Seq.index v i]);
  admit();
}

inline_for_extraction noextract
let mk_kernel
  (tile : szp) (* block dim *)
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : SZ.t)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared tile tile)
  (#eB : ematrix4 et mshared mcols   tile tile)
  (#eC : ematrix4 et mrows   mcols   tile tile)
  (_ : squash (mrows * mcols <= max_blocks
               /\ tile * tile <= max_threads))
  : kernel_desc
      ((gA |-> eA) ** (gB |-> eB) ** (gC |-> eC))
      ((gA |-> eA) ** (gB |-> eB) ** (gC |-> MS.matmul eA eB))
= {
  nblk = mrows *^ mcols;
  nthr = tile *^ tile;

  shmem_type = et;
  shmem_type_is_sized = solve;
  shmem_sz = (tile *^ tile);

  block_pre  = (fun ar bid tid -> gpu_pts_to_array1 ar tid);
  block_post = (fun ar bid tid -> gpu_pts_to_array1 ar tid);
  block_setup = block_setup tile (mrows *^ mcols);

  kpre      = kpre gA gB gC eA eB 1.0R;
  setup     = (fun () -> setup tile gA gB gC #eA #eB #eC ());

  kpost     = kpost gA gB gC eA eB 1.0R;
  teardown  = teardown tile gA gB gC #eA #eB #eC;

  f = kernel tile #et #_ #mrows #mshared #mcols gA gB gC #eA #eB #1.0R;
}

inline_for_extraction noextract
fn matmul_gpu
  (tile : szp) (* block dim *)
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : szp)
  (lA : mlayout4 mrows   mshared tile tile)
  (lB : mlayout4 mshared mcols   tile tile)
  (lC : mlayout4 mrows   mcols   tile tile)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared tile tile)
  (#eB : ematrix4 et mshared mcols   tile tile)
  (#eC : ematrix4 et mrows   mcols   tile tile)
  preserves
    cpu **
    (gA |-> eA) **
    (gB |-> eB)
  requires
    pure (mrows * mcols <= max_blocks) **
    pure (tile * tile <= max_threads) **
    (gC |-> eC)
  ensures
    gC |-> MS.matmul eA eB
{
  launch_kernel_sync (mk_kernel tile #et #_ #mrows #mshared #mcols #lA #lB #lC gA gB gC #eA #eB #eC ());
  ()
}
