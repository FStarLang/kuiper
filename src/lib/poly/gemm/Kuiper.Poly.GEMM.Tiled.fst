module Kuiper.Poly.GEMM.Tiled

#lang-pulse

open Kuiper
module M4  = Kuiper.Matrix4
module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Poly.GEMM.Util
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
  (comb : et -> et -> et)
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (eA : ematrix4 et mrows mshared tile tile)
  (eB : ematrix4 et mshared mcols tile tile)
  (eC : ematrix4 et mrows mcols tile tile)
  (f : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  m4_pts_to gA #(f /. mlayout_size lC) eA **
  m4_pts_to gB #(f /. mlayout_size lC) eB **
  m4_pts_to_cell gC #1.0R
      (bid / mcols) (bid % mcols)
      (tid / tile) (tid % tile) (macc eC (bid / mcols) (bid % mcols)
                                         (tid / tile) (tid % tile))

(* NO FUNCTIONAL SPEC RIGHT NOW *)
unfold
let kpost
  (#et : Type0) {| scalar et |}
  (comb : et -> et -> et)
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (eA : ematrix4 et mrows mshared tile tile)
  (eB : ematrix4 et mshared mcols tile tile)
  (eC : ematrix4 et mrows mcols tile tile)
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


let block_pre
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : et -> et -> et)
  (nblk : pos)
  (ar: gpu_array et (2sz *^ tile *^ tile))
  (bid : natlt nblk)
  (tid : natlt (tile *^ tile))
  : slprop
= forall+ (tid : natlt (tile *^ tile)).
    gpu_pts_to_array1 ar tid **
    gpu_pts_to_array1 ar (tid + tile * tile)
let block_post = block_pre


inline_for_extraction noextract
fn kf
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : et -> et -> et)
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
  (#f : perm)
  (bid : szlt2 mrows mcols)
  (tid : szlt2 tile  tile)
  ()
  requires
    gpu **
    kpre comb tile gA gB gC eA eB eC f bid tid **
    thread_id (tile * tile) tid **
    block_id (mrows * mcols) bid
  ensures
    gpu **
    kpost comb tile gA gB gC eA eB eC f bid tid **
    thread_id (tile * tile) tid **
    block_id (mrows * mcols) bid
{
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
  let v0 = M4.gpu_matrix_read_cell gC mrow mcol brow bcol;
  let v1 = comb v0 s;
  M4.gpu_matrix_write_cell gC mrow mcol brow bcol v1;

  with v'.
    rewrite
      M4.gpu_matrix_pts_to_cell gC mrow mcol brow bcol v'
    as
      M4.gpu_matrix_pts_to_cell gC
        (bid / mcols) (bid % mcols)
        (tid / tile) (tid % tile) v';

  ()
}

ghost
fn setup
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : et -> et -> et)
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
    (forall+ (bid : natlt2 mrows mcols)
            (tid : natlt2 tile  tile).
      kpre comb tile gA gB gC eA eB eC 1.0R bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn teardown
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : et -> et -> et)
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
    (forall+ (bid : natlt2 mrows mcols)
            (tid : natlt2 tile  tile).
      kpost comb tile gA gB gC eA eB eC 1.0R bid tid) **
    emp (* frame *)
  ensures
    (gA |-> eA) **
    (gB |-> eB) **
    (gC |-> MS.gemm comb eC eA eB)
{
  admit();
}

inline_for_extraction noextract
let mk_kernel
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : et -> et -> et)
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
  : kernel_desc_m_n
      ((gA |-> eA) ** (gB |-> eB) ** (gC |-> eC))
      ((gA |-> eA) ** (gB |-> eB) ** (gC |-> MS.gemm comb eC eA eB))
= {
  nblk = mrows *^ mcols;
  nthr = tile *^ tile;

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt2 tile tile). kpre  comb tile gA gB gC eA eB eC 1.0R bid tid);
  block_post = (fun bid -> forall+ (tid : natlt2 tile tile). kpost comb tile gA gB gC eA eB eC 1.0R bid tid);
  setup     = setup    tile comb gA gB gC #eA #eB #eC;
  teardown  = teardown tile comb gA gB gC #eA #eB #eC;

  block_frame    = (fun _bid -> emp);
  block_setup    = (fun bid -> Kuiper.Frame.emp_intro_r ());
  block_teardown = (fun bid -> Kuiper.Frame.emp_elim_r ());

  kpre      = kpre  comb tile gA gB gC eA eB eC 1.0R;
  kpost     = kpost comb tile gA gB gC eA eB eC 1.0R;

  f = kf tile #et #_ comb #mrows #mshared #mcols gA gB gC #eA #eB #eC #1.0R;
}

inline_for_extraction noextract
fn matmul_gpu
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : et -> et -> et)
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
    gC |-> MS.gemm comb eC eA eB
{
  dassert (tile `SZ.gt` 0sz);
  launch_sync (mk_kernel tile comb gA gB gC ());
}
