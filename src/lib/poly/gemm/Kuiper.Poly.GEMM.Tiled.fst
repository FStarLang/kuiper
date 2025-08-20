module Kuiper.Poly.GEMM.Tiled

#lang-pulse

open Kuiper
module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Poly.GEMM.Util
module SZ = FStar.SizeT
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix
open Kuiper.EMatrix
open Kuiper.Matrix.Tiling

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA : ematrix et (mrows   * tile) (mshared * tile))
  (eB : ematrix et (mshared * tile) (mcols   * tile))
  (eC : ematrix et (mrows   * tile) (mcols   * tile))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  (gA |-> Frac fA eA) **
  (gB |-> Frac fB eB) **
  (exists* v.
    gpu_matrix_pts_to_cell
      (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      (tid / tile)
      (tid % tile)
      v)
  **
  emp

(* NO FUNCTIONAL SPEC RIGHT NOW *)
unfold
let kpost
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA : ematrix et (mrows   * tile) (mshared * tile))
  (eB : ematrix et (mshared * tile) (mcols   * tile))
  (eC : ematrix et (mrows   * tile) (mcols   * tile))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  kpre comb tile gA gB gC eA eB eC fA fB bid tid

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, cC : clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA eB eC : ematrix et _ _)
  (fA fB : perm)
  (bid : szlt (mrows * mcols))
  (tid : szlt (tile * tile))
  ()
  requires
    gpu **
    kpre comb tile gA gB gC eA eB eC fA fB bid tid **
    thread_id (tile * tile) tid **
    block_id (mrows * mcols) bid
  ensures
    gpu **
    kpost comb tile gA gB gC eA eB eC fA fB bid tid **
    thread_id (tile * tile) tid **
    block_id (mrows * mcols) bid
{
  let mrow, mcol = s_divmod mcols bid;
  let brow, bcol = s_divmod tile  tid;

  with i0 j0 v0.
    rewrite
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) #1.0R i0 j0 v0
    as
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) #1.0R brow bcol v0
    ;

  assert (pure (mrow < mrows));
  assert (pure (mcol < mcols));
  assert (pure (brow < tile));
  assert (pure (bcol < tile));

  let gTile = gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols);
  assert (rewrites_to gTile (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)));

  let s = MU.matmul_tiled_dotprod gA gB mrow mcol brow bcol;
  let v0 = gpu_matrix_read_cell gTile brow bcol;
  let v1 = comb v0 s;
  gpu_matrix_write_cell gTile brow bcol v1;

  rewrite
    gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      brow bcol v1
  as
    gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      (tid / tile) (tid % tile) v1;

  ()
}

ghost
fn setup
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA #eB #eC : ematrix _ _ _)
  ()
  requires
    (gA |-> Frac fA eA) **
    (gB |-> Frac fB eB) **
    (gC |-> eC)
  ensures
    (forall+ (bid : natlt2 mrows mcols)
            (tid : natlt2 tile  tile).
      kpre comb tile gA gB gC eA eB eC fA fB bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn teardown
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA #eB #eC : ematrix _ _ _)
  ()
  requires
    (forall+ (bid : natlt2 mrows mcols)
            (tid : natlt2 tile  tile).
      kpost comb tile gA gB gC eA eB eC fA fB bid tid) **
    emp (* frame *)
  ensures
    (gA |-> Frac fA eA) **
    (gB |-> Frac fB eB) **
    (gC |-> MS.mmcomb comb eC eA eB)
{
  admit();
}

inline_for_extraction noextract
let mk_kernel
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA #eB #eC : ematrix _ _ _)
  (_ : squash (mrows * mcols <= max_blocks
               /\ tile * tile <= max_threads))
  : kernel_desc_m_n
      ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) ** (gC |-> eC))
      ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) ** (gC |-> MS.mmcomb comb eC eA eB))
= {
  nblk = mrows *^ mcols;
  nthr = tile *^ tile;

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt2 tile tile). kpre  comb tile gA gB gC eA eB eC fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt2 tile tile). kpost comb tile gA gB gC eA eB eC fA fB bid tid);
  setup     = setup    tile comb gA gB gC #eA #eB #eC;
  teardown  = teardown tile comb gA gB gC #eA #eB #eC;

  block_frame    = (fun _bid -> emp);
  block_setup    = (fun bid -> Kuiper.Frame.emp_intro_r2 ());
  block_teardown = (fun bid -> Kuiper.Frame.emp_elim_r ());

  kpre      = kpre  comb tile gA gB gC eA eB eC fA fB;
  kpost     = kpost comb tile gA gB gC eA eB eC fA fB;

  f = kf #et #_ comb #mrows #mshared #mcols tile gA gB gC eA eB eC fA fB;
}

inline_for_extraction noextract
fn mmcomb_gpu
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (lA : mlayout (mrows   * tile) (mshared * tile))
  (lB : mlayout (mshared * tile) (mcols   * tile))
  (lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA #eB #eC : ematrix _ _ _)
  preserves
    cpu **
    (gA |-> Frac fA eA) **
    (gB |-> Frac fB eB)
  requires
    pure (mrows * mcols <= max_blocks) **
    pure (tile * tile <= max_threads) **
    (gC |-> eC)
  ensures
    gC |-> MS.mmcomb comb eC eA eB
{
  dassert (tile `SZ.gt` 0sz);
  launch_sync (mk_kernel tile comb gA gB gC ());
}
