module Kuiper.Poly.MatMul.Tiled

#lang-pulse

open Kuiper
module M   = Kuiper.Matrix
module M4  = Kuiper.Matrix4
module MS = Kuiper.Spec.MatMul
module MU = Kuiper.Poly.MatMul.Util
module SZ = FStar.SizeT
module EM = Kuiper.EMatrix
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
  (ebid : enatlt2 mrows mcols)
  (etid : enatlt2 tile  tile)
  ()
  requires
    gpu **
    kpre tile gA gB gC eA eB f ebid etid **
    thread_id (tile * tile) etid **
    block_id (mrows * mcols) ebid
  ensures
    gpu **
    kpost tile gA gB gC eA eB f ebid etid **
    thread_id (tile * tile) etid **
    block_id (mrows * mcols) ebid
{
  let bid = get_bid (); rewrite each ebid as SZ.v bid;
  let tid = get_tid (); rewrite each etid as SZ.v tid;

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
  (tile : valid_tile)
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
    (forall+ (bid : natlt2 mrows mcols)
            (tid : natlt2 tile  tile).
      kpre tile gA gB gC eA eB 1.0R bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn teardown
  (tile : valid_tile)
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
    (forall+ (bid : natlt2 mrows mcols)
            (tid : natlt2 tile  tile).
      kpost tile gA gB gC eA eB 1.0R bid tid) **
    emp (* frame *)
  ensures
    (gA |-> eA) **
    (gB |-> eB) **
    (gC |-> MS.matmul eA eB)
{
  admit();
}

inline_for_extraction noextract
let mk_kernel
  (tile : valid_tile)
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
  : kernel_desc_m_n
      ((gA |-> eA) ** (gB |-> eB) ** (gC |-> eC))
      ((gA |-> eA) ** (gB |-> eB) ** (gC |-> MS.matmul eA eB))
= {
  nblk = mrows *^ mcols;
  nthr = tile *^ tile;

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt2 tile tile). kpre  tile gA gB gC eA eB 1.0R bid tid);
  block_post = (fun bid -> forall+ (tid : natlt2 tile tile). kpost tile gA gB gC eA eB 1.0R bid tid);
  setup     = setup tile gA gB gC #eA #eB #eC;
  teardown  = teardown tile gA gB gC #eA #eB #eC;

  block_frame    = (fun _bid -> emp);
  block_setup    = (fun bid -> Kuiper.Frame.emp_intro_r ());
  block_teardown = (fun bid -> Kuiper.Frame.emp_elim_r ());

  kpre      = kpre  tile gA gB gC eA eB 1.0R;
  kpost     = kpost tile gA gB gC eA eB 1.0R;

  f = kf tile #et #_ #mrows #mshared #mcols gA gB gC #eA #eB #1.0R;
}

inline_for_extraction noextract
fn matmul_gpu_tiled
  (tile : valid_tile)
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
  launch_sync (mk_kernel tile gA gB gC ());
}

(* This will dinamically abort if the dimensions (rows/shared/cols) are not
   multiples of tile. *)
inline_for_extraction noextract
fn matmul_gpu
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| cA : clayout lA |}
  {| cB : clayout lB |}
  {| cC : clayout lC |}
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  (#eA : EM.ematrix et rows shared)
  (#eB : EM.ematrix et shared cols)
  (#eC : EM.ematrix et rows cols)
  preserves
    cpu **
    (gA |-> eA) **
    (gB |-> eB)
  requires
    pure (rows * cols <= max_blocks) **
    (gC |-> eC)
  ensures
    gC |-> MS.matmul eA eB
{
  dguard (rows   %^ tile = 0sz);
  dguard (shared %^ tile = 0sz);
  dguard (cols   %^ tile = 0sz);
  let mrows   = rows   /^ tile;
  let mshared = shared /^ tile;
  let mcols   = cols   /^ tile;

  let lA4 : mlayout4 mrows  mshared tile tile = lA;
  let gA4 = M4.from_matrix2 (SZ.v tile) (SZ.v mrows) (SZ.v mshared) #_ #_ #lA4 gA;
  let lB4 : mlayout4 mshared mcols tile tile = lB;
  let gB4 = M4.from_matrix2 (SZ.v tile) (SZ.v mshared) (SZ.v mcols) #_ #_ #lB4 gB;
  let lC4 : mlayout4 mrows  mcols tile tile = lC;
  let gC4 = M4.from_matrix2 (SZ.v tile) (SZ.v mrows) (SZ.v mcols) #_ #_ #lC4 gC;
  matmul_gpu_tiled tile
    #et #_ #mrows #mshared #mcols
    lA4 lB4 lC4
    #(M4.clayout4_from_clayout tile cA)
    #(M4.clayout4_from_clayout tile cB)
    #(M4.clayout4_from_clayout tile cC)
    gA4 gB4 gC4;

  let gA' = M4.to_matrix2 (SZ.v tile) (SZ.v mrows) (SZ.v mshared) #_ #_ #lA4 gA4;
  M.core_match gA gA'; rewrite each gA' as gA;
  let gB' = M4.to_matrix2 (SZ.v tile) (SZ.v mshared) (SZ.v mcols) #_ #_ #lB4 gB4;
  M.core_match gB gB'; rewrite each gB' as gB;
  let gC' = M4.to_matrix2 (SZ.v tile) (SZ.v mrows) (SZ.v mcols) #_ #_ #lC4 gC4;
  M.core_match gC gC'; rewrite each gC' as gC;
  ()
}
