module Kuiper.Poly.MatMul.Tiled.SHMem

#lang-pulse

open Kuiper
module M4  = Kuiper.Matrix4
module MS = Kuiper.Spec.MatMul
module SZ = FStar.SizeT
open Kuiper.EMatrix4
open Kuiper.Matrix.Reprs.Type

type valid_tile = tile:szp{tile * tile <= max_threads}

open Kuiper.Matrix4 {
  gpu_matrix as gpu_matrix4,
  gpu_matrix_pts_to as m4_pts_to,
  gpu_matrix_pts_to_cell as m4_pts_to_cell,
  mlayout4,
  clayout4
}

(* without shmem ownership *)
unfold
let kpre1
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
  (ar : gpu_array et (2sz *^ tile *^ tile))
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  kpre1 tile gA gB gC eA eB f bid tid **
  gpu_pts_to_array1 ar tid **
  gpu_pts_to_array1 ar (tid + tile *^ tile) **
  shmem_tok ar

(* NO FUNCTIONAL SPEC RIGHT NOW *)
unfold
let kpost1
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
  (ar : gpu_array et (2sz *^ tile *^ tile))
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  kpost1 tile gA gB gC eA eB f bid tid **
  gpu_pts_to_array1 ar tid **
  gpu_pts_to_array1 ar (tid + tile *^ tile)

inline_for_extraction noextract
fn fakesync ()
  requires emp
  ensures emp
{
  open Kuiper.Barrier.RPM;
  let p : rpm_t 1 = (fun _ _ _ -> emp);
  assume (mbarrier_tok 1 p 0 0);
  assume (row p 0 0);
  mbarrier_wait ();
  drop_ (mbarrier_tok 1 p 1 0);
  drop_ (col p 1 0);
  ()
}


inline_for_extraction noextract
fn eqplus
  (#et : Type0) {| scalar et |}
  (r : ref et)
  (v : et)
  (#v0 : erased et)
  requires
    r |-> v0
  ensures
    r |-> v0 `add #et` v
{
  let v0 = !r;
  r := v0 `add` v;
}

inline_for_extraction noextract
fn incr
  (#et : Type0) {| scalar et |}
  (r : ref et)
  (#v0 : erased et)
  requires
    r |-> v0
  ensures
    r |-> (v0 `add #et` one)
{
  eqplus r one;
}

inline_for_extraction noextract
fn kernel
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
  (ear : erased (gpu_array et (2sz *^ tile *^ tile)))
  (ebid : enatlt2 mrows mcols)
  (etid : enatlt2 tile  tile)
  ()
  requires
    gpu **
    kpre tile gA gB gC eA eB f ear ebid etid **
    thread_id (tile * tile) etid **
    block_id (mrows * mcols) ebid
  ensures
    gpu **
    kpost tile gA gB gC eA eB f ear ebid etid **
    thread_id (tile * tile) etid **
    block_id (mrows * mcols) ebid
{
  assume (pure False);
  let bid = get_bid (); rewrite each ebid as SZ.v bid;
  let tid = get_tid (); rewrite each etid as SZ.v tid;
  let ar = obtain_shmem ear; rewrite each ear as ar;

  unfold gpu_pts_to_array1 ar tid;
  unfold gpu_pts_to_array1 ar (tid + tile *^ tile);

  let mrow, mcol = s_divmod mcols bid;
  let brow, bcol = s_divmod tile  tid;

  with bi0 bj0 i0 j0 v0.
    rewrite m4_pts_to_cell gC bi0  bj0  i0   j0   v0
         as m4_pts_to_cell gC mrow mcol brow bcol v0;

  let mut sum : et = zero #et #_;
  let mut bk  : sz = 0sz;

  while (let vbk = !bk; SZ.(vbk <^ mshared))
    invariant b.
      exists* (vbk : SZ.t{vbk <= mshared}) sumv.
        pure (b == (SZ.v vbk < mshared)) **
        pts_to bk vbk **
        pts_to #_ #et sum sumv **
        m4_pts_to gA #(f /. mlayout_size lC) eA **
        m4_pts_to gB #(f /. mlayout_size lC) eB **
        (exists* x. gpu_pts_to_slice ar tid (tid + 1) x) **
        (exists* x. gpu_pts_to_slice ar (tid + tile*tile) (tid + tile*tile+1) x) **
        // (exists* arv. gpu_pts_to_array ar arv) **
        gpu
  {
    let vbk = !bk;
    let v1 = M4.gpu_matrix_read gA mrow vbk brow bcol;
    let v2 = M4.gpu_matrix_read gB vbk mcol brow bcol;
    gpu_array_write #_ #_ #(tid) #(tid + 1) ar tid v1;
    gpu_array_write #_ #_ #(tid + tile*tile) #(tid + tile*tile + 1) ar (tid +^ tile *^ tile) v2;

    fakesync ();
    drop_ (exists* x. gpu_pts_to_slice ar (tid) (tid + 1) x);
    drop_ (exists* x. gpu_pts_to_slice ar (tid + tile*tile) (tid + tile*tile+1) x);
    assume (exists* x. gpu_pts_to_slice ar #0.5R 0 (2sz *^ tile *^ tile) x);

    let mut sk : sz = 0sz;
    while (let vsk = !sk; SZ.(vsk <^ tile))
      invariant b.
        exists* (vsk : SZ.t{vsk <= tile}) sumv.
          pure (b == (SZ.v vsk < tile)) **
          pts_to sk vsk **
          pts_to #_ #et sum sumv **
          m4_pts_to gA #(f /. mlayout_size lC) eA **
          m4_pts_to gB #(f /. mlayout_size lC) eB **
          (exists* x. gpu_pts_to_slice ar #0.5R 0 (2sz *^ tile *^ tile) x) **
          gpu
    {
      assume (pure False);
      let vsk = !sk;
      let sidx1 = brow *^ tile +^ vsk;
      assert (pure (SZ.v sidx1 < tile * tile));
      let v1 = gpu_array_read #_ #(2sz *^ tile *^ tile) #0 #(2sz *^ tile *^ tile) ar sidx1;
      let sidx2 = vsk *^ tile +^ bcol +^ tile *^ tile;
      let v2 = gpu_array_read #_ #(2sz *^ tile *^ tile) #0 #(2sz *^ tile *^ tile) ar sidx2;
      let v = v1 `mul` v2;
      eqplus sum v;
      sk := vsk +^ 1sz;
    };

    fakesync ();
    drop_ (exists* x. gpu_pts_to_slice ar #0.5R 0 (2sz *^ tile *^ tile) x);
    assume (exists* x. gpu_pts_to_slice ar (tid) (tid + 1) x);
    assume (exists* x. gpu_pts_to_slice ar (tid + tile*tile) (tid + tile*tile+1) x);

    bk := vbk +^ 1sz;
  };

  let s = !sum;
  M4.gpu_matrix_write_cell gC mrow mcol brow bcol s;

  with v'.
    rewrite
      M4.gpu_matrix_pts_to_cell gC mrow mcol brow bcol v'
    as
      M4.gpu_matrix_pts_to_cell gC
        (ebid / mcols) (ebid % mcols)
        (etid / tile) (etid % tile) v';

  fold gpu_pts_to_array1 ar tid;
  fold gpu_pts_to_array1 ar (tid + tile *^ tile);

  rewrite each SZ.v tid as reveal etid;
  rewrite each ar as ear;

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
      kpre1 tile gA gB gC eA eB 1.0R bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn block_setup
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
  (ar : gpu_array et (2sz *^ tile *^ tile))
  (bid : natlt (mrows *^ mcols))
  ()
  requires
    block_setup_tok (tile *^ tile) **
    (exists* v. gpu_pts_to_array ar #1.0R v) **
    (forall+ (tid : natlt2 tile  tile).
      kpre1 tile gA gB gC eA eB 1.0R bid tid)
  ensures
    block_setup_tok (tile *^ tile) **
    (forall+ (tid : natlt2 tile  tile).
      kpre tile gA gB gC eA eB 1.0R ar bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn block_teardown
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
  (ar : gpu_array et (2sz *^ tile *^ tile))
  (bid : natlt (mrows *^ mcols))
  ()
  requires
    (forall+ (tid : natlt2 tile  tile).
      kpost tile gA gB gC eA eB 1.0R ar bid tid) **
    emp (* frame *)
  ensures
    (exists* v. gpu_pts_to_array ar #1.0R v) **
    (forall+ (tid : natlt2 tile  tile).
      kpost1 tile gA gB gC eA eB 1.0R bid tid)
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
      kpost1 tile gA gB gC eA eB 1.0R bid tid) **
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
  : kernel_desc
      ((gA |-> eA) ** (gB |-> eB) ** (gC |-> eC))
      ((gA |-> eA) ** (gB |-> eB) ** (gC |-> MS.matmul eA eB))
= {
  nblk = mrows *^ mcols;
  nthr = tile *^ tile;

  shmem_type = et;
  shmem_type_is_sized = solve;
  shmem_sz = (2sz *^ tile *^ tile);

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt2 tile tile). kpre1  tile gA gB gC eA eB 1.0R bid tid);
  block_post = (fun bid -> forall+ (tid : natlt2 tile tile). kpost1 tile gA gB gC eA eB 1.0R bid tid);
  setup      = (fun () -> setup tile gA gB gC #eA #eB #eC ());
  teardown   = teardown tile gA gB gC #eA #eB #eC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup    tile gA gB gC #eA #eB #eC;
  block_teardown = block_teardown tile gA gB gC #eA #eB #eC;

  kpre      = kpre  tile gA gB gC eA eB 1.0R;
  kpost     = kpost tile gA gB gC eA eB 1.0R;

  f = kernel tile #et #_ #mrows #mshared #mcols gA gB gC #eA #eB #1.0R;
}

inline_for_extraction noextract
fn matmul_gpu
  (tile : szp)
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
