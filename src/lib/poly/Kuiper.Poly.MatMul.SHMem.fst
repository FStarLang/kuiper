module Kuiper.Poly.MatMul.SHMem

#lang-pulse

open Kuiper
open Kuiper.EMatrix4
open Kuiper.Matrix.Reprs.Type
open Kuiper.Math { even, odd, even_2x, odd_2x1 }

open Kuiper.Matrix4 {
  gpu_matrix as gpu_matrix4,
  gpu_matrix_pts_to as m4_pts_to,
  gpu_matrix_pts_to_cell as m4_pts_to_cell,
  mlayout4,
  clayout4
}

module M4  = Kuiper.Matrix4
module MS = Kuiper.Spec.MatMul
module SZ = FStar.SizeT
module B = Kuiper.Barrier

(* The barrier flip-flops between an initial state
where every threads shares all of the array, and
a second state where each thread owns two cells
of the array, related to their tid.

So in even steps, they give their shared ownership,
and receive their cells. (p)
*)

(* To verify functional correctness: the existentials here should be made
precise, and parametrize this over the starting input matrices. *)
let barrier_p
  (#et : Type0) {| scalar et |}
  (tile : valid_tile)
  (ar : gpu_array et (2sz *^ tile *^ tile))
  : B.barrier_side (tile *^ tile) =
  fun it tid ->
    if even it
    then (exists* x. gpu_pts_to_slice ar #(1.0R /. (tile *^ tile)) 0 (2sz *^ tile *^ tile) x)
    else (
      (exists* x. gpu_pts_to_slice ar (tid) (tid + 1) x) **
      (exists* x. gpu_pts_to_slice ar (tid + tile*^tile) (tid + tile*^tile+1) x)
    )

(* Same thing, switching the condition. *)
let barrier_q
  (#et : Type0) {| scalar et |}
  (tile : valid_tile)
  (ar : gpu_array et (2sz *^ tile *^ tile))
  : B.barrier_side (tile *^ tile) =
  fun it tid ->
    if odd it
    then (exists* x. gpu_pts_to_slice ar #(1.0R /. (tile *^ tile)) 0 (2sz *^ tile *^ tile) x)
    else (
      (exists* x. gpu_pts_to_slice ar (tid) (tid + 1) x) **
      (exists* x. gpu_pts_to_slice ar (tid + tile*^tile) (tid + tile*^tile+1) x)
    )

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
  (exists* x. gpu_pts_to_slice ar #(1.0R /. (tile *^ tile)) 0 (2sz *^ tile *^ tile) x) **
  B.barrier_tok (barrier_p tile ar) (barrier_q tile ar) 0 tid **
  shmem_tok ar


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
  (exists* x. gpu_pts_to_slice ar #(1.0R /. (tile *^ tile)) 0 (2sz *^ tile *^ tile) x) **
  B.barrier_tok (barrier_p tile ar) (barrier_q tile ar) (2 * mshared) tid

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
  let bid = get_bid (); rewrite each ebid as SZ.v bid;
  let tid = get_tid (); rewrite each etid as SZ.v tid;
  let ar = obtain_shmem ear; rewrite each ear as ar;

  let mrow, mcol = s_divmod mcols bid;
  let brow, bcol = s_divmod tile  tid;
  assert (pure (brow < tile));

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
        (exists* x. gpu_pts_to_slice ar #(1.0R /. (tile *^ tile)) 0 (2sz *^ tile *^ tile) x) **
        B.barrier_tok (barrier_p tile ar) (barrier_q tile ar) (2 * vbk) tid **
        gpu
  {
    let vbk = !bk;
    let v1 = M4.gpu_matrix_read gA mrow vbk brow bcol;
    let v2 = M4.gpu_matrix_read gB vbk mcol brow bcol;

    (* This assert should not be needed. *)
    assert B.barrier_tok (barrier_p tile ar) (barrier_q tile ar) (2 * vbk) tid;
    even_2x vbk;
    assert (pure (even (2 * vbk)));
    rewrite (exists* x. gpu_pts_to_slice ar #(1.0R /. (tile *^ tile)) 0 (2sz *^ tile *^ tile) x)
         as (barrier_p tile ar (2 * vbk) tid);
    B.barrier_wait ();
    rewrite (barrier_q tile ar (2 * vbk) tid)
         as ((exists* x. gpu_pts_to_slice ar (tid) (tid + 1) x) **
             (exists* x. gpu_pts_to_slice ar (tid + tile*^tile) (tid + tile*^tile+1) x));

    gpu_array_write #_ #_ #(tid) #(tid + 1) ar tid v1;
    gpu_array_write #_ #_ #(tid + tile*^tile) #(tid + tile*^tile + 1) ar (tid +^ tile *^ tile) v2;

    assert (B.barrier_tok (barrier_p tile ar) (barrier_q tile ar) (2 * vbk + 1) tid);
    odd_2x1 vbk;
    assert (pure (odd (2 * vbk + 1)));
    rewrite ((exists* x. gpu_pts_to_slice ar (tid) (tid + 1) x) **
             (exists* x. gpu_pts_to_slice ar (tid + tile*^tile) (tid + tile*^tile+1) x))
         as (barrier_p tile ar (2 * vbk + 1) tid);
    B.barrier_wait ();
    rewrite (barrier_q tile ar (2 * vbk + 1) tid)
         as (exists* x. gpu_pts_to_slice ar #(1.0R /. (tile *^ tile)) 0 (2sz *^ tile *^ tile) x);

    let mut sk : sz = 0sz;
    while (let vsk = !sk; SZ.(vsk <^ tile))
      invariant b.
        exists* (vsk : SZ.t{vsk <= tile}) sumv.
          pure (b == (SZ.v vsk < tile)) **
          pts_to sk vsk **
          pts_to #_ #et sum sumv **
          m4_pts_to gA #(f /. mlayout_size lC) eA **
          m4_pts_to gB #(f /. mlayout_size lC) eB **
          (exists* x. gpu_pts_to_slice ar #(1.0R /. (tile *^ tile)) 0 (2sz *^ tile *^ tile) x) **
          gpu
    {
      let vsk = !sk;
      (* Z3 Takes some convincing here... *)
      assert (pure (brow < tile));
      assert (pure (brow <= tile - 1));
      assert (pure (brow *^ tile <= tile*tile - tile));
      let sidx1 = brow *^ tile +^ vsk;
      assert (pure (brow *^ tile +^ vsk <= tile*tile - 1));
      assert (pure (SZ.v sidx1 < tile * tile));
      let v1 = gpu_array_read #_ #(2sz *^ tile *^ tile) #0 #(2sz *^ tile *^ tile) ar sidx1;
      let sidx2 = vsk *^ tile +^ bcol +^ tile *^ tile;
      assert (pure (SZ.v sidx2 < 2sz *^ tile *^ tile));
      let v2 = gpu_array_read #_ #(2sz *^ tile *^ tile) #0 #(2sz *^ tile *^ tile) ar sidx2;
      let v = v1 `mul` v2;
      eqplus sum v;
      sk := vsk +^ 1sz;
    };

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

  f = kf tile #et #_ #mrows #mshared #mcols gA gB gC #eA #eB #1.0R;
}

inline_for_extraction noextract
fn matmul_gpu
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
  dassert (tile `SZ.gt` 0sz);
  launch_sync (mk_kernel tile gA gB gC ());
}
