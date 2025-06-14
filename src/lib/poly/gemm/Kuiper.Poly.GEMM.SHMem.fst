module Kuiper.Poly.GEMM.SHMem

#lang-pulse

open Kuiper
open Kuiper.EMatrix4
open Kuiper.Matrix.Reprs.Type
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.View.TwoTiles { aview_2tile2, mkAIdx, mkCIdx }

open Kuiper.Matrix4 {
  gpu_matrix as gpu_matrix4,
  gpu_matrix_pts_to as m4_pts_to,
  gpu_matrix_pts_to_cell as m4_pts_to_cell,
  mlayout4,
  clayout4
}

module M4 = Kuiper.Matrix4
module MS = Kuiper.Spec.GEMM
module SZ = FStar.SizeT
module B = Kuiper.Barrier

open Kuiper.EMatrix { ematrix }
open Kuiper.ArrayView {
  varray,
  varray_pts_to,
  varray_pts_to_cell
}
module AV = Kuiper.ArrayView

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
  (#et : Type0)
  (tile : valid_tile)
  (ar : varray (aview_2tile2 et tile))
  : B.barrier_side (tile *^ tile) =
  fun it tid ->
    if even it
    then (exists* (x : _ & _). ar |-> Fraction (1.0R /. (tile *^ tile)) x)
    else (
      (exists* x. varray_pts_to_cell ar (mkAIdx 0 (tid / tile) (tid % tile)) x) **
      (exists* x. varray_pts_to_cell ar (mkAIdx 1 (tid / tile) (tid % tile)) x)
    )

let barrier_q
  (#et : Type0)
  (tile : valid_tile)
  (ar : varray (aview_2tile2 et tile))
  : B.barrier_side (tile *^ tile) =
  fun it tid -> barrier_p tile ar (it+1) tid (* flip flop *)

(* without shmem ownership *)
unfold
let kpre1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
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
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  (gA |-> Fraction (fA /. mlayout_size lC) eA) **
  (gB |-> Fraction (fB /. mlayout_size lC) eB) **
  (exists* v.
    m4_pts_to_cell gC #1.0R
      (bid / mcols) (bid % mcols)
      (tid / tile) (tid % tile) v)

(* NO FUNCTIONAL SPEC RIGHT NOW *)
unfold
let kpost1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
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
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  (gA |-> Fraction (fA /. mlayout_size lC) eA) **
  (gB |-> Fraction (fB /. mlayout_size lC) eB) **
  (exists* v.
    m4_pts_to_cell gC #1.0R
      (bid / mcols) (bid % mcols)
      (tid / tile) (tid % tile) v)

let barrier_tok
  (#et : Type0)
  (tile : valid_tile)
  (ar: gpu_array et (2sz *^ tile *^ tile))
  (it : nat)
  (tid : natlt (tile *^ tile))
  : slprop
  =
  B.barrier_tok (barrier_p tile (AV.from_array (aview_2tile2 et tile) ar))
                (barrier_q tile (AV.from_array (aview_2tile2 et tile) ar)) it tid

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (comb : binop et)
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
  (fA fB : perm)
  (ar : gpu_array et (2sz *^ tile *^ tile))
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  kpre1 comb tile gA gB gC eA eB fA fB bid tid **
  (exists* x. gpu_pts_to_slice ar #(1.0R /. (tile *^ tile)) 0 (2sz *^ tile *^ tile) x) **
  barrier_tok tile ar 0 tid **
  shmem_tok ar


unfold
let kpost
  (#et : Type0) {| scalar et |}
  (comb : binop et)
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
  (fA fB : perm)
  (ar : gpu_array et (2sz *^ tile *^ tile))
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  kpost1 comb tile gA gB gC eA eB fA fB bid tid **
  (exists* x. gpu_pts_to_slice ar #(1.0R /. (tile *^ tile)) 0 (2sz *^ tile *^ tile) x) **
  barrier_tok tile ar (2 * mshared) tid

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
fn subproduct
  (#et : Type0) {| scalar et |}
  (tile : valid_tile)
  (acc : ref et)
  (ar : varray (aview_2tile2 et tile))
  (i j : szlt tile)
  (#acc0 : erased et)
  (#ar0 : erased _)
  (#f : perm)
  preserves
    gpu **
    (ar |-> Fraction f ar0)
  requires
    acc |-> acc0
  ensures
    exists* acc'.
      acc |-> acc'
{
  let mut sk : sz = 0sz;
  while (let vsk = !sk; SZ.(vsk <^ tile))
    invariant b.
      exists* (vsk : SZ.t{vsk <= tile}) accv.
        pure (b == (SZ.v vsk < tile)) **
        (sk |-> vsk) **
        (acc |-> accv) **
        (ar |-> Fraction f ar0) **
        gpu
  {
    let vsk = !sk;
    let v1 = AV.varray_read ar (mkCIdx #tile 0sz i vsk);
    let v2 = AV.varray_read ar (mkCIdx #tile 1sz vsk j);

    let v = v1 `mul` v2;
    eqplus acc v;
    sk := vsk +^ 1sz;
  };
}

(* TODO: Find out where the time is going when checking this function,
it feels a lot slower than the others. *)
inline_for_extraction noextract
fn kf
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
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
  (#fA #fB : perm)
  (ear : erased (gpu_array et (2sz *^ tile *^ tile)))
  (bid : szlt2 mrows mcols)
  (tid : szlt2 tile  tile)
  ()
  requires
    gpu **
    kpre comb tile gA gB gC eA eB fA fB ear bid tid **
    thread_id (tile * tile) tid **
    block_id (mrows * mcols) bid
  ensures
    gpu **
    kpost comb tile gA gB gC eA eB fA fB ear bid tid **
    thread_id (tile * tile) tid **
    block_id (mrows * mcols) bid
{
  gpu_pts_to_ref ear;

  let ar0 = obtain_shmem ear; rewrite each ear as ar0;
  unfold barrier_tok tile ar0 0 tid;

  // Does not work:
  // rewrite each AV.from_array #et #(2sz *^ tile *^ tile) #(ematrix et tile tile & ematrix et tile tile) ar (aview_2tile2 et tile)
  //           as ar;
  AV.varray_abs' (aview_2tile2 et tile) ar0;
  let ar = AV.from_array (aview_2tile2 et tile <: erased _) ar0;
  rewrite
    B.barrier_tok (barrier_p tile (AV.from_array (aview_2tile2 et tile) ar0))
                  (barrier_q tile (AV.from_array (aview_2tile2 et tile) ar0)) 0 tid
  as
    B.barrier_tok (barrier_p tile ar)
                  (barrier_q tile ar) 0 tid;

  with x0.
    rewrite
      AV.varray_pts_to (AV.from_array (aview_2tile2 et tile) ar0) #(1.0R /. (tile *^ tile)) x0
    as
      AV.varray_pts_to ar #(1.0R /. (tile *^ tile)) x0;

  let mrow, mcol = s_divmod mcols bid;
  let brow, bcol = s_divmod tile  tid;
  assert (pure (SZ.v brow == tid / tile));
  assert (pure (SZ.v bcol == tid % tile));
  assert (pure (brow < tile));
  assert (pure (bcol < tile));

  with bi0 bj0 i0 j0 v0.
    rewrite m4_pts_to_cell gC bi0  bj0  i0   j0   v0
         as m4_pts_to_cell gC mrow mcol brow bcol v0;

  let mut sum : et = zero #et #_;
  let mut bk  : sz = 0sz;

  while (let vbk = !bk; SZ.(vbk <^ mshared))
    invariant b.
      exists* (vbk : SZ.t{vbk <= mshared}) sumv.
        pure (b == (SZ.v vbk < mshared)) **
        (bk |-> vbk) **
        (sum |-> sumv) **
        (gA |-> Fraction (fA /. mlayout_size lC) eA) **
        (gB |-> Fraction (fB /. mlayout_size lC) eB) **
        (exists* x. varray_pts_to ar #(1.0R /. (tile *^ tile)) x) **
        B.barrier_tok (barrier_p tile ar) (barrier_q tile ar) (2 * vbk) tid **
        gpu
  {
    let vbk = !bk;
    let v1 = M4.gpu_matrix_read gA mrow vbk brow bcol;
    let v2 = M4.gpu_matrix_read gB vbk mcol brow bcol;

    (* This assert should not be needed. I don't know what effect it even has. *)
    assert B.barrier_tok (barrier_p tile ar) (barrier_q tile ar) (2 * vbk) tid;
    even_2x vbk;
    assert (pure (even (2 * vbk)));
    rewrite (exists* (x : _ & _). ar |-> Fraction (1.0R /. (tile *^ tile)) x)
         as (barrier_p tile ar (2 * vbk) tid);
    B.barrier_wait ();
    rewrite (barrier_q tile ar (2 * vbk) tid)
         as (exists* x. varray_pts_to_cell ar (mkAIdx 0 (tid / tile) (tid % tile)) x) **
            (exists* x. varray_pts_to_cell ar (mkAIdx 1 (tid / tile) (tid % tile)) x);

    AV.varray_write_cell' ar (mkCIdx 0sz brow bcol) (mkAIdx 0 (tid / tile) (tid % tile)) v1;
    AV.varray_write_cell' ar (mkCIdx 1sz brow bcol) (mkAIdx 1 (tid / tile) (tid % tile)) v2;

    assert (B.barrier_tok (barrier_p tile ar) (barrier_q tile ar) (2 * vbk + 1) tid);
    odd_2x1 vbk;
    assert (pure (odd (2 * vbk + 1)));
    rewrite (exists* x. varray_pts_to_cell ar (mkAIdx 0 (tid / tile) (tid % tile)) x) **
            (exists* x. varray_pts_to_cell ar (mkAIdx 1 (tid / tile) (tid % tile)) x)
         as (barrier_p tile ar (2 * vbk + 1) tid);
    B.barrier_wait ();
    even_2x (vbk + 1);
    (* sigh *)
    assert (pure (2 * (vbk + 1) == 2 * vbk + 2));
    assert (pure (even (2 * vbk + 2)));
    rewrite (barrier_q tile ar (2 * vbk + 1) tid)
         as (exists* (x : _ & _). ar |-> Fraction (1.0R /. (tile *^ tile)) x);

    (* At this point the SHMem cache is filled with the submatrices
       and we have RO permission to it. Compute product for our cell in
       the tile and add to sum. *)

    subproduct tile sum ar brow bcol;

    (* Move to next tile *)
    bk := vbk +^ 1sz;
  };

  let s = !sum;
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

  AV.varray_concr ar;
  rewrite each SZ.v tid as reveal tid;
  rewrite each ar as AV.from_array (aview_2tile2 et tile) ar0;
  with x1.
    rewrite
      gpu_pts_to_array (AV.core (AV.from_array (aview_2tile2 et tile) ar0)) #(1.0R /. (tile *^ tile)) x1
    as
      ar0 |-> Fraction (1.0R /. (tile *^ tile)) x1;
  rewrite each ar0 as ear;
  fold barrier_tok tile ear (2 * mshared) tid;

  ()
}

ghost
fn setup
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : SZ.t)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (#fA : perm)
  (gB : gpu_matrix4 et lB)
  (#fB : perm)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared tile tile)
  (#eB : ematrix4 et mshared mcols   tile tile)
  (#eC : ematrix4 et mrows   mcols   tile tile)
  ()
  requires
    (gA |-> Fraction fA eA) **
    (gB |-> Fraction fB eB) **
    (gC |-> eC)
  ensures
    (forall+ (bid : natlt2 mrows mcols)
             (tid : natlt2 tile  tile).
      kpre1 comb tile gA gB gC eA eB fA fB bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn block_setup
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : SZ.t)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (#fA : perm)
  (gB : gpu_matrix4 et lB)
  (#fB : perm)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared tile tile)
  (#eB : ematrix4 et mshared mcols   tile tile)
  (#eC : ematrix4 et mrows   mcols   tile tile)
  (ar : gpu_array et (2sz *^ tile *^ tile))
  (bid : natlt (mrows *^ mcols))
  ()
  requires
    block_setup_tok (tile *^ tile) **
    (exists* v. ar |-> v) **
    (forall+ (tid : natlt2 tile  tile).
      kpre1 comb tile gA gB gC eA eB fA fB bid tid)
  ensures
    block_setup_tok (tile *^ tile) **
    (forall+ (tid : natlt2 tile  tile).
      kpre comb tile gA gB gC eA eB fA fB ar bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn block_teardown
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : SZ.t)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (#fA : perm)
  (gB : gpu_matrix4 et lB)
  (#fB : perm)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared tile tile)
  (#eB : ematrix4 et mshared mcols   tile tile)
  (#eC : ematrix4 et mrows   mcols   tile tile)
  (ar : gpu_array et (2sz *^ tile *^ tile))
  (bid : natlt (mrows *^ mcols))
  ()
  requires
    (forall+ (tid : natlt2 tile  tile).
      kpost comb tile gA gB gC eA eB fA fB ar bid tid) **
    emp (* frame *)
  ensures
    (exists* v. ar |-> v) **
    (forall+ (tid : natlt2 tile  tile).
      kpost1 comb tile gA gB gC eA eB fA fB bid tid)
{
  admit();
}

ghost
fn teardown
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : SZ.t)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (#fA : perm)
  (gB : gpu_matrix4 et lB)
  (#fB : perm)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared tile tile)
  (#eB : ematrix4 et mshared mcols   tile tile)
  (#eC : ematrix4 et mrows   mcols   tile tile)
  ()
  requires
    (forall+ (bid : natlt2 mrows mcols)
             (tid : natlt2 tile  tile).
      kpost1 comb tile gA gB gC eA eB fA fB bid tid) **
    emp (* frame *)
  ensures
    (gA |-> Fraction fA eA) **
    (gB |-> Fraction fB eB) **
    (gC |-> MS.mmcomb comb eC eA eB)
{
  admit();
}

inline_for_extraction noextract
let mk_kernel
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : SZ.t)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (#fA : perm)
  (gB : gpu_matrix4 et lB)
  (#fB : perm)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared tile tile)
  (#eB : ematrix4 et mshared mcols   tile tile)
  (#eC : ematrix4 et mrows   mcols   tile tile)
  (_ : squash (mrows * mcols <= max_blocks
               /\ tile * tile <= max_threads))
  : kernel_desc
      ((gA |-> Fraction fA eA) ** (gB |-> Fraction fB eB) ** (gC |-> eC))
      ((gA |-> Fraction fA eA) ** (gB |-> Fraction fB eB) ** (gC |-> MS.mmcomb comb eC eA eB))
= {
  nblk = mrows *^ mcols;
  nthr = tile *^ tile;

  shmem_type = et;
  shmem_type_is_sized = solve;
  shmem_sz = (2sz *^ tile *^ tile);

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt2 tile tile). kpre1  comb tile gA gB gC eA eB fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt2 tile tile). kpost1 comb tile gA gB gC eA eB fA fB bid tid);
  setup      = setup    tile comb gA gB gC #eA #eB #eC;
  teardown   = teardown tile comb gA gB gC #eA #eB #eC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup    tile comb gA #fA gB #fB gC #eA #eB #eC;
  block_teardown = block_teardown tile comb gA #fA gB #fB gC #eA #eB #eC;

  kpre      = kpre  comb tile gA gB gC eA eB fA fB;
  kpost     = kpost comb tile gA gB gC eA eB fA fB;

  f = kf tile #et #_ comb #mrows #mshared #mcols gA gB gC #eA #eB;
}

inline_for_extraction noextract
fn mmcomb_gpu
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (lA : mlayout4 mrows   mshared tile tile)
  (lB : mlayout4 mshared mcols   tile tile)
  (lC : mlayout4 mrows   mcols   tile tile)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (#fA : perm)
  (gB : gpu_matrix4 et lB)
  (#fB : perm)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared tile tile)
  (#eB : ematrix4 et mshared mcols   tile tile)
  (#eC : ematrix4 et mrows   mcols   tile tile)
  preserves
    cpu **
    (gA |-> Fraction fA eA) **
    (gB |-> Fraction fB eB)
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
