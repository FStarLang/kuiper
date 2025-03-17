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

open Kuiper.EMatrix { ematrix }
open Kuiper.ArrayView {
  aview,
  cview,
  varray,
  varray_pts_to,
  varray_pts_to_cell
}
module AV = Kuiper.ArrayView

(* these types help avoid bad inference, and mismatches in the implicit arguments
of tuple constructors. *)
type ait (tile : erased nat) = | AIdx : natlt 2 -> natlt tile -> natlt tile -> ait tile
inline_for_extraction noextract
type cit (tile : erased nat) = | CIdx : szlt 2  -> szlt tile  -> szlt tile  -> cit tile

// FIXME: the erased here should not be needed, this is an erasable type,
// but again inference and re-checking is hurt badly without it.
let aview_2tile2
  (et : Type0)
  (tile : valid_tile)
  : aview et (2sz *^ tile *^ tile) (ematrix et tile tile & ematrix et tile tile)
= {
  it = ait tile; // natlt 2 & natlt tile & natlt tile;
  igm = magic ();
  ibij = {
    ff = (fun (i : ait tile) ->
      [@@inline_let] let AIdx i j k = i in
      (i * SZ.v tile * SZ.v tile + j * SZ.v tile + k) <: natlt (2 * tile * tile));
    gg = (fun (n : natlt (2 * tile * tile)) ->
      [@@inline_let] let i, n = divmod (SZ.v tile * SZ.v tile) n in
      [@@inline_let] let j, k = divmod (SZ.v tile) n in
      AIdx i j k);
    ff_gg = (fun _ -> ());
    gg_ff = (fun _ -> admit());
  };
}

inline_for_extraction noextract
instance cview_2tile2
  (et : Type0)
  (tile : valid_tile)
  : cview (aview_2tile2 et tile)
= {
  lenfits = ();
  cit = cit tile; // szlt 2 & szlt tile & szlt tile;
  cibij = {
    ff = (fun cidx ->
      [@@inline_let] let CIdx i j k = cidx in
      (i *^ tile *^ tile +^ j *^ tile +^ k) <: szlt (2sz *^ tile *^ tile));
    gg = (fun (n : szlt (2sz *^ tile *^ tile))  ->
      [@@inline_let] let i, n = s_divmod (tile *^ tile) n in
      [@@inline_let] let j, k = s_divmod tile n in
      CIdx i j k <: cit tile);
    ff_gg = (fun _ -> ());
    gg_ff = (fun _ -> admit());
  }
}

let l1
  (#et : Type0)
  (tile : valid_tile)
  (i : szlt 2)
  (j : szlt tile)
  (k : szlt tile)
  : Lemma (AV.cit_to_it (aview_2tile2 et tile) (CIdx i j k) == AIdx i j k)
  = ()

// let l2
//   (#et : Type0)
//   (tile : valid_tile)
//   (i : natlt 2)
//   (j : natlt tile)
//   (k : natlt tile)
//   : Lemma (AV.cit_to_it (aview_2tile2 et tile) (CIdx i j k) == AIdx i j k)
//   = ()


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
    then (exists* x. varray_pts_to ar #(1.0R /. (tile *^ tile)) x)
    else (
      (exists* x. varray_pts_to_cell ar (AIdx 0 (tid / tile) (tid % tile)) x) **
      (exists* x. varray_pts_to_cell ar (AIdx 1 (tid / tile) (tid % tile)) x)
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
  barrier_tok tile ar 0 tid **
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

(* TODO: This function is REALLY slow to check since we are using
the shared memory array as a flat buffer, and computing compound
indices into it. We should instead split in two, and use
two adjacent Matrix2 views on it, which would eliminate all of this.

This is now done, but this function is still very slow. I tried to remove
the (tid/tile) (tid%tile) mentions in the context replacing them with
brow,bow, but that didn't work. It is somewhat faster than before.. and
seems a bit more stable.

The even/odd reasoning is also annoying, it stumps Z3. *)
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
  gpu_pts_to_ref ear;
  let bid = get_bid (); rewrite each reveal ebid as SZ.v bid;
  let tid = get_tid (); rewrite each reveal etid as SZ.v tid;

  let ar0 = obtain_shmem ear; rewrite each ear as ar0;
  unfold barrier_tok tile ar0 0 tid;

  // Does not work:
  // rewrite each AV.from_array #et #(2sz *^ tile *^ tile) #(ematrix et tile tile & ematrix et tile tile) ear (aview_2tile2 et tile)
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
        pts_to bk vbk **
        pts_to #_ #et sum sumv **
        m4_pts_to gA #(f /. mlayout_size lC) eA **
        m4_pts_to gB #(f /. mlayout_size lC) eB **
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
    rewrite (exists* x. AV.varray_pts_to ar #(1.0R /. (tile *^ tile)) x)
         as (barrier_p tile ar (2 * vbk) tid);
    B.barrier_wait ();
    rewrite (barrier_q tile ar (2 * vbk) tid)
         as (exists* x. varray_pts_to_cell ar (AIdx 0 (tid / tile) (tid % tile)) x) **
            (exists* x. varray_pts_to_cell ar (AIdx 1 (tid / tile) (tid % tile)) x);

    (* tedious *)
    with x.
      rewrite varray_pts_to_cell ar (AIdx 0 (tid / tile) (tid % tile)) x
           as varray_pts_to_cell ar (AV.cit_to_it (aview_2tile2 et tile) (CIdx 0sz brow bcol)) x;
    with x.
      rewrite varray_pts_to_cell ar (AIdx 1 (tid / tile) (tid % tile)) x
           as varray_pts_to_cell ar (AV.cit_to_it (aview_2tile2 et tile) (CIdx 1sz brow bcol)) x;
    AV.varray_write_cell ar (CIdx 0sz brow bcol) v1;
    AV.varray_write_cell ar (CIdx 1sz brow bcol) v2;
    with x.
      rewrite varray_pts_to_cell ar (AV.cit_to_it (aview_2tile2 et tile) (CIdx 0sz brow bcol)) x
           as varray_pts_to_cell ar (AIdx 0 (tid / tile) (tid % tile)) x;
    with x.
      rewrite varray_pts_to_cell ar (AV.cit_to_it (aview_2tile2 et tile) (CIdx 1sz brow bcol)) x
           as varray_pts_to_cell ar (AIdx 1 (tid / tile) (tid % tile)) x;


    assert (B.barrier_tok (barrier_p tile ar) (barrier_q tile ar) (2 * vbk + 1) tid);
    odd_2x1 vbk;
    assert (pure (odd (2 * vbk + 1)));
    rewrite (exists* x. varray_pts_to_cell ar (AIdx 0 (tid / tile) (tid % tile)) x) **
            (exists* x. varray_pts_to_cell ar (AIdx 1 (tid / tile) (tid % tile)) x)
         as (barrier_p tile ar (2 * vbk + 1) tid);
    B.barrier_wait ();
    even_2x (vbk + 1);
    (* sigh *)
    assert (pure (2 * (vbk + 1) == 2 * vbk + 2));
    assert (pure (even (2 * vbk + 2)));
    rewrite (barrier_q tile ar (2 * vbk + 1) tid)
         as (exists* x. AV.varray_pts_to ar #(1.0R /. (tile *^ tile)) x);

    let mut sk : sz = 0sz;
    while (let vsk = !sk; SZ.(vsk <^ tile))
      invariant b.
        exists* (vsk : SZ.t{vsk <= tile}) sumv.
          pure (b == (SZ.v vsk < tile)) **
          pts_to sk vsk **
          pts_to #_ #et sum sumv **
          m4_pts_to gA #(f /. mlayout_size lC) eA **
          m4_pts_to gB #(f /. mlayout_size lC) eB **
          (exists* x. AV.varray_pts_to ar #(1.0R /. (tile *^ tile)) x) **
          gpu
    {
      let vsk = !sk;
      let v1 = AV.varray_read ar (CIdx #tile 0sz brow vsk);
      let v2 = AV.varray_read ar (CIdx #tile 1sz vsk bcol);

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

  AV.varray_concr ar;
  rewrite each SZ.v tid as reveal etid;
  rewrite each ar as AV.from_array (aview_2tile2 et tile) ar0;
  with x1.
    rewrite
      gpu_pts_to_array (AV.core (AV.from_array (aview_2tile2 et tile) ar0)) #(1.0R /. (tile *^ tile)) x1
    as
      gpu_pts_to_array ar0 #(1.0R /. (tile *^ tile)) x1;
  rewrite each ar0 as ear;
  fold barrier_tok tile ear (2 * mshared) tid;

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
