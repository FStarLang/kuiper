module Kuiper.Poly.GEMM.BlockTiling2D

#lang-pulse

open Kuiper
open Kuiper.EMatrix4
open Kuiper.EMatrix6
open Kuiper.Matrix.Reprs.Type
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.View.TwoTiles { aview_2tile2, mkAIdx, mkCIdx }

open Kuiper.Matrix {
  gpu_matrix as gpu_matrix2,
  gpu_matrix_pts_to as gpu_matrix_pts_to2
}

open Kuiper.Matrix4 {
  gpu_matrix as gpu_matrix4,
  gpu_matrix_pts_to as m4_pts_to,
  gpu_matrix_pts_to_cell as m4_pts_to_cell,
  mlayout4,
  clayout4
}

open Kuiper.Matrix6 {
  gpu_matrix as gpu_matrix6,
  gpu_matrix_pts_to as m6_pts_to,
  gpu_matrix_pts_to_cell as m6_pts_to_cell,
  mlayout6,
  clayout6
}

module M2 = Kuiper.Matrix
module M4 = Kuiper.Matrix4
module M6 = Kuiper.Matrix6
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
let own_2_mats
  (#et : Type0)
  (browscols : szp { SZ.v browscols <= 32 })
  (tile : valid_tile)
  (ar : varray (aview_2tile2 et (tile *^ browscols)))
  (tid : natlt (tile * tile))
  : slprop =
    forall+ (ii : natlt browscols). forall+ (jj : natlt browscols).
      (exists* x. varray_pts_to_cell ar (mkAIdx 0 (ii * tile + tid / tile) (jj * tile + tid % tile)) x) **
      (exists* x. varray_pts_to_cell ar (mkAIdx 1 (ii * tile + tid / tile) (jj * tile + tid % tile)) x)

let barrier_p
  (#et : Type0)
  (browscols : szp { SZ.v browscols <= 32 })
  (tile : valid_tile)
  (ar : varray (aview_2tile2 et (tile *^ browscols)))
  : B.barrier_side (tile * tile) =
  fun it tid ->
    if even it
    then (exists* x. varray_pts_to ar #(1.0R /. (tile * tile)) x)
    else own_2_mats browscols tile ar tid

let barrier_q
  (#et : Type0)
  (browscols : szp { SZ.v browscols <= 32 })
  (tile : valid_tile)
  (ar : varray (aview_2tile2 et (tile *^ browscols)))
  : B.barrier_side (tile * tile) =
  fun it tid -> barrier_p browscols tile ar (it+1) tid (* flip flop *)

(* without shmem ownership *)
unfold
let kpre1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols #trows #tcols : szp)
  (tile : valid_tile)
  (#lA : mlayout4 mrows   mshared (tile * trows) (tile * tcols))
  (#lB : mlayout4 mshared mcols   (tile * trows) (tile * tcols))
  (#lC : mlayout6 mrows   mcols   tile tile trows tcols)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix6 et lC)
  (eA : ematrix4 et mrows mshared (tile * trows) (tile * tcols))
  (eB : ematrix4 et mshared mcols (tile * trows) (tile * tcols))
  (f : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  (* mlayout_size lC: wrong, should be (mrows*mcols)*tile*tile *)
  m4_pts_to gA #(f /. mlayout_size lC) eA **
  m4_pts_to gB #(f /. mlayout_size lC) eB **
  (* each thread owns a square *)
  (exists* v. gpu_matrix_pts_to2 (M6.gpu_matrix6_to_gpu_matrix2 gC (bid / mcols) (bid % mcols) (tid / tile) (tid % tile)) v)

(* NO FUNCTIONAL SPEC RIGHT NOW *)
unfold
let kpost1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols #trows #tcols : szp)
  (tile : valid_tile)
  (#lA : mlayout4 mrows   mshared (tile * trows) (tile * tcols))
  (#lB : mlayout4 mshared mcols   (tile * trows) (tile * tcols))
  (#lC : mlayout6 mrows   mcols   tile tile trows tcols)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix6 et lC)
  (eA : ematrix4 et mrows mshared (tile * trows) (tile * tcols))
  (eB : ematrix4 et mshared mcols (tile * trows) (tile * tcols))
  (f : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  m4_pts_to gA #(f /. mlayout_size lC) eA **
  m4_pts_to gB #(f /. mlayout_size lC) eB **
  (* each thread owns a square *)
  (exists* v. gpu_matrix_pts_to2 (M6.gpu_matrix6_to_gpu_matrix2 gC (bid / mcols) (bid % mcols) (tid / tile) (tid % tile)) v)

let barrier_tok
  (#et : Type0)
  (browscols : szp { SZ.v browscols <= 32 })
  (tile : valid_tile)
  (ar: gpu_array et (2sz *^ (tile *^ browscols) *^ (tile *^ browscols)))
  (it : nat)
  (tid : natlt (tile * tile))
  : slprop
  =
  B.barrier_tok (barrier_p browscols tile (AV.from_array (aview_2tile2 et (tile *^ browscols)) ar))
                (barrier_q browscols tile (AV.from_array (aview_2tile2 et (tile *^ browscols)) ar)) it tid

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols #trows #tcols : szp)
  (#browscols : szp { SZ.v browscols <= 32 /\ browscols == trows /\ browscols == tcols })
  (tile : valid_tile)
  (#lA : mlayout4 mrows   mshared (tile * trows) (tile * tcols))
  (#lB : mlayout4 mshared mcols   (tile * trows) (tile * tcols))
  (#lC : mlayout6 mrows   mcols   tile tile trows tcols)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix6 et lC)
  (eA : ematrix4 et mrows mshared (tile * trows) (tile * tcols))
  (eB : ematrix4 et mshared mcols (tile * trows) (tile * tcols))
  (f : perm)
  (ar: gpu_array et (2sz *^ (tile *^ browscols) *^ (tile *^ browscols)))
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  kpre1 comb tile gA gB gC eA eB f bid tid **
  (exists* x. gpu_pts_to_slice ar #(1.0R /. (tile * tile)) 0 (2sz *^ (tile *^ browscols) *^ (tile *^ browscols)) x) **
  barrier_tok browscols tile ar 0 tid **
  shmem_tok ar


unfold
let kpost
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols #trows #tcols : szp)
  (#browscols : szp { SZ.v browscols <= 32 /\ browscols == trows /\ browscols == tcols })
  (tile : valid_tile)
  (#lA : mlayout4 mrows   mshared (tile * trows) (tile * tcols))
  (#lB : mlayout4 mshared mcols   (tile * trows) (tile * tcols))
  (#lC : mlayout6 mrows   mcols   tile tile trows tcols)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix6 et lC)
  (eA : ematrix4 et mrows mshared (tile * trows) (tile * tcols))
  (eB : ematrix4 et mshared mcols (tile * trows) (tile * tcols))
  (f : perm)
  (ar: gpu_array et (2sz *^ (tile *^ browscols) *^ (tile *^ browscols)))
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  kpost1 comb tile gA gB gC eA eB f bid tid **
  (exists* x. gpu_pts_to_slice ar #(1.0R /. (tile * tile)) 0 (2sz *^ (tile *^ browscols) *^ (tile *^ browscols)) x) **
  barrier_tok browscols tile ar (2 * mshared) tid

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
fn subproduct
  (#et : Type0) {| scalar et |}
  (#browscols : szp { SZ.v browscols <= 32 })
  (tile : valid_tile)
  (acc : array et)
  (ar : varray (aview_2tile2 et (tile *^ browscols)))
  (trow tcol : szlt tile)
  (#acc0 : erased (seq et))
  (#ar0 : erased _)
  (#f : perm)
  preserves
    gpu **
    AV.varray_pts_to ar #f ar0
  requires
    pure (Seq.length acc0 == browscols * browscols) **
    (acc |-> acc0)
  ensures
    exists* acc'.
      pure (Seq.length acc' == browscols * browscols) **
      (acc |-> acc')
{
  let trow' = trow *^ browscols;
  let tcol' = tcol *^ browscols;

  let mut sk : sz = 0sz;
  while (let vsk = !sk; SZ.(vsk <^ tile *^ browscols))
    invariant b.
      exists* (vsk : SZ.t{vsk <= tile * browscols}) (accv : erased (lseq et (browscols * browscols))).
        pure (b == (SZ.v vsk < tile * browscols)) **
        pts_to #_ #sz sk vsk **
        pts_to #_ #(seq et) acc accv **
        AV.varray_pts_to ar #f ar0 **
        gpu
  {
    let vsk = !sk;

    let mut regM : Pulse.Lib.Array.array et = [| zero #et #_ ; browscols |];
    let mut regN : Pulse.Lib.Array.array et = [| zero #et #_ ; browscols |];

    // load relevant As & Bs entries into registers
    let mut m = 0sz;
    while (let vm = !m; SZ.(vm <^ browscols))
      invariant b.
        exists* (vm : SZ.t{vm <= browscols}) (accm : erased (lseq et browscols)).
          pure (b == (SZ.v vm < browscols)) **
          pts_to #_ #sz m vm **
          pts_to #_ #(seq et) regM accm **
          AV.varray_pts_to ar #f ar0 **
          gpu
    {
      let vm = !m;
      let vM = AV.varray_read ar (mkCIdx #(tile *^ browscols) 0sz (trow' +^ vm) vsk);
      open Pulse.Lib.Array;
      regM.(vm) <- vM;
      m := vm +^ 1sz;
    };
    let mut n = 0sz;
    while (let vn = !n; SZ.(vn <^ browscols))
      invariant b.
        exists* (vn : SZ.t{vn <= browscols}) (accn : erased (lseq et browscols)).
          pure (b == (SZ.v vn < browscols)) **
          pts_to #_ #sz n vn **
          pts_to #_ #(seq et) regN accn **
          AV.varray_pts_to ar #f ar0 **
          gpu
    {
      let vn = !n;
      let vM = AV.varray_read ar (mkCIdx #(tile *^ browscols) 1sz vsk (tcol' +^ vn));
      open Pulse.Lib.Array;
      regN.(vn) <- vM;
      n := vn +^ 1sz;
    };

    let mut x = 0sz;
    while (let vx = !x; SZ.(vx <^ browscols))
      invariant b.
        exists* (vx : SZ.t{vx <= browscols}) (accm : erased (lseq et browscols)) (accn : erased (lseq et browscols)) (accv : erased (lseq et (browscols * browscols))).
          pure (b == (SZ.v vx < browscols)) **
          pts_to #_ #sz x vx **
          pts_to #_ #(seq et) regM accm **
          pts_to #_ #(seq et) regN accn **
          pts_to #_ #(seq et) acc accv **
          AV.varray_pts_to ar #f ar0 **
          gpu
    {
      let vx = !x;

      let mut y = 0sz;
      while (let vy = !y; SZ.(vy <^ browscols))
        invariant b.
          exists* (vy : SZ.t{vy <= browscols}) (accm : erased (lseq et browscols)) (accn : erased (lseq et browscols)) (accv : erased (lseq et (browscols * browscols))).
            pure (b == (SZ.v vy < browscols)) **
            pts_to #_ #sz y vy **
            pts_to #_ #(seq et) regM accm **
            pts_to #_ #(seq et) regN accn **
            pts_to #_ #(seq et) acc accv **
            AV.varray_pts_to ar #f ar0 **
            gpu
      {
        let vy = !y;

        open Pulse.Lib.Array;
        let vM = regM.(vx);
        let vN = regN.(vy);
        let sum0 = acc.(vy);
        let sum1 = sum0 `add` (vM `mul` vN);
        acc.(vy *^ browscols +^ vx) <- sum1;

        y := vy +^ 1sz;
      };

      x := vx +^ 1sz;
    };
    sk := vsk +^ 1sz;
    ();
  }
}

inline_for_extraction noextract
fn bring_2cols
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols #trows #tcols : erased nat)
  (#browscols : szp { SZ.v browscols <= 32 /\ SZ.v browscols == trows /\ SZ.v browscols == tcols })
  (#lA : mlayout4 mrows   mshared (tile * trows) (tile * tcols))
  (#lB : mlayout4 mshared mcols   (tile * trows) (tile * tcols))
  {| clayout4 lA |}
  {| clayout4 lB |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (ar : varray (aview_2tile2 et (tile *^ browscols)))
  (mrow : szlt mrows)
  (mk : szlt mshared)
  (mcol : szlt mcols)
  (tid : szlt (tile * tile))
  (#f : perm)
  (#eA : ematrix4 et mrows   mshared (tile * trows) (tile * tcols))
  (#eB : ematrix4 et mshared mcols   (tile * trows) (tile * tcols))
  preserves
    gpu
  requires
    m4_pts_to gA #f eA **
    m4_pts_to gB #f eB **
    own_2_mats browscols tile ar tid
  ensures
    m4_pts_to gA #f eA **
    m4_pts_to gB #f eB **
    own_2_mats browscols tile ar tid
{
  let tidrow, tidcol = s_divmod tile tid;
  let mut i = 0sz;
  while (let vi = !i; SZ.(vi <^ (browscols *^ browscols)))
    invariant b.
      exists* (vi : SZ.t{vi <= browscols * browscols}).
        pure (b == (SZ.v vi < browscols * browscols)) **
        pts_to #_ #sz i vi **
        m4_pts_to gA #f eA **
        m4_pts_to gB #f eB **
        own_2_mats browscols tile ar tid **
        gpu
  {
    let vi = !i;
    let vrow, vcol = s_divmod browscols vi;
    assert pure (vrow < browscols /\ vcol < browscols);
    let vrow' = vrow *^ tile +^ tidrow;
    let vcol' = vcol *^ tile +^ tidcol;

    unfold own_2_mats browscols tile ar tid;
    forevery_extract #(natlt browscols) vrow _;
    forevery_extract #(natlt browscols) vcol _;
    let v1 = M4.gpu_matrix_read gA mrow mk vrow' vcol';
    admit();
    AV.varray_write_cell' ar (mkCIdx 0sz vrow' vcol') (mkAIdx 0 vrow' vcol') v1;
    let v2 = M4.gpu_matrix_read gB mk mcol vrow' vcol';
    AV.varray_write_cell' ar (mkCIdx 1sz vrow' vcol') (mkAIdx 1 vrow' vcol') v2;
    Pulse.Lib.Trade.elim_trade _ _;
    Pulse.Lib.Trade.elim_trade _ _;
    fold own_2_mats browscols tile ar tid;
    i := vi +^ 1sz;
  }
}

// Set flag to show implicit arguments when printing
// #set-options "--print_implicits"

inline_for_extraction noextract
fn kf
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols #trows #tcols : SZ.t)
  (#browscols : szp { SZ.v browscols <= 32 /\ browscols == trows /\ browscols == tcols })
  (#lA : mlayout4 mrows   mshared (tile * trows) (tile * tcols))
  (#lB : mlayout4 mshared mcols   (tile * trows) (tile * tcols))
  (#lC : mlayout6 mrows   mcols   tile tile trows tcols)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout6 lC |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix6 et lC)
  (#eA : ematrix4 et mrows   mshared (tile * trows) (tile * tcols))
  (#eB : ematrix4 et mshared mcols   (tile * trows) (tile * tcols))
  (#f : perm)
  (ear : erased (gpu_array et (2sz *^ (tile *^ browscols) *^ (tile *^ browscols))))
  (bid : szlt2 mrows mcols)
  (tid : szlt (tile *^ tile))
  ()
  requires
    gpu **
    kpre comb tile gA gB gC eA eB f ear bid tid **
    thread_id (tile *^ tile) tid **
    block_id (mrows * mcols) bid
  ensures
    gpu **
    kpost comb tile gA gB gC eA eB f ear bid tid **
    thread_id (tile *^ tile) tid **
    block_id (mrows * mcols) bid

{
  gpu_pts_to_ref ear;

  let ar0 = obtain_shmem ear; rewrite each ear as ar0;
  unfold barrier_tok browscols tile ar0 0 tid;

  // Does not work:
  // rewrite each AV.from_array #et #(2sz *^ tile *^ tile) #(ematrix et tile tile & ematrix et tile tile) ar (aview_2tile2 et tile)
  //           as ar;
  AV.varray_abs' (aview_2tile2 et (tile *^ browscols)) ar0;
  let ar = AV.from_array (aview_2tile2 et (tile *^ browscols) <: erased _) ar0;
  rewrite
    B.barrier_tok (barrier_p browscols tile (AV.from_array (aview_2tile2 et (tile *^ browscols)) ar0))
                  (barrier_q browscols tile (AV.from_array (aview_2tile2 et (tile *^ browscols)) ar0)) 0 tid
  as
    B.barrier_tok (barrier_p browscols tile ar)
                  (barrier_q browscols tile ar) 0 tid;

  with x0.
    rewrite
      AV.varray_pts_to (AV.from_array (aview_2tile2 et (tile *^ browscols)) ar0) #(1.0R /. (tile * tile)) x0
    as
      AV.varray_pts_to ar #(1.0R /. (tile * tile)) x0;

  let mrow, mcol = s_divmod mcols bid;
  let brow, bcol = s_divmod tile tid;
  // assert (pure (SZ.v bcol == tid % tile));
  // assert (pure (bcol < tile));

  (* thread-local result cache *)
  let mut sums : Pulse.Lib.Array.array et = [| zero #et #_ ; trows *^ tcols |];
  let mut bk  : sz = 0sz;
  while (let vbk = !bk; SZ.(vbk <^ mshared))
    invariant b.
      exists* (vbk : SZ.t{vbk <= mshared}) (sumv : lseq et (trows *^ tcols)).
        pure (b == (SZ.v vbk < mshared)) **
        pts_to #_ #sz bk vbk **
        pts_to #_ #(seq et) sums sumv **
        m4_pts_to gA #(f /. mlayout_size lC) eA **
        m4_pts_to gB #(f /. mlayout_size lC) eB **
        (exists* x. varray_pts_to ar #(1.0R /. (tile * tile)) x) **
        B.barrier_tok (barrier_p browscols tile ar) (barrier_q browscols tile ar) (2 * vbk) tid **
        gpu
  {
    let vbk = !bk;

    (* This assert should not be needed. I don't know what effect it even has. *)
    assert B.barrier_tok (barrier_p browscols tile ar) (barrier_q browscols tile ar) (2 * vbk) tid;
    even_2x vbk;
    assert (pure (even (2 * vbk)));
    rewrite (exists* x. AV.varray_pts_to ar #(1.0R /. (tile * tile)) x)
         as (barrier_p browscols tile ar (2 * vbk) tid);
    B.barrier_wait ();
    rewrite (barrier_q browscols tile ar (2 * vbk) tid)
         as own_2_mats browscols tile ar tid;

    (* At this point we exclusively own a full column of the SHMEM
       cache. Populate it. *)
    bring_2cols tile gA gB ar mrow vbk mcol tid;

    assert (B.barrier_tok (barrier_p browscols tile ar) (barrier_q browscols tile ar) (2 * vbk + 1) tid);
    odd_2x1 vbk;
    assert (pure (odd (2 * vbk + 1)));
    rewrite own_2_mats browscols tile ar tid
         as (barrier_p browscols tile ar (2 * vbk + 1) tid);
    B.barrier_wait ();
    even_2x (vbk + 1);
    (* sigh *)
    assert (pure (2 * (vbk + 1) == 2 * vbk + 2));
    assert (pure (even (2 * vbk + 2)));
    rewrite (barrier_q browscols tile ar (2 * vbk + 1) tid)
         as (exists* x. AV.varray_pts_to ar #(1.0R /. (tile * tile)) x);

    (* At this point the SHMem cache is filled with the submatrices
       and we have RO permission to it. Compute product for our cell in
       the tile and add to sum. *)

    subproduct #_ #_ #browscols tile sums ar brow bcol;

    (* Move to next tile *)
    bk := vbk +^ 1sz;
  };

  (* Write all the accumulated sums. *)

  let mut idx : sz = 0sz;
  Pulse.Lib.Array.pts_to_len sums;
  while (let vidx = !idx; SZ.(vidx <^ trows *^ tcols))
    invariant b.
      exists* (vidx : SZ.t{vidx <= trows * tcols}) (sumv : lseq et (trows *^ tcols)).
        pure (b == (SZ.v vidx < trows * tcols)) **
        pts_to #_ #sz idx vidx **
        pts_to #_ #(seq et) sums sumv **
        (exists* v. gpu_matrix_pts_to2 (M6.gpu_matrix6_to_gpu_matrix2 gC (bid / mcols) (bid % mcols) (tid / tile) (tid % tile)) v) **
        gpu
  {
    let vidx = !idx;
    let row, col = s_divmod tcols vidx;
    // forevery_extract #(natlt tile) (SZ.v vrow) _;

    // (* tedious *)
    // with bi0 bj0 i0 j0 v0.
    //   rewrite m4_pts_to_cell gC bi0  bj0  i0   j0   v0
    //       as m4_pts_to_cell gC mrow mcol vrow bcol v0;

    let layout = (M6.mlayout6_to_mlayout2 #et lC (bid / mcols) (bid % mcols) (tid / tile) (tid % tile));
    admit();
    let clayout = M6.clayout6_to_clayout2 #et lC (bid / mcols) (bid % mcols) (tid / tile) (tid % tile);
    let v0 = M2.gpu_matrix_read #_ #_ #_ #layout #clayout (M6.gpu_matrix6_to_gpu_matrix2 gC (bid / mcols) (bid % mcols) (tid / tile) (tid % tile)) row col;
    open Pulse.Lib.Array;
    let v1 = sums.(vidx);
    let v' = comb v0 v1;
    M4.gpu_matrix_write_cell gC row col v';

    // with bi0 bj0 i0 j0 v0.
    //   rewrite m4_pts_to_cell gC bi0  bj0  i0   j0   v0
    //       as m4_pts_to_cell gC (bid / mcols) (bid % mcols) vrow tid v0;

    idx := vidx +^ 1sz;
    // Pulse.Lib.Trade.elim_trade _
    //   (forall+ (ii : natlt tile).
    //     (exists* v.
    //       m4_pts_to_cell gC #1.0R
    //         (bid / mcols) (bid % mcols)
    //         ii tid v));
  };

  AV.varray_concr ar;
  rewrite each SZ.v tid as reveal tid;
  rewrite each ar as AV.from_array (aview_2tile2 et (tile *^ browscols)) ar0;
  with x1.
    rewrite
      gpu_pts_to_array (AV.core (AV.from_array (aview_2tile2 et (tile *^ browscols)) ar0)) #(1.0R /. (tile * tile)) x1
    as
      gpu_pts_to_array ar0 #(1.0R /. (tile * tile)) x1;
  rewrite each ar0 as ear;
  fold barrier_tok browscols tile ear (2 * mshared) tid;

  ()
}


ghost
fn setup
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols #trows #tcols : SZ.t)
  (#lA : mlayout4 mrows   mshared (tile * trows) (tile * tcols))
  (#lB : mlayout4 mshared mcols   (tile * trows) (tile * tcols))
  (#lC : mlayout6 mrows   mcols   tile tile trows tcols)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout6 lC |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix6 et lC)
  (#eA : ematrix4 et mrows   mshared (tile * trows) (tile * tcols))
  (#eB : ematrix4 et mshared mcols   (tile * trows) (tile * tcols))
  (#eC : ematrix6 et mrows   mcols   tile tile trows tcols)
  ()
  requires
    (gA |-> eA) **
    (gB |-> eB) **
    (gC |-> eC)
  ensures
    (forall+ (bid : natlt2 mrows mcols)
             (tid : natlt (tile * tile)).
      kpre1 comb tile gA gB gC eA eB 1.0R bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn block_setup
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols #trows #tcols : SZ.t)
  (#browscols : SZ.t { SZ.v browscols <= 32 /\ browscols == trows /\ browscols == tcols })
  (#lA : mlayout4 mrows   mshared (tile * trows) (tile * tcols))
  (#lB : mlayout4 mshared mcols   (tile * trows) (tile * tcols))
  (#lC : mlayout6 mrows   mcols   tile tile trows tcols)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout6 lC |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix6 et lC)
  (#eA : ematrix4 et mrows   mshared (tile * trows) (tile * tcols))
  (#eB : ematrix4 et mshared mcols   (tile * trows) (tile * tcols))
  (#eC : ematrix6 et mrows   mcols   tile tile trows tcols)
  (ar: gpu_array et (2sz *^ (tile *^ browscols) *^ (tile *^ browscols)))
  (bid : natlt (mrows *^ mcols))
  ()
  requires
    block_setup_tok (tile *^ tile) **
    (exists* v. gpu_pts_to_array ar #1.0R v) **
    (forall+ (tid : natlt (tile * tile)).
      kpre1 comb tile gA gB gC eA eB 1.0R bid tid)
  ensures
    block_setup_tok (tile *^ tile) **
    (forall+ (tid : natlt (tile *^ tile)).
      kpre comb tile gA gB gC eA eB 1.0R ar bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn block_teardown
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols #trows #tcols : SZ.t)
  (#browscols : SZ.t { SZ.v browscols <= 32 /\ browscols == trows /\ browscols == tcols })
  (#lA : mlayout4 mrows   mshared (tile * trows) (tile * tcols))
  (#lB : mlayout4 mshared mcols   (tile * trows) (tile * tcols))
  (#lC : mlayout6 mrows   mcols   tile tile trows tcols)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout6 lC |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix6 et lC)
  (#eA : ematrix4 et mrows   mshared (tile * trows) (tile * tcols))
  (#eB : ematrix4 et mshared mcols   (tile * trows) (tile * tcols))
  (#eC : ematrix6 et mrows   mcols   tile tile trows tcols)
  (ar: gpu_array et (2sz *^ (tile *^ browscols) *^ (tile *^ browscols)))
  (bid : natlt (mrows *^ mcols))
  ()
  requires
    (forall+ (tid : natlt (tile *^ tile)).
      kpost comb tile gA gB gC eA eB 1.0R ar bid tid) **
    emp (* frame *)
  ensures
    (exists* v. gpu_pts_to_array ar #1.0R v) **
    (forall+ (tid : natlt (tile * tile)).
      kpost1 comb tile gA gB gC eA eB 1.0R bid tid)
{
  admit();
}

ghost
fn teardown
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols #trows #tcols : SZ.t)
  (#lA : mlayout4 mrows   mshared (tile * trows) (tile * tcols))
  (#lB : mlayout4 mshared mcols   (tile * trows) (tile * tcols))
  (#lC : mlayout6 mrows   mcols   tile tile trows tcols)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout6 lC |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix6 et lC)
  (#eA : ematrix4 et mrows   mshared (tile * trows) (tile * tcols))
  (#eB : ematrix4 et mshared mcols   (tile * trows) (tile * tcols))
  (#eC : ematrix6 et mrows   mcols   tile tile trows tcols)
  ()
  requires
    (forall+ (bid : natlt2 mrows mcols)
             (tid : natlt (tile * tile)).
      kpost1 comb tile gA gB gC eA eB 1.0R bid tid) **
    emp (* frame *)
  ensures
    (gA |-> eA) **
    (gB |-> eB) **
    // (gC |-> MS.mmcomb comb eC eA eB)
    (exists* eC'. (gC |-> eC'))
{
  admit();
  forevery_flatten #(natlt2 mrows mcols) #_ #(natlt tile)
    (fun bid tid -> kpost1 comb tile gA gB gC eA eB 1.0R bid tid);
  forevery_unzip #(natlt2 mrows mcols & natlt tile) _ _;
  forevery_unzip #(natlt2 mrows mcols & natlt tile) _ _;
  forevery_tostar #(natlt2 mrows mcols & natlt tile) (fun _tid -> m4_pts_to gA #(1.0R /. mlayout_size lC) eA);

    // (fun (bid, tid) -> kpost1 comb tile gA gB gC eA eB 1.0R bid tid);
  admit();
}

inline_for_extraction noextract
let mk_kernel
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols #trows #tcols : SZ.t)
  (#browscols : SZ.t { SZ.v browscols <= 32 /\ browscols == trows /\ browscols == tcols })
  (#lA : mlayout4 mrows   mshared (tile * trows) (tile * tcols))
  (#lB : mlayout4 mshared mcols   (tile * trows) (tile * tcols))
  (#lC : mlayout6 mrows   mcols   tile tile trows tcols)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout6 lC |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix6 et lC)
  (#eA : ematrix4 et mrows   mshared (tile * trows) (tile * tcols))
  (#eB : ematrix4 et mshared mcols   (tile * trows) (tile * tcols))
  (#eC : ematrix6 et mrows   mcols   tile tile trows tcols)
  (_ : squash (mrows * mcols <= max_blocks
               /\ tile * tile <= max_threads))
  : kernel_desc
      ((gA |-> eA) ** (gB |-> eB) ** (gC |-> eC))
      ((gA |-> eA) ** (gB |-> eB) ** (exists* eC'. (gC |-> eC')))
= {
  nblk = mrows *^ mcols;
  nthr = tile *^ tile;

  shmem_type = et;
  shmem_type_is_sized = solve;
  shmem_sz = (2sz *^ (tile *^ browscols) *^ (tile *^ browscols));

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt (tile * tile)). kpre1  comb tile gA gB gC eA eB 1.0R bid tid);
  block_post = (fun bid -> forall+ (tid : natlt (tile * tile)). kpost1 comb tile gA gB gC eA eB 1.0R bid tid);
  setup      = setup    tile comb gA gB gC #eA #eB #eC;
  teardown   = teardown tile comb gA gB gC #eA #eB #eC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup    tile comb gA gB gC #eA #eB #eC;
  block_teardown = block_teardown tile comb gA gB gC #eA #eB #eC;

  kpre      = kpre  comb tile gA gB gC eA eB 1.0R;
  kpost     = kpost comb tile gA gB gC eA eB 1.0R;

  f = kf tile #et #_ comb #mrows #mshared #mcols #trows #tcols #browscols gA gB gC #eA #eB #1.0R;
}

inline_for_extraction noextract
fn matmul_gpu
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols #trows #tcols : szp)
  (#browscols : szp { SZ.v browscols <= 32 /\ browscols == trows /\ browscols == tcols })
  (lA : mlayout4 mrows   mshared (tile * trows) (tile * tcols))
  (lB : mlayout4 mshared mcols   (tile * trows) (tile * tcols))
  (lC : mlayout6 mrows   mcols   tile tile trows tcols)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout6 lC |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix6 et lC)
  (#eA : ematrix4 et mrows   mshared (tile * trows) (tile * tcols))
  (#eB : ematrix4 et mshared mcols   (tile * trows) (tile * tcols))
  (#eC : ematrix6 et mrows   mcols   tile tile trows tcols)
  preserves
    cpu **
    (gA |-> eA) **
    (gB |-> eB)
  requires
    pure (mrows * mcols <= max_blocks) **
    pure (tile * tile <= max_threads) **
    (gC |-> eC)
  ensures
    (exists* eC'. (gC |-> eC'))
{
  dassert (tile `SZ.gt` 0sz);
  launch_sync (mk_kernel tile comb #mrows #mshared #mcols #trows #tcols #browscols gA gB gC ());
}
