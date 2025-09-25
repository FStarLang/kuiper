module Kuiper.Poly.GEMM.TensorCore

#lang-pulse

open Kuiper

#set-options "--z3rlimit 20"

open Kuiper.Matrix.Reprs.Type
open Kuiper.Math { even, odd, even_2x, odd_2x1 }

open Kuiper.Matrix

module MS = Kuiper.Spec.GEMM
module SZ = FStar.SizeT
module B = Kuiper.Barrier

module R = Kuiper.Matrix.Reprs

open Kuiper.EMatrix { ematrix }
open Kuiper.VArray {
  varray,
  varray_pts_to,
  varray_pts_to_cell
}
open Kuiper.TensorCore
open Kuiper.Float16
open Kuiper.Matrix.Tiling

open Kuiper.Poly.GEMM.Copy
open Kuiper.Poly.GEMM.Tiled.Common

let live_warp_tile
  (#et : Type0) {| scalar et |}
  // Since this is an slprop, I would like to not erase the nat.
  // Unfortunately, when unfolding live_warp_tile, after passing
  // a (reveal x) as argument, this leads to (reveal (hide (reveal x)))
  // which creates problems with type equalities.
  (#rows : erased nat)
  (#cols : nat)
  (#lC : mlayout rows cols)
  (gC : gpu_matrix et lC)
  (bm : nat{bm > 0 /\ bm /? rows})
  (bn : nat{bn > 0 /\ bn /? cols})
  (tm : nat{tm > 0 /\ tm /? bm})
  (tn : nat{tn > 0 /\ tn /? bn})
  (bid : natlt ((rows/bm) * (cols/bn)))
  (wid : natlt (bm/tm * (bn/tn)))
  : slprop
  =
    live (warp_tile (block_tile gC bm bn bid) tm tn wid)

// TODO look again at why we cannot use nats here instead of sizet
unfold
let kpre1
  (#et_ab #et_c : Type0) {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (fA fB : perm)
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt (bm/tm*(bn/tn)*warp_size))
  : slprop
  =
  pure (SZ.fits (rows * shared)) **
  pure (SZ.fits (shared * cols)) **
  pure (valid_frag_et_dims et_ab FragA tm tn tk) **
  pure (valid_frag_et_dims et_ab FragB tm tn tk) **
  pure (valid_frag_et_dims et_c FragAcc tm tn tk) **
  pure (valid_frag_et_comb et_ab et_c) **
  gA |-> Frac (fA /. (rows/tm * (cols/tn) * warp_size)) eA **
  gB |-> Frac (fB /. (rows/tm * (cols/tn) * warp_size)) eB **
  live_warp_tile gC bm bn tm tn bid (tid/warp_size)

unfold
let kpost1
  (#et_ab #et_c : Type0) {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (fA fB : perm)
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt (bm/tm*(bn/tn)*warp_size))
  : slprop
  =
  gA |-> Frac (fA /. (rows/tm * (cols/tn) * warp_size)) eA **
  gB |-> Frac (fB /. (rows/tm * (cols/tn) * warp_size)) eB **
  live_warp_tile gC bm bn tm tn bid (tid/warp_size)

let barrier_p
  (#et : Type0)
  (#bm #bn #bk : szp)
  (#l1 : mlayout bm bk)
  (#l2 : mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  (nthr : pos)
  : B.barrier_side nthr =
  fun it tid ->
    if even it then
      (exists* (x : ematrix _ _ _). m1 |-> Frac (1.0R /. nthr) x) **
      (exists* (x : ematrix _ _ _). m2 |-> Frac (1.0R /. nthr) x)
    else
      live_tile_stride_cells m1 nthr tid **
      live_tile_stride_cells m2 nthr tid

let barrier_q
  (#et : Type0)
  (#bm #bn #bk : szp)
  (#l1 : mlayout bm bk)
  (#l2 : mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  (nthr : pos)
  : B.barrier_side nthr =
  fun it tid -> barrier_p m1 m2 nthr (it+1) tid (* flip flop *)

let barrier_tok
  (#et : Type0)
  (#bm #bn #bk : szp)
  (* This is defined over the base shared gpu_arrays, as
  this spec must make sense before the arrays are viewed as
  a matrix. *)
  (l1 : full_mlayout bm bk)
  (l2 : full_mlayout bk bn)
  (sar1 : gpu_array et (bm * bk))
  (sar2 : gpu_array et (bk * bn))
  (it : nat)
  (nthr : pos)
  (tid : natlt nthr)
  : slprop
  =
  B.barrier_tok (barrier_p (from_array l1 sar1) (from_array l2 sar2) nthr)
                (barrier_q (from_array l1 sar1) (from_array l2 sar2) nthr)
                it tid

unfold
let kpre
  (#et_ab #et_c : Type0) {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt (bm/tm*(bn/tn)*warp_size))
  : slprop
  =
  kpre1 gA eA gB eB gC bm bn bk tm tn tk fA fB bid tid **
  (exists* (x : seq _). gpu_pts_to_array (fst sh)       #(1.0R /. (bm/tm*(bn/tn)*warp_size)) x) **
  (exists* (x : seq _). gpu_pts_to_array (fst (snd sh)) #(1.0R /. (bm/tm*(bn/tn)*warp_size)) x) **
  barrier_tok (R.row_major bm bk) (R.row_major bk bn) (fst sh) (fst (snd sh)) 0 (bm/tm * (bn/tn) * warp_size) tid

unfold
let kpost
  (#et_ab #et_c : Type0) {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt (bm/tm*(bn/tn)*warp_size))
  : slprop
  =
  kpost1 gA eA gB eB gC bm bn tm tn fA fB bid tid **
  exists* (x : seq _). fst sh |-> Frac (1.0R /. (bm/tm*(bn/tn)*warp_size)) x **
  exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. (bm/tm*(bn/tn)*warp_size)) x **
  barrier_tok (R.row_major bm bk) (R.row_major bk bn) (fst sh) (fst (snd sh)) (2 * (shared/bk)) (bm/tm*(bn/tn)*warp_size) tid

inline_for_extraction noextract
fn subproducts_tc
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, scalar et_acc |}
  (bm bn bk: szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (aFrag : fragment et_ab FragA tm tn tk FragLRM)
  (#vaFrag : ematrix et_ab tm tk)
  (bFrag : fragment et_ab FragB tm tn tk FragLRM)
  (#vbFrag : ematrix et_ab tk tn)
  (accumFrag : fragment et_acc FragAcc tm tn tk FragLAcc)
  (#vaccumFrag : ematrix et_acc tm tn)
  (gA : gpu_matrix et_ab (R.row_major bm bk))
  (gB : gpu_matrix et_ab (R.row_major bk bn))
  (#eA : ematrix et_ab bm bk)
  (#eB : ematrix et_ab bk bn)
  (#fA #fB : perm)
  (arow: szlt (bm/tm))
  (bcol : szlt (bn/tn))
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure (valid_frag_et_comb et_ab et_acc) **
    aFrag |-> vaFrag **
    bFrag |-> vbFrag **
    accumFrag |-> vaccumFrag
  ensures
    exists* vaFrag' vbFrag' vaccumFrag'.
      aFrag |-> vaFrag' **
      bFrag |-> vbFrag' **
      accumFrag |-> vaccumFrag'
{
  let mut dotIdx : sz = 0sz;
  while (SZ.(!dotIdx <^ (bk/^tk)))
    invariant
      live aFrag **
      live bFrag **
      live accumFrag **
      live dotIdx
  {
    let a_tile = gpu_matrix_extract_tile_ro' gA (SZ.v tm) (SZ.v tk) (SZ.v arow) (SZ.v !dotIdx);
    let b_tile = gpu_matrix_extract_tile_ro' gB (SZ.v tk) (SZ.v tn) (SZ.v !dotIdx) (SZ.v bcol);

    mma_loadA aFrag a_tile;
    mma_loadB bFrag b_tile;
    mma_sync' aFrag bFrag accumFrag;

    with etA.
      assert (gpu_matrix_pts_to a_tile #fA etA);
      Pulse.Lib.Trade.elim_trade (a_tile |-> Frac fA etA) (gA |-> Frac fA eA);
    with etB.
      assert (gpu_matrix_pts_to b_tile #fB etB);
      Pulse.Lib.Trade.elim_trade (b_tile |-> Frac fB etB) (gB |-> Frac fB eB);

    dotIdx := !dotIdx +^ 1sz;
  };

  ()
}

inline_for_extraction noextract
fn epilogue
  (#et : Type0) {| scalar et |}
  (#rows : erased nat)
  // cols is concretized so using size is more succinct
  (#cols : sz)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (#tk : erased nat)
  (accumFrag : fragment et FragAcc tm tn tk FragLAcc)
  (gC : gpu_matrix et (R.row_major rows cols))
  (bid : szlt (rows/bm * (cols/bn)))
  (wid : szlt (bm/tm * (bn/tn)))
  requires
    gpu **
    live_warp_tile gC bm bn tm tn bid wid **
    (exists* vaccumFrag.
      accumFrag |-> vaccumFrag)
  ensures
    gpu **
    live_warp_tile gC bm bn tm tn bid wid **
    (exists* vaccumFrag.
      accumFrag |-> vaccumFrag)
{
  unfold live_warp_tile;

  (* Only create a tile in gC and write the accumulator values. In this version the input from gC
     was added by loading the tile into the accumulator before any other computations *)
  let w_tile = warp_tile (block_tile gC (SZ.v bm) (SZ.v bn) (SZ.v bid))
    (SZ.v tm) (SZ.v tn) (SZ.v wid);
  assert (rewrites_to w_tile (warp_tile (block_tile gC (SZ.v bm) (SZ.v bn) (SZ.v bid))
    (SZ.v tm) (SZ.v tn) (SZ.v wid)));

  // from looking at the type of mma_store, it is not clear that cols mut be concretizable
  // 1. know that strided_row_major needs concrete sizes
  // 2. search the code base for the appropriate instance and see which of the arguments
  //   must be concretizable
  // 3. figure out which expression is which argument and make concretizable accordingly
  mma_store accumFrag w_tile;

  // rewrite each w_tile as warp_tile (block_tile gC bm bn bid) tm tn tid;
  fold live_warp_tile;
  ()
}

inline_for_extraction noextract
fn kf
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared) {| clayout lA |}
  (gA : gpu_matrix et_ab lA)
  (#eA : ematrix et_ab rows shared)
  (#lB : mlayout shared cols) {| clayout lB |}
  (gB : gpu_matrix et_ab lB)
  (#eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c (R.row_major rows cols))
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn)*warp_size -1)))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn)*warp_size -1)))
  (#fA #fB : perm)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : szlt (rows/bm * (cols/bn)))
  (tid : szlt (bm/tm * (bn/tn) * warp_size))
  ()
  requires
    gpu **
    kpre gA eA gB eB gC bm bn bk tm tn tk fA fB sh bid tid **
    thread_id (bm/tm * (bn/tn) * warp_size) tid **
    block_id (rows/bm * (cols/bn)) bid
  ensures
    gpu **
    kpost gA eA gB eB gC bm bn bk tm tn fA fB sh bid tid **
    thread_id (bm/tm * (bn/tn) * warp_size) tid **
    block_id (rows/bm * (cols/bn)) bid
{
  let sarA : gpu_array et_ab (bm * bk) = fst sh;
  let sarB : gpu_array et_ab (bk * bn) = fst (snd sh);
  rewrite each fst sh as sarA;
  rewrite each fst (snd sh) as sarB;

  gpu_pts_to_ref sarA;
  gpu_pts_to_ref sarB;
  // This leads to a faillure to resolve the clayout when calling populate_shmem
  // let slA = R.row_major bm bk;
  // assert (rewrites_to slA (R.row_major bm bk));
  // let slB = R.row_major bk bn;
  // assert (rewrites_to slB (R.row_major bk bn));
  unfold barrier_tok (R.row_major bm bk) (R.row_major bk bn) sarA sarB 0 (bm/tm * (bn/tn) * warp_size) tid;

  gpu_matrix_abs' (R.row_major bm bk) sarA;
  let sA = from_array (R.row_major bm bk) sarA;
  rewrite each from_array (R.row_major bm bk) sarA as sA;

  gpu_matrix_abs' (R.row_major bk bn) sarB;
  let sB = from_array (R.row_major bk bn) sarB;
  rewrite each from_array (R.row_major bk bn) sarB as sB;

  let num_k_tiles = shared /^ bk;
  let num_n_tiles = cols /^ bn;
  let mrow = bid /^ num_n_tiles;
  let mcol = bid %^ num_n_tiles;

  let wid = tid /^ warp_sz;
  let warpRow = wid /^ (bn/^tn);
  let warpCol = wid %^ (bn/^tn);

  (* tensor core fragments *)
  let aFrag = __alloc_fragment et_ab FragA tm tn tk FragLRM;
  let bFrag = __alloc_fragment et_ab FragB tm tn tk FragLRM;
  let accumFrag = __alloc_fragment et_c FragAcc tm tn tk FragLAcc;

  (* get ownership over the thread's gC tile and load it into the accumulator *)
  unfold live_warp_tile;
  let t_tile = warp_tile (block_tile gC (SZ.v bm) (SZ.v bn) (SZ.v bid)) (SZ.v tm) (SZ.v tn) (tid / warp_size);
  assert (rewrites_to t_tile (warp_tile (block_tile gC (SZ.v bm) (SZ.v bn) (SZ.v bid)) (SZ.v tm) (SZ.v tn) (tid / warp_size)));
  mma_loadAccum accumFrag t_tile;
  fold live_warp_tile;

  let mut bkIdx  : sz = 0sz;
  while (SZ.(!bkIdx <^ num_k_tiles))
    invariant
      // Why can I not use `live` here?
      exists* (vbkIdx : SZ.t{vbkIdx <= num_k_tiles})
        (vaFrag : ematrix et_ab tm tk) (vbFrag : ematrix et_ab tk tn) (vaccumFrag : ematrix et_c tm tn).
        bkIdx |-> vbkIdx **
        aFrag |-> vaFrag **
        bFrag |-> vbFrag **
        accumFrag |-> vaccumFrag **
        (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. ((bm/tm*(bn/tn) * warp_size))) x) **
        (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. ((bm/tm*(bn/tn) * warp_size))) x) **
        B.barrier_tok (barrier_p sA sB ((bm/tm*(bn/tn)* warp_size))) (barrier_q sA sB ((bm/tm*(bn/tn) * warp_size))) (2 * !bkIdx) tid
  {
    (* This assert should not be needed. I don't know what effect it even has. *)
    assert B.barrier_tok (barrier_p sA sB (bm/tm * (bn/tn) * warp_size)) (barrier_q sA sB (bm/tm * (bn/tn) * warp_size)) (2 * !bkIdx) tid;
    even_2x !bkIdx;
    assert pure((2 * !bkIdx % 2 = 0) == true);
    rewrite
        (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. (bm/tm * (bn/tn) * warp_size)) x) **
        (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. (bm/tm * (bn/tn) * warp_size)) x)
      as barrier_p sA sB (bm/tm * (bn/tn) * warp_size) (2 * !bkIdx) tid;

    B.barrier_wait ();
    rewrite (barrier_q sA sB (bm/tm * (bn/tn) * warp_size) (2 * !bkIdx) tid)
        as live_tile_stride_cells sA (bm/tm * (bn/tn) * warp_size) tid **
           live_tile_stride_cells sB (bm/tm * (bn/tn) * warp_size) tid;

    copy_tiles_out_of_matrices bm bn bk sA sB gA gB mrow !bkIdx mcol (bm/^tm*^(bn/^tn)*^warp_sz) tid;

    assert (B.barrier_tok (barrier_p sA sB (bm/tm * (bn/tn) * warp_size)) (barrier_q sA sB (bm/tm * (bn/tn) * warp_size)) (2 * !bkIdx + 1) tid);
    odd_2x1 !bkIdx;
    assert (pure (odd (2 * !bkIdx + 1)));
    rewrite live_tile_stride_cells sA (bm/tm * (bn/tn) * warp_size) tid **
            live_tile_stride_cells sB (bm/tm * (bn/tn) * warp_size) tid
         as (barrier_p sA sB (bm/tm * (bn/tn) * warp_size) (2 * !bkIdx + 1) tid);

    B.barrier_wait ();

    even_2x (!bkIdx + 1);
    assert (pure (2 * (!bkIdx + 1) == 2 * !bkIdx + 2));
    assert (pure (even (2 * !bkIdx + 2)));
    rewrite (barrier_q sA sB (bm/tm * (bn/tn) * warp_size) (2 * !bkIdx + 1) tid)
    as
      (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. (bm/tm * (bn/tn) * warp_size)) x) **
      (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. (bm/tm * (bn/tn) * warp_size)) x);

    subproducts_tc bm bn bk tm tn tk aFrag bFrag accumFrag sA sB warpRow warpCol;

    bkIdx := !bkIdx +^ 1sz;
  };

  epilogue bm bn tm tn accumFrag gC bid wid;

  with vaFrag. assert aFrag |-> vaFrag; drop_ (aFrag |-> vaFrag);
  with vbFrag. assert bFrag |-> vbFrag; drop_ (bFrag |-> vbFrag);
  with vaccumFrag. assert accumFrag |-> vaccumFrag; drop_ (accumFrag |-> vaccumFrag);

  gpu_matrix_concr sA; rewrite each core sA as sarA;
  gpu_matrix_concr sB; rewrite each core sB as sarB;

  fold barrier_tok (R.row_major bm bk) (R.row_major bk bn) sarA sarB (2 * num_k_tiles) (bm/tm * (bn/tn) * warp_size) tid;

  rewrite each sarA as fst sh;
  rewrite each sarB as fst (snd sh);
  ()
}

ghost
fn setup
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c rows cols)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  // (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_: squash (SZ.fits (bm * bn)))
  (#_: squash (SZ.fits (rows * cols)))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/tm * (bn/tn) * warp_size})
  (fA fB : perm)
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (bid : natlt nblk)
             (tid : natlt nthr).
      kpre1 (*comb*) gA eA gB eB gC bm bn bk tm tn tk fA fB bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn block_setup
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c rows cols)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  // (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_: squash (SZ.fits (bm * bn)))
  (#_: squash (SZ.fits (rows * cols)))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/tm * (bn/tn) * warp_size})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    block_setup_tok nthr **
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpre1 (* comb *) gA eA gB eB gC bm bn bk tm tn tk fA fB bid tid)
  ensures
    block_setup_tok nthr **
    (forall+ (tid : natlt nthr).
      kpre (* comb *) gA eA gB eB gC bm bn bk tm tn tk fA fB sh bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn block_teardown
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c rows cols)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_: squash (SZ.fits (rows * cols)))
  (#_: squash (SZ.fits (bm * bn)))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/tm * (bn/tn) * warp_size})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
      kpost (* comb *) gA eA gB eB gC bm bn bk tm tn fA fB sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpost1 (* comb *) gA eA gB eB gC bm bn tm tn fA fB bid tid)
{
  admit();
}

ghost
fn teardown
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c rows cols)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (#_: squash (SZ.fits (rows * cols)))
  (#_: squash (SZ.fits (bm * bn)))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/tm * (bn/tn) * warp_size})
  (fA fB : perm)
  ()
  norewrite
  requires
    (forall+ (bid : natlt nblk)
             (tid : natlt nthr).
      kpost1 (* comb *) gA eA gB eB gC bm bn tm tn fA fB bid tid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    // underspec not implemented anyway
    (exists* eC'. gC |-> eC')
    // (gC |-> MS.mmcomb comb eC eA eB)
{
  // forevery_flatten #(natlt2 mrows mcols) #_ #(natlt tile)
  //   (fun bid tid -> kpost1 comb tile gA gB gC eA eB 1.0R bid tid);
  // forevery_unzip #(natlt2 mrows mcols & natlt tile) _ _;
  // forevery_unzip #(natlt2 mrows mcols & natlt tile) _ _;
  // forevery_tostar #(natlt2 mrows mcols & natlt tile) (fun _tid -> m4_pts_to gA #(1.0R /. mlayout_size lC) eA);

    // (fun (bid, tid) -> kpost1 comb tile gA gB gC eA eB 1.0R bid tid);
  admit();
}

inline_for_extraction noextract
let mk_kernel
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared) {| clayout lA |}
  (gA : gpu_matrix et_ab lA)
  (#eA : ematrix et_ab rows shared)
  (#lB : mlayout shared cols) {| clayout lB |}
  (gB : gpu_matrix et_ab lB)
  (#eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c (R.row_major rows cols))
  (#_ : squash (SZ.fits (rows * cols)))
  (#eC : ematrix et_c rows cols)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (#fA #fB : perm)
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  // WARNING the previous version was wrong, it was assuming that each
  //  thread computes tm*tk results similar to 2D-Blocktiling.
  // There is nothing that catches this.
  // (nthr : szp{SZ.v nthr == bm/tm * (bn/tn)})
  // correct: the amount of tensor core tiles in gC multiplied
  //  by the warp size (each warp computes one tile)
  (nthr : szp{SZ.v nthr == bm/tm*(bn/tn)*warp_size})
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_c FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_c))
  (#_ : squash (SZ.fits (bm*bk + nthr-1)))
  (#_ : squash (SZ.fits (bk*bn + nthr-1)))
  (#_ : squash (nblk <= max_blocks))
  (#_ : squash (nthr <= max_threads))
  ()
  : kernel_desc
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** (exists* eC'. gC |-> eC'))
= {
  nblk;
  nthr;

  shmems_desc = shmems_desc et_ab bm bn bk;

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt nthr). kpre1  gA eA gB eB gC bm bn bk tm tn tk fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt nthr). kpost1 gA eA gB eB gC bm bn tm tn fA fB bid tid);

  setup      = setup    gA eA gB eB gC eC bm bn bk tm tn tk nblk nthr fA fB;
  teardown   = teardown gA eA gB eB gC eC bm bn bk tm tn nblk nthr fA fB;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup gA eA gB eB gC eC bm bn bk tm tn tk nblk nthr fA fB;
  block_teardown = block_teardown gA eA gB eB gC eC bm bn bk tm tn nblk nthr fA fB ;

  kpre      = kpre  gA eA gB eB gC bm bn bk tm tn tk fA fB;
  kpost     = kpost gA eA gB eB gC bm bn bk tm tn fA fB;

  f = kf gA #eA gB #eB gC bm bn bk tm tn tk #() #() #fA #fB;
}
