module Kuiper.Poly.GEMM.TensorCore2D

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

open Pulse.Lib.Array
open Pulse.Lib.Trade

inline_for_extraction noextract
fn subproducts_tc_2d
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, scalar et_acc |}
  (bm bn bk: szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (wm : szp{wm * tm /? bm})
  (wn : szp{wn * tn /? bn})
  (aFrags : array (fragment et_ab FragA tm tn tk FragLRM))
  (#emAFrags : erased (seq (ematrix et_ab tm tk)))
  (bFrags : array (fragment et_ab FragB tm tn tk FragLRM))
  (#emBFrags : erased (seq (ematrix et_ab tk tn)))
  (accumFrags : array (fragment et_acc FragAcc tm tn tk FragLAcc))
  (#emAccumFrags : erased (seq (ematrix et_acc tm tn)))
  (gA : gpu_matrix et_ab (R.row_major bm bk))
  (gB : gpu_matrix et_ab (R.row_major bk bn))
  (#eA : ematrix et_ab bm bk)
  (#eB : ematrix et_ab bk bn)
  (#fA #fB : perm)
  (arow: szlt (bm/(wm*tm)))
  (bcol : szlt (bn/(wn*tn)))
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure (valid_frag_et_comb et_ab et_acc) **
    pure (Seq.length emAFrags == wm) **
    pure (Seq.length emBFrags == wn) ** 
    pure (Seq.length emAccumFrags == wm * wn) **
    pure (SZ.fits (wm * wn)) **
    array_fragment_pts_to aFrags emAFrags **
    array_fragment_pts_to bFrags emBFrags **
    array_fragment_pts_to accumFrags emAccumFrags
  ensures
    exists* emAFrags' emBFrags' emAccumFrags'.
      array_fragment_pts_to aFrags emAFrags' **
      array_fragment_pts_to bFrags emBFrags' **
      array_fragment_pts_to accumFrags emAccumFrags'
{
  let mut dotIdx : sz = 0sz;
  while (SZ.(!dotIdx <^ (bk/^tk)))
    invariant
      exists*
        (vdotIdx : sz{vdotIdx <= bk})
        (emAFrags : seq (ematrix et_ab tm tk))
        (emBFrags : seq (ematrix et_ab tk tn))
        (emAccumFrags : seq (ematrix et_acc tm tn)).
          pure (Seq.length emAFrags == wm) **
          pure (Seq.length emBFrags == wn) **
          pure (Seq.length emAccumFrags == wm*wn)**
          dotIdx |-> vdotIdx **
          array_fragment_pts_to aFrags emAFrags **
          array_fragment_pts_to bFrags emBFrags **
          array_fragment_pts_to accumFrags emAccumFrags
  {
    // TODO are the gpu_matrix_extract creating too many pointers or is everything inlined properly?!

    // create tile for tensor core tiles that belong to the warp
    let tile_for_tc_a_tiles = gpu_matrix_extract_tile_ro' gA (wm*tm) (SZ.v tk) (SZ.v arow) (SZ.v !dotIdx);
    let mut i0 = 0sz;
    while (SZ.(!i0 <^ wm))
      invariant
        exists*
          (vi : sz{vi <= wm})
          (emAFrags : seq (ematrix et_ab tm tk)).
            pure (Seq.length emAFrags == wm) **
            i0 |-> vi **
            array_fragment_pts_to aFrags emAFrags
    {
      let a_tile = gpu_matrix_extract_tile_ro' tile_for_tc_a_tiles (SZ.v tm) (SZ.v tk) (SZ.v !i0) 0;
      // Expected are only nats, but later on when the tile is used we need to concretize.
      // In this case wm*tm and 0 must be concretizable which means that either we have to write (SZ.v (wm*^tm)) and (SZ.v 0sz),
      // which is odd, because a nat is expected, or there must be type classes that can resolve this.
      assert (rewrites_to a_tile (
        gpu_matrix_subtile (
          gpu_matrix_subtile gA (wm*tm) (SZ.v tk) (SZ.v arow) (SZ.v !dotIdx))
          (SZ.v tm) (SZ.v tk) (SZ.v !i0) 0));

      // unfortunately, when inferring emAFrags, the solver cannot prove that !i0 is small enough
      with emAFrags. assert array_fragment_pts_to aFrags emAFrags;
      array_fragment_extract aFrags emAFrags !i0;

      mma_loadA aFrags.(!i0) a_tile;
      Pulse.Lib.Forall.elim_forall
        #(value_for et_ab FragA tm tn tk)
        (ematrix_subtile (ematrix_subtile eA (wm*tm) tk arow !dotIdx) tm tk !i0 0);

      ambig_trade_elim ();
      ambig_trade_elim ();

      i0 := !i0 +^ 1sz;
    };
    ambig_trade_elim ();

    // create tile for tensor core tiles that belong to the warp
    let tile_for_tc_b_tiles = gpu_matrix_extract_tile_ro' gB (SZ.v tk) (wn*tn) (SZ.v !dotIdx) (SZ.v bcol);
    let mut i1 = 0sz;
    while (SZ.(!i1 <^ wn))
      invariant
        exists*
          (vi : sz{vi <= wn})
          (emBFrags : seq (ematrix et_ab tk tn)).
            pure (Seq.length emBFrags == wn) **
            i1 |-> vi **
            array_fragment_pts_to bFrags emBFrags
    {
      let b_tile = gpu_matrix_extract_tile_ro' tile_for_tc_b_tiles (SZ.v tk) (SZ.v tn) 0 (SZ.v !i1);
      // Expected are only nats, but later on when the tile is used we need to concretize.
      // In this case wm*tm and 0 must be concretizable which means that either we have to write (SZ.v (wm*^tm)) and (SZ.v 0sz),
      // which is odd, because a nat is expected, or there must be type classes that can resolve this.
      assert (rewrites_to b_tile (
        gpu_matrix_subtile (
          gpu_matrix_subtile gB (SZ.v tk) (wn*tn) (SZ.v !dotIdx) (SZ.v bcol))
          (SZ.v tk) (SZ.v tn) 0 (SZ.v !i1)));

      // unfortunately, when inferring emBFrags, the solver cannot prove that !i1 is small enough
      with emBFrags. assert array_fragment_pts_to bFrags emBFrags;
      array_fragment_extract bFrags emBFrags !i1;

      mma_loadB bFrags.(!i1) b_tile;
      Pulse.Lib.Forall.elim_forall
        #(value_for et_ab FragB tm tn tk)
        (ematrix_subtile (ematrix_subtile eB tk (wn*tn) !dotIdx bcol) tk tn 0 !i1);

      ambig_trade_elim ();
      ambig_trade_elim ();

      i1 := !i1 +^ 1sz;
    };
    ambig_trade_elim ();

    let mut resIdxM = 0sz;
    while (SZ.(!resIdxM <^ wm))
      invariant
        exists*
          (vresIdxM : sz{vresIdxM <= wm})
          (emAccumFrags : seq (ematrix et_acc tm tn)).
            pure (Seq.length emAccumFrags == wm*wn) **
            resIdxM |-> vresIdxM **
            array_fragment_pts_to accumFrags emAccumFrags
    {
      let mut resIdxN = 0sz;
      while (SZ.(!resIdxN <^ wn))
        invariant
          exists*
            (vresIdxN : sz{vresIdxN <= wn})
            (emAccumFrags : seq (ematrix et_acc tm tn)).
              pure (Seq.length emAccumFrags == wm*wn) **
              resIdxN |-> vresIdxN **
              array_fragment_pts_to accumFrags emAccumFrags
      {
        with emAFrags. assert array_fragment_pts_to aFrags emAFrags;
        with emBFrags. assert array_fragment_pts_to bFrags emBFrags;
        with emAccumFrags. assert array_fragment_pts_to accumFrags emAccumFrags;
        array_fragment_extract_ro aFrags emAFrags !resIdxM;
        array_fragment_extract_ro bFrags emBFrags !resIdxN;
        array_fragment_extract accumFrags emAccumFrags (!resIdxM * wn + !resIdxN);

        let a_frag = aFrags.(!resIdxM);
        let b_frag = bFrags.(!resIdxN);
        let acc_frag = accumFrags.(!resIdxM *^ wn +^ !resIdxN);
        mma_sync' a_frag b_frag acc_frag;

        ambig_trade_elim ();
        ambig_trade_elim ();

        Pulse.Lib.Forall.elim_forall
          #(value_for et_acc FragAcc tm tn tk) 
          (MS.mma (Seq.index emAccumFrags (!resIdxM * wn + !resIdxN))
                  (Seq.index emAFrags !resIdxM)
                  (Seq.index emBFrags !resIdxN));
        
        ambig_trade_elim ();

        resIdxN := !resIdxN +^ 1sz;
      };

      resIdxM := !resIdxM +^ 1sz;
    };

    dotIdx := !dotIdx +^ 1sz;
  }
}

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
  (wm : nat{wm > 0 /\ wm * tm /? bm})
  (wn : nat{wn > 0 /\ wn * tn /? bn})
  (bid : natlt ((rows/bm) * (cols/bn)))
  (wid : natlt (bm/(wm*tm) * (bn/(wn*tn))))
  : slprop
  =
  live (warp_tile (block_tile gC bm bn bid) (wm*tm) (wn*tn) wid)

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
  (wm : szp{wm * tm /? bm})
  (wn : szp{wn * tn /? bn})
  (nthr : nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (fA fB : perm)
  (bid : enatlt (rows/bm * (cols/bn)))
  (tid : enatlt nthr)
  : slprop
  =
  pure (SZ.fits (rows * shared)) **
  pure (SZ.fits (shared * cols)) **
  pure (SZ.fits (wm * wn)) **
  pure (SZ.fits (wm * tm)) **
  pure (SZ.fits (wn * tn)) **
  pure (valid_frag_et_dims et_ab FragA tm tn tk) **
  pure (valid_frag_et_dims et_ab FragB tm tn tk) **
  pure (valid_frag_et_dims et_c FragAcc tm tn tk) **
  pure (valid_frag_et_comb et_ab et_c) **
  gA |-> Frac (fA /. nthr) eA **
  gB |-> Frac (fB /. nthr) eB **
  live_warp_tile gC bm bn tm tn wm wn bid (tid/warp_size)

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
  (wm : szp{wm * tm /? bm})
  (wn : szp{wn * tn /? bn})
  (nthr : nat {reveal nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (fA fB : perm)
  (bid : enatlt (rows/bm * (cols/bn)))
  (tid : enatlt nthr)
  : slprop
  =
  gA |-> Frac (fA /. nthr) eA **
  gB |-> Frac (fB /. nthr) eB **
  live_warp_tile gC bm bn tm tn wm wn bid (tid/warp_size)

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
  (wm : szp{wm * tm /? bm})
  (wn : szp{wn * tn /? bn})
  (nthr : nat {reveal nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt nthr)
  : slprop
  =
  kpre1 gA eA gB eB gC bm bn bk tm tn tk wm wn nthr fA fB bid tid **
  exists* (x : seq _). fst sh |-> Frac (1.0R /. nthr) x **
  exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. nthr) x **
  barrier_tok (R.row_major bm bk) (R.row_major bk bn) (fst sh) (fst (snd sh)) 0 (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) tid

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
  (wm : szp{wm * tm /? bm})
  (wn : szp{wn * tn /? bn})
  (nthr : nat {reveal nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt nthr)
  : slprop
  =
  kpost1 gA eA gB eB gC bm bn tm tn wm wn nthr fA fB bid tid **
  exists* (x : seq _). fst sh |-> Frac (1.0R /. nthr) x **
  exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. nthr) x **
  barrier_tok (R.row_major bm bk) (R.row_major bk bn) (fst sh) (fst (snd sh)) 0 (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) tid

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
  (wm : szp{wm * tm /? bm})
  (wn : szp{wn * tn /? bn})
  (accumFrags : array (fragment et FragAcc tm tn tk FragLAcc))
  (#emAccumFrags: erased (seq (ematrix et tm tn)))
  (gC : gpu_matrix et (R.row_major rows cols))
  (bid : szlt (rows/bm * (cols/bn)))
  (wid : szlt (bm/(wm*tm) * (bn/(wn*tn))))
  preserves
    gpu
  requires
    pure (Seq.length emAccumFrags == wm*wn) **
    pure (SZ.fits (wm * wn)) **
    live_warp_tile gC bm bn tm tn wm wn bid wid **
    array_fragment_pts_to accumFrags emAccumFrags
  ensures
    live_warp_tile gC bm bn tm tn wm wn bid wid **
    (exists* emAccumFrags'.
      pure (Seq.length emAccumFrags' == wm*wn) **
      array_fragment_pts_to accumFrags emAccumFrags')
{


  let mut i = 0sz;
  while (SZ.(!i <^ wm))
    invariant
      live_warp_tile gC bm bn tm tn wm wn bid wid
  {
    let mut j = 0sz;
    while (SZ.(!j <^ wn))
      invariant
        live_warp_tile gC bm bn tm tn wm wn bid wid
    {
      unfold live_warp_tile;

      // TODO does this create more pointer arithmetic than necessary?
      // tile in gC with all values that are computed by the warp
      // will be tiled into tiles for tensor core operations
      let tile_for_tc_tiles = warp_tile (block_tile gC (SZ.v bm) (SZ.v bn) (SZ.v bid))
        (wm*tm) (wn*tn) (SZ.v wid);
      rewrite each (warp_tile (block_tile gC (SZ.v bm) (SZ.v bn) (SZ.v bid))
      (wm*tm) (wn*tn) (SZ.v wid)) as tile_for_tc_tiles;

      gpu_matrix_extract_tile tile_for_tc_tiles tm tn !i !j;
      let tc_tile = gpu_matrix_subtile tile_for_tc_tiles (SZ.v tm) (SZ.v tn) (SZ.v !i) (SZ.v !j);
      rewrite each (gpu_matrix_subtile tile_for_tc_tiles (SZ.v tm) (SZ.v tn) (SZ.v !i) (SZ.v !j)) as tc_tile;

      with emAccumFrags. assert array_fragment_pts_to accumFrags emAccumFrags;
      array_fragment_extract_ro accumFrags emAccumFrags (!i * wn + !j);
      
      mma_store accumFrags.(!i *^ wn +^ !j) tc_tile;

      Pulse.Lib.Forall.elim_forall (Seq.Base.index emAccumFrags (!i * wn + !j)); 
      ambig_trade_elim ();
      ambig_trade_elim ();

      rewrite each tile_for_tc_tiles as warp_tile (block_tile gC (SZ.v bm) (SZ.v bn) (SZ.v bid))
        (wm*tm) (wn*tn)(SZ.v wid);
      fold live_warp_tile;

      j := !j +^ 1sz;
    };
    i := !i +^ 1sz;
  };

  ()
}

#push-options "--split_queries always --debug SMTFail"
// #push-options "--z3rlimit 40 --retry 5"
// #push-options "--print_implicits"
inline_for_extraction noextract
fn kf
  (#et_ab #et_c : Type0)
  {| scalar et_ab, sc : scalar et_c |}
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
  (wm : szp{wm * tm /? bm})
  (wn : szp{wn * tn /? bn})
  (nthr : erased nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (#_ : squash (SZ.fits (bm*bk + nthr-1)))
  (#_ : squash (SZ.fits (bk*bn + nthr-1)))
  (#fA #fB : perm)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : szlt (rows/bm * (cols/bn)))
  (tid : szlt nthr)
  ()
  requires
    gpu **
    kpre gA eA gB eB gC bm bn bk tm tn tk wm wn nthr fA fB sh bid tid **
    thread_id nthr tid **
    block_id (rows/bm * (cols/bn)) bid
  ensures
    gpu **
    kpost gA eA gB eB gC bm bn bk tm tn wm wn nthr fA fB sh bid tid **
    thread_id nthr tid **
    block_id (rows/bm * (cols/bn)) bid
{
  let sarA : gpu_array et_ab (bm * bk) = fst sh;
  let sarB : gpu_array et_ab (bk * bn) = fst (snd sh);
  rewrite each fst sh as sarA;
  rewrite each fst (snd sh) as sarB;

  gpu_pts_to_ref sarA;
  gpu_pts_to_ref sarB;

  unfold barrier_tok (R.row_major bm bk) (R.row_major bk bn) sarA sarB 0 nthr tid;

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
  let warpRow = wid /^ (bn/^(wn*^tn));
  let warpCol = wid %^ (bn/^(wn*^tn));

  (* tensor core fragments *)
  let aFrags = __alloc_array_fragment et_ab FragA tm tn tk FragLRM wm; 
  let bFrags = __alloc_array_fragment et_ab FragB tm tn tk FragLRM wn;
  let accFrags = __alloc_array_fragment et_c FragAcc tm tn tk FragLAcc (wm *^ wn);

  (* get ownership over the thread's gC tile and load it into the accumulator *)
  // unfold live_warp_tile;
  // let t_tile = warp_tile (block_tile gC (SZ.v bm) (SZ.v bn) (SZ.v bid)) (wm*tm) (wn*tn) (SZ.v wid);
  // assert (rewrites_to t_tile (warp_tile (block_tile gC (SZ.v bm) (SZ.v bn) (SZ.v bid)) (wm*tm) (wn*tn) (SZ.v wid)));
  // fold live_warp_tile;

  // fill accumulators with 0 for now
  let mut fi : sz = 0sz;
  while (SZ.(!fi <^ wm*^wn)) 
    invariant 
      exists* vaccFrags.
        pure (Seq.length vaccFrags == wm*wn) **
        array_fragment_pts_to accFrags vaccFrags
  {
    with vaccFrags. assert (array_fragment_pts_to accFrags vaccFrags);
    array_fragment_extract accFrags vaccFrags !fi;
    mma_fill accFrags.(!fi) sc.zero;

    Pulse.Lib.Forall.elim_forall
        #(value_for et_c FragAcc tm tn tk)
        (fill_value sc.zero);
    ambig_trade_elim();
  };

  let mut bkIdx  : sz = 0sz;
  while (SZ.(!bkIdx <^ num_k_tiles))
    invariant
      exists* (vbkIdx : SZ.t{vbkIdx <= num_k_tiles})
        (vaFrags : seq (ematrix et_ab tm tk))
        (vbFrags : seq (ematrix et_ab tk tn))
        (vaccFrags : seq (ematrix et_c tm tn)).
          bkIdx |-> vbkIdx **
          pure (Seq.length vaFrags == wm) **
          pure (Seq.length vbFrags == wn) **
          pure (Seq.length vaccFrags == wm*wn) **
          array_fragment_pts_to aFrags vaFrags **
          array_fragment_pts_to bFrags vbFrags **
          array_fragment_pts_to accFrags vaccFrags **
          (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. nthr) x) **
          (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. nthr) x) **
          B.barrier_tok (barrier_p sA sB nthr) (barrier_q sA sB nthr) (2 * !bkIdx) tid
  {
    (* This assert should not be needed. I don't know what effect it even has. *)
    assert B.barrier_tok (barrier_p sA sB nthr) (barrier_q sA sB nthr) (2 * !bkIdx) tid;
    even_2x !bkIdx;
    assert pure((2 * !bkIdx % 2 = 0) == true);
    rewrite
        (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. nthr) x) **
        (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. nthr) x)
      as barrier_p sA sB nthr (2 * !bkIdx) tid;

    B.barrier_wait ();
    rewrite (barrier_q sA sB nthr (2 * !bkIdx) tid)
        as live_tile_stride_cells sA nthr tid **
           live_tile_stride_cells sB nthr tid;

    populate_shmem bm bn bk tm tn sA sB gA gB mrow !bkIdx mcol tid;

    assert (B.barrier_tok (barrier_p sA sB nthr) (barrier_q sA sB nthr) (2 * !bkIdx + 1) tid);
    odd_2x1 !bkIdx;
    assert (pure (odd (2 * !bkIdx + 1)));
    rewrite live_tile_stride_cells sA nthr tid **
            live_tile_stride_cells sB nthr tid
         as (barrier_p sA sB nthr (2 * !bkIdx + 1) tid);

    B.barrier_wait ();

    even_2x (!bkIdx + 1);
    assert (pure (2 * (!bkIdx + 1) == 2 * !bkIdx + 2));
    assert (pure (even (2 * !bkIdx + 2)));
    rewrite (barrier_q sA sB nthr (2 * !bkIdx + 1) tid)
    as
      (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. nthr) x) **
      (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. nthr) x);

    subproducts_tc_2d bm bn bk tm tn tk wm wn aFrags bFrags accFrags sA sB warpRow warpCol;

    bkIdx := !bkIdx +^ 1sz;
  };

  epilogue bm bn tm tn wm wn accFrags gC bid wid;

  with vaFrags. assert array_fragment_pts_to aFrags vaFrags; drop_ (array_fragment_pts_to aFrags vaFrags);
  with vbFrags. assert array_fragment_pts_to bFrags vbFrags; drop_ (array_fragment_pts_to bFrags vbFrags);
  with vaccumFrags. assert array_fragment_pts_to accFrags vaccumFrags; drop_ (array_fragment_pts_to accFrags vaccumFrags);

  gpu_matrix_concr sA; rewrite each core sA as sarA;
  gpu_matrix_concr sB; rewrite each core sB as sarB;

  fold barrier_tok (R.row_major bm bk) (R.row_major bk bn) sarA sarB (2 * num_k_tiles) nthr tid;

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
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_: squash (SZ.fits (bm * bn)))
  (#_: squash (SZ.fits (rows * cols)))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (wm : szp{wm * tm /? bm})
  (wn : szp{wn * tn /? bn})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
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
      kpre1 gA eA gB eB gC bm bn bk tm tn tk wm wn nthr fA fB bid tid) **
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
  (wm : szp{wm * tm /? bm})
  (wn : szp{wn * tn /? bn})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    block_setup_tok nthr **
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpre1 gA eA gB eB gC bm bn bk tm tn tk wm wn nthr fA fB bid tid)
  ensures
    block_setup_tok nthr **
    (forall+ (tid : natlt nthr).
      kpre gA eA gB eB gC bm bn bk tm tn tk wm wn nthr fA fB sh bid tid) **
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
  (wm : szp{wm * tm /? bm})
  (wn : szp{wn * tn /? bn})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
      kpost (* comb *) gA eA gB eB gC bm bn bk tm tn wm wn nthr fA fB sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpost1 (* comb *) gA eA gB eB gC bm bn tm tn wm wn nthr fA fB bid tid)
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
  (wm : szp{wm * tm /? bm})
  (wn : szp{wn * tn /? bn})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (fA fB : perm)
  ()
  norewrite
  requires
    (forall+ (bid : natlt nblk)
             (tid : natlt nthr).
      kpost1 (* comb *) gA eA gB eB gC bm bn tm tn wm wn nthr fA fB bid tid) **
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

#push-options "--debug SMTFail --split_queries always"
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
  (wm : szp{wm * tm /? bm})
  (wn : szp{wn * tn /? bn})
  (#fA #fB : perm)
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
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
  block_pre  = (fun bid -> forall+ (tid : natlt nthr). kpre1  gA eA gB eB gC bm bn bk tm tn tk wm wn nthr fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt nthr). kpost1 gA eA gB eB gC bm bn tm tn wm wn nthr fA fB bid tid);

  setup      = setup    gA eA gB eB gC eC bm bn bk tm tn tk wm wn nblk nthr fA fB;
  teardown   = teardown gA eA gB eB gC eC bm bn bk tm tn wm wn nblk nthr fA fB;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup gA eA gB eB gC eC bm bn bk tm tn tk wm wn nblk nthr fA fB;
  block_teardown = block_teardown gA eA gB eB gC eC bm bn bk tm tn wm wn nblk nthr fA fB;

  kpre      = kpre  gA eA gB eB gC bm bn bk tm tn tk wm wn nthr fA fB;
  kpost     = kpost gA eA gB eB gC bm bn bk tm tn wm wn nthr fA fB;

  f = kf gA #eA gB #eB gC bm bn bk tm tn tk wm wn (SZ.v nthr) #() #() #fA #fB;
}
