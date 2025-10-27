module Kuiper.Poly.GEMM.TensorCore2D

#lang-pulse

open Kuiper

#set-options "--z3rlimit 50"

open Pulse.Lib.Array
open Pulse.Lib.Trade
open Kuiper.Matrix.Reprs.Type
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Matrix
open Kuiper.EMatrix { ematrix }
open Kuiper.VArray {
  varray,
  varray_pts_to,
  varray_pts_to_cell
}
open Kuiper.TensorCore
open Kuiper.Float16
open Kuiper.Matrix.Tiling
open Kuiper.Poly.GEMM.Copy.Vec
open Kuiper.Poly.GEMM.Tiled.Common.Vec

module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module R = Kuiper.Matrix.Reprs

open Kuiper.Poly.GEMM.TensorCore2D.KernelDesc

inline_for_extraction noextract
fn subproducts_tc_2d
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, scalar et_acc |}
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
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
  (arow : szlt (bm/(wm*tm)))
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
    aFrags |-> emAFrags **
    bFrags |-> emBFrags **
    accumFrags |-> emAccumFrags
  ensures
    exists* emAFrags' emBFrags' emAccumFrags'.
      pure (Seq.length emAFrags' == wm) **
      pure (Seq.length emBFrags' == wn) **
      pure (Seq.length emAccumFrags' == wm * wn) **
      aFrags |-> emAFrags' **
      bFrags |-> emBFrags' **
      accumFrags |-> emAccumFrags'
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
          aFrags |-> emAFrags **
          bFrags |-> emBFrags **
          accumFrags |-> emAccumFrags
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
            aFrags |-> emAFrags
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
      with emAFrags. assert aFrags |-> emAFrags;
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
            bFrags |-> emBFrags
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
      with emBFrags. assert bFrags |-> emBFrags;
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
            accumFrags |-> emAccumFrags
    {
      let mut resIdxN = 0sz;
      while (SZ.(!resIdxN <^ wn))
        invariant
          exists*
            (vresIdxN : sz{vresIdxN <= wn})
            (emAccumFrags : seq (ematrix et_acc tm tn)).
              pure (Seq.length emAccumFrags == wm*wn) **
              resIdxN |-> vresIdxN **
              accumFrags |-> emAccumFrags
      {
        with emAFrags. assert aFrags |-> emAFrags;
        with emBFrags. assert bFrags |-> emBFrags;
        with emAccumFrags. assert accumFrags |-> emAccumFrags;
        array_fragment_extract_ro aFrags emAFrags !resIdxM;
        array_fragment_extract_ro bFrags emBFrags !resIdxN;
        array_fragment_extract accumFrags emAccumFrags (!resIdxM * wn + !resIdxN);

        let a_frag = aFrags.(!resIdxM);
        let b_frag = bFrags.(!resIdxN);
        let acc_frag = accumFrags.(!resIdxM *^ wn +^ !resIdxN);
        mma_sync' a_frag b_frag acc_frag;

        ambig_trade_elim ();
        ambig_trade_elim ();

        with v. assert acc_frag `fragment_pts_to` v;
        Pulse.Lib.Forall.elim_forall
          #(value_for et_acc FragAcc tm tn tk)
          v;

        ambig_trade_elim ();

        resIdxN := !resIdxN +^ 1sz;
      };

      resIdxM := !resIdxM +^ 1sz;
    };

    dotIdx := !dotIdx +^ 1sz;
  }
}

#push-options "--z3rlimit 80"
inline_for_extraction noextract
fn epilogue
  (#et : Type0) {| scalar et |}
  (#rows : erased nat)
  // cols is concretized so using size is more succinct
  (#cols : sz)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
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
    invariant live i ** pure (!i <=^ wm)
  {
    let mut j = 0sz;
    while (SZ.(!j <^ wn))
      invariant live j ** pure (!j <=^ wn)
    {
      unfold live_warp_tile gC bm bn tm tn wm wn bid wid;

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

      with emAccumFrags. assert accumFrags `array_fragment_pts_to` emAccumFrags;
      let vi = !i;
      let vj = !j;
      let eidx : erased nat = vi * wn + vj;

      assert pure (vi < wm);
      assert pure (vj < wn);
      assert pure (eidx < wm * wn);
      assert pure (SZ.fits eidx);
      let idx = !i *^ wn +^ !j;

      array_fragment_extract_ro accumFrags emAccumFrags idx;
      mma_store accumFrags.(idx) tc_tile;

      Pulse.Lib.Forall.elim_forall (Seq.Base.index emAccumFrags idx);
      ambig_trade_elim ();
      ambig_trade_elim ();

      rewrite each tile_for_tc_tiles as warp_tile (block_tile gC (SZ.v bm) (SZ.v bn) (SZ.v bid))
        (wm*tm) (wn*tn)(SZ.v wid);
      fold live_warp_tile gC bm bn tm tn wm wn bid wid;
      j := !j +^ 1sz;
    };
    i := !i +^ 1sz;
  };

  ()
}
#pop-options

inline_for_extraction noextract
fn kf
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, sc : scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared) {| clayout lA, strided_row_major lA |}
  (gA : gpu_matrix et_ab lA)
  (#eA : ematrix et_ab rows shared)
  (#lB : mlayout shared cols) {| clayout lB, strided_row_major lB |}
  (gB : gpu_matrix et_ab lB)
  (#eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c (R.row_major rows cols))
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (bk /?+ shared))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (SZ.fits (rows * shared)))
  (#_ : squash (SZ.fits (rows * cols)))
  (#_ : squash (SZ.fits (shared * cols)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_c FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_c))
  (#fA #fB : perm)
  (nthr : erased nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm*bk + nthr-1)))
  (#_ : squash (SZ.fits (bk*bn + nthr-1)))
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : szlt (rows/bm * (cols/bn)))
  (tid : szlt nthr)
  ()
  requires
    gpu **
    kpre gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr sh bid tid **
    thread_id nthr tid **
    block_id (rows/bm * (cols/bn)) bid
  ensures
    gpu **
    kpost gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr sh bid tid **
    thread_id nthr tid **
    block_id (rows/bm * (cols/bn)) bid
{
  let (sarA, (sarB, _)) = sh;

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
      live fi **
      (exists* vaccFrags.
        pure (Seq.length vaccFrags == wm*wn) **
        accFrags |-> vaccFrags)
  {
    with vaccFrags. assert (accFrags |-> vaccFrags);
    array_fragment_extract accFrags vaccFrags !fi;
    mma_fill accFrags.(!fi) sc.zero;

    Pulse.Lib.Forall.elim_forall
        #(value_for et_c FragAcc tm tn tk)
        (fill_value sc.zero);
    ambig_trade_elim();

    fi := !fi +^ 1sz;
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
          aFrags |-> vaFrags **
          bFrags |-> vbFrags **
          accFrags |-> vaccFrags **
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

    copy_tiles_out_of_matrices_vec bm bn bk sA sB gA gB mrow !bkIdx mcol (bm/^(wm*^tm)*^(bn/^(wn*^tn))*^warp_sz) tid;

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

  rewrite each (tid / 32) as wid;
  epilogue bm bn bk tm tn tk wm wn accFrags gC bid wid;
  rewrite each v wid as (tid / 32);

  with vaFrags. assert aFrags |-> vaFrags; drop_ (aFrags |-> vaFrags);
  with vbFrags. assert bFrags |-> vbFrags; drop_ (bFrags |-> vbFrags);
  with vaccumFrags. assert accFrags |-> vaccumFrags; drop_ (accFrags |-> vaccumFrags);

  gpu_matrix_concr sA; rewrite each core sA as sarA;
  gpu_matrix_concr sB; rewrite each core sB as sarB;

  rewrite
    B.barrier_tok (barrier_p sA sB nthr)
      (barrier_q sA sB nthr)
      (2 * v !bkIdx)
      (v tid)
  as
    B.barrier_tok (barrier_p (from_array (R.row_major (v bm) (v bk)) sarA)
          (from_array (R.row_major (v bk) (v bn)) sarB)
          nthr)
      (barrier_q (from_array (R.row_major (v bm) (v bk)) sarA)
          (from_array (R.row_major (v bk) (v bn)) sarB)
          nthr)
      (2 * (shared / bk))
      (v tid);
  fold barrier_tok (R.row_major bm bk) (R.row_major bk bn) sarA sarB (2 * (shared / bk)) nthr tid;

  rewrite each sarA as fst sh;
  rewrite each sarB as fst (snd sh);
  ()
}

inline_for_extraction noextract
let mk_kernel
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared) {| clayout lA, strided_row_major lA |}
  (gA : gpu_matrix et_ab lA)
  (#eA : ematrix et_ab rows shared)
  (#lB : mlayout shared cols) {| clayout lB, strided_row_major lB |}
  (gB : gpu_matrix et_ab lB)
  (#eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c (R.row_major rows cols))
  (#_ : squash (SZ.fits (rows * cols)))
  (#eC : ematrix et_c rows cols)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (bk /?+ shared))
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#fA #fB : perm)
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (wm * wn)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
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
  block_pre  = (fun bid -> forall+ (tid : natlt nthr). kpre1  gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr bid tid);
  block_post = (fun bid -> forall+ (tid : natlt nthr). kpost1 gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr bid tid);

  setup      = setup    gA eA gB eB gC eC bm bn bk tm tn tk wm wn nblk nthr fA fB;
  teardown   = teardown gA eA gB eB gC eC bm bn bk tm tn tk wm wn nblk nthr fA fB;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup gA eA gB eB gC eC bm bn bk tm tn tk wm wn nblk nthr fA fB;
  block_teardown = block_teardown gA eA gB eB gC eC bm bn bk tm tn tk wm wn nblk nthr fA fB;

  kpre      = kpre  gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr ;
  kpost     = kpost gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr ;

  f = kf gA #eA gB #eB gC bm bn bk tm tn tk wm wn (SZ.v nthr);
}
