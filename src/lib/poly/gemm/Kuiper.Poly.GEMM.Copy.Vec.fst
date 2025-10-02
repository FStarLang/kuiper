module Kuiper.Poly.GEMM.Copy.Vec

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.Matrix
open Kuiper.Matrix.Vectorized
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
module SZ = FStar.SizeT
module GR = Pulse.Lib.GhostReference
open Pulse.Lib.Trade { trade }

let freeze (p : slprop) : slprop = p

#push-options "--z3rlimit 30"
inline_for_extraction noextract
fn cp_matrix_vec
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (rows cols: sz)
  (#lsrc #ldst : mlayout rows cols)
  {| clayout lsrc, clayout ldst |}
  {| strided_row_major lsrc |}
  (src : gpu_matrix et lsrc)
  (#f : perm)
  (#esrc : ematrix et rows cols)
  (dst : gpu_matrix et ldst)
  (nthr : sz)
  (tid : szlt nthr)
  preserves
    gpu **
    pure (SZ.fits (rows * cols + nthr - 1)) **
    pure (chunk et /? cols) **
    src |-> Frac f esrc **
    live_tile_stride_cells dst nthr tid
{
  open FStar.SizeT;
  let mlen = rows *^ cols;

  assume pure (SZ.fits (tid * chunk et)); // ?
  let mut i : sz = tid *^ chunk et;

  // i is the flat_idx
  let git = Pulse.Lib.GhostReference.alloc 0;
  while ((!i <^ mlen))
    invariant
      live i ** live git **
      pure (SZ.v !i == tid * chunk et + GR.read git * nthr * chunk et) **
      live_tile_stride_cells dst nthr tid
  {
    let mut local = [| zero #et #_; chunk et |];

    let row = !i /^ cols; assert (rewrites_to row (!i /^ cols));
    let col = !i %^ cols; assert (rewrites_to col (!i %^ cols));
    assume (pure (col + chunk et <= cols)); // ?
    assert pure (row < rows && col < cols - chunk et + 1);

    gpu_matrix_vec_read src row col local;

    let ite : erased int = GR.read git;
    assert (pure (ite == (!i / chunk et - tid) / nthr));

    unfold live_tile_stride_cells dst nthr tid;
    assert (pure (ite < divup (rows*cols) (chunk et * nthr)));
    forevery_extract #(natlt (divup (rows*cols) (chunk et * nthr)))
      (reveal ite) _;

    // awful, here to match exactly what's in live_tile_stride_cells
    rewrite each ((tid * chunk et + ite * nthr * chunk et) / cols < rows
                  && (tid * chunk et + ite * nthr * chunk et) % cols < cols - chunk et + 1) as true;

    assert (live_chunk dst row col);

    // "freeze" the trade we have here to avoid ambiguity inside the loop
    with p q. assert (trade p q);
    rewrite trade p q as freeze (trade p q);

    let mut k = 0sz;
    while ((!k <^ chunk et))
      invariant live k ** pure (!k <= chunk et)
    {
      unfold live_chunk dst row col;
      forevery_extract #(natlt (chunk et)) !k _;
      assume (pure ((!i + !k) / cols < rows)); // check this
      assume (pure ((!i + !k) / cols == !i / cols)); // should be provable (chunk et divides cols)
      assume (pure ((!i + !k) % cols == !i % cols + !k));

      unfold live_cell dst row (col +^ !k);
      let v = Pulse.Lib.Array.(local.(!k));
      gpu_matrix_write_cell dst row (col +^ !k) v;
      fold live_cell dst row (col +^ !k);

      Pulse.Lib.Trade.elim_trade _ _;
      fold live_chunk dst row col;
      k := !k +^ 1sz;
    };

    unfold freeze;

    Pulse.Lib.Trade.elim_trade _ _;
    fold live_tile_stride_cells dst nthr tid;

    let vi = !i;
    assume pure (SZ.fits (nthr * chunk et));
    assume pure (SZ.fits (vi + nthr * chunk et));
    i := vi +^ nthr *^ chunk et;
    GR.write git (GR.read git + 1);
    assume pure
      (vi + nthr == tid * chunk et + (GR.read git + 1) * nthr * chunk et);
  };

  drop_ (git |-> _);

  ()
}
#pop-options
