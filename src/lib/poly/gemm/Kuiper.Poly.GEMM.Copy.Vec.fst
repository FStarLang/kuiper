module Kuiper.Poly.GEMM.Copy.Vec

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.Matrix
open Kuiper.Matrix.Vectorized
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
module SZ = Kuiper.SizeT
module GR = Pulse.Lib.GhostReference
open Pulse.Lib.Trade { trade }

let freeze (p : slprop) : slprop = p

let mul_inv_2 (a b c d:nat)
: Lemma (a == b * c * d /\ c<>0 /\ d<>0 ==> b == (a / d) / c)
= ()

let add_helper
  (i git nthr chunk_et : int)
  : Lemma (requires i == git * nthr * chunk_et)
          (ensures i + nthr * chunk_et == (git + 1) * nthr * chunk_et)
  = ()

let divides_helper
  (d : pos)
  (a b r c : nat)
  : Lemma (requires d /? a /\ d /? b /\ d /? c)
          (ensures d /? (a + b * r + c))
  = lemma_divides_product_l d b r;
    lemma_divides_sum d a (b * r);
    lemma_divides_sum d (a + b * r) c;
    ()

#push-options "--z3rlimit 45 --fuel 0 --ifuel 1"
inline_for_extraction noextract
fn cp_matrix_vec
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (rows cols: sz)
  (#lsrc #ldst : mlayout rows cols)
  {| clayout lsrc, clayout ldst |}
  {| src_str : strided_row_major lsrc |}
  (src : gpu_matrix et lsrc)
  (#f : perm)
  (#esrc : ematrix et rows cols)
  (dst : gpu_matrix et ldst)
  (nthr : sz)
  (tid : szlt nthr)
  preserves gpu
  preserves
    src |-> Frac f esrc
  requires
    pure (SZ.fits (rows * cols + nthr - 1)) **
    pure (chunk et /?+ cols) **
    pure (chunk et * nthr /?+ (rows * cols)) **
    pure (aligned 16 (core src)) **
    pure (rows * cols > 0)
  requires
    live_tile_stride_cells dst nthr tid
  ensures
    live_tile_stride_cells dst nthr tid
{
  // These should be exposed to the precondition and bubbled up.
  assume pure (chunk et /?+ src_str.stride);
  assume pure (chunk et /?+ src_str.offset);
  open FStar.SizeT;
  let mlen = rows *^ cols;

  assert pure (SZ.fits (tid * chunk et)); // ?
  let offset : sz = tid *^ chunk et;
  let mut i : sz = 0sz;

  (* It's very important to make the initializer and bound of this loop
  independent of the tid, this way NVCC can see clearly that it can be unrolled
  (when rows/cols are defined constants). Note, something like:
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 1024U; i2 += 256U) { ... }
  is NOT unrolled, even if NVCC could see statically that threadIdx.x < 32U, and
  therefore this is always 4 iterations. *)

  let git = Pulse.Lib.GhostReference.alloc 0;
  while ((!i <^ mlen))
    invariant
      live i ** live git **
      pure (SZ.v !i == GR.read git * nthr * chunk et) **
      pure (GR.read git <= (rows*cols) / (nthr * chunk et)) **
      live_tile_stride_cells dst nthr tid
  {
    assert pure (GR.read git < (rows*cols) / (chunk et * nthr));
    let vi = !i;
    assert pure (vi + offset < mlen); // prove this, it follows from rounding down, it works some times
    let mut local = [| zero #et #_; chunk et |];

    assert pure (SZ.fits (!i + nthr * chunk et));
    let row = (!i +^ offset) /^ cols; assert (rewrites_to row ((!i +^ offset) /^ cols));
    let col = (!i +^ offset) %^ cols; assert (rewrites_to col ((!i +^ offset) %^ cols));
    assert pure (chunk et /?+ cols);
    assume pure (chunk et /?+ col); // PROVE
    assert (pure (col + chunk et <= cols));
    assert pure (SZ.v offset == tid * chunk et);
    assert pure (row < rows);
    assert pure (col < cols - chunk et + 1);

    assert pure (chunk et * size #et == 16);
    src_str.pf row col;
    divides_helper (chunk et) src_str.offset src_str.stride row col;
    assert pure (chunk et /? cell_of_pos lsrc row col);
    assert pure (16 /?+ (cell_of_pos lsrc row col * size #et));

    gpu_matrix_vec_read src row col local;

    let ite : erased int = GR.read git;
    mul_inv_2 ite (!i) nthr (chunk et);

    unfold live_tile_stride_cells dst nthr tid;
    assert (pure (ite < divup (rows*cols) (chunk et * nthr)));
    assert pure (!i + offset = GR.read git * nthr * chunk et + tid * chunk et);
    assert pure (!i + offset = (GR.read git * nthr + tid) * chunk et);
    assert pure ((!i + offset)/chunk et = GR.read git * nthr + tid);
    FStar.Math.Lemmas.modulo_addition_lemma tid nthr (GR.read git);
    assert pure ((!i + offset)/chunk et % nthr = tid);
    forevery_extract #(idx:natlt (rows*cols){SZ.v tid == idx/(chunk et)%nthr /\ chunk et /?+ idx}) (!i + offset) _;

    // awful, here to match exactly what's in live_tile_stride_cells
    rewrite each ((!i + offset) % cols < cols - chunk et + 1) as true;

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
      assume pure ((!i + !k) / cols < rows); // sometimes works
      assume pure ((!i + !k) / cols == !i / cols); // should be provable (chunk et divides cols)
      assume (pure ((!i + !k) % cols == !i % cols + !k));

      unfold live_cell dst row (col +^ !k);
      // FIXME: for some reason, I have to pass s explicitly below,
      // or we get an error about not being to prove that !k is in
      // bounds (even if I assert it).
      with s. assert local |-> s;
      let v = Pulse.Lib.Array.op_Array_Access local !k #_ #s;
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
    let vgit = GR.read git;
    assert pure (SZ.fits (nthr * chunk et));
    assert pure (SZ.fits (vi + nthr * chunk et));
    i := vi +^ nthr *^ chunk et;
    GR.write git (vgit  + 1);

    assert pure (SZ.v vi == vgit * nthr * chunk et);
    add_helper vi vgit nthr (chunk et); // sigh
    assert pure (SZ.v !i == GR.read git * nthr * chunk et);
  };

  drop_ (git |-> _);

  ()
}
#pop-options
