module Kuiper.Kernel.GEMM.Tiled.Common.Vec

#lang-pulse

open Kuiper
open Kuiper.Matrix
open Kuiper.Matrix.Reprs
open Kuiper.Matrix.Tiling
open Kuiper.Kernel.GEMM.Copy.Vec
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
module Trade = Pulse.Lib.Trade

open Kuiper.EMatrix

module SZ = Kuiper.SizeT

// Sad...
let divides_helper
  (d : pos)
  (a b r c : int)
  : Lemma (requires d /? a /\ d /? b /\ d /? c)
          (ensures d /? (a + b * r + c))
  = lemma_divides_product_l d b r;
    lemma_divides_sum d a (b * r);
    lemma_divides_sum d (a + b * r) c;
    ()

inline_for_extraction noextract
fn copy_tiles_out_of_matrices_vec
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (#rows #shared #cols : erased nat)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (#_ : squash (chunk et /? bk)) // extra req
  (#_ : squash (chunk et /? bn)) // extra req
  (#slA : mlayout bm bk) {| clayout slA |}
  (#slB : mlayout bk bn) {| clayout slB |}
  (sA : gpu_matrix et slA)
  (sB : gpu_matrix et slB)
  (#lA : mlayout rows shared) {| clayout lA, str_A : strided_row_major lA |}
  (#lB : mlayout shared cols) {| clayout lB, str_B : strided_row_major lB |}
  (gA : gpu_matrix et lA)
  (#eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (#fA #fB : perm)
  (#eB : ematrix et shared cols)
  (tile_row : szlt (rows/bm))
  (tile_shared : szlt (shared/bk))
  (tile_col : szlt (cols/bn))
  (nthr : szp)
  (#_ : squash (chunk et * nthr /?+ (bm * bk))) // extra req
  (#_ : squash (chunk et * nthr /?+ (bk * bn))) // extra req
  (#_ : squash (SZ.fits (bm*bk + nthr-1)))
  (#_ : squash (SZ.fits (bk*bn + nthr-1)))
  (tid : szlt nthr)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    thread_id nthr tid
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (aligned_strided_row_major (chunk et) str_A) **
    pure (aligned_strided_row_major (chunk et) str_B)
  requires
    live_strided_chunks sA nthr tid **
    live_strided_chunks sB nthr tid
  ensures
    own_strided_chunks sA (ematrix_subtile eA bm bk tile_row tile_shared) nthr tid **
    own_strided_chunks sB (ematrix_subtile eB bk bn tile_shared tile_col) nthr tid
{
  {
    unfold live_strided_chunks sA nthr tid;
    let tileA = gpu_matrix_extract_tile_ro' gA
      (SZ.v bm) (SZ.v bk) (SZ.v tile_row) (SZ.v tile_shared);
    // get_bdim() is not specialized, when the block dimensions are specialized
    // cp_matrix bm bk tileA sA (get_bdim()) tid;

    // Z3 needs some convicing.
    Kuiper.Divides.lemma_divides_product_l (chunk et) str_A.stride (tile_row * bm);
    Kuiper.Divides.lemma_divides_product_r (chunk et) tile_shared bk;
    divides_helper (chunk et) str_A.offset str_A.stride (tile_row * bm) (tile_shared * bk);
    assert pure (chunk et /?+ (str_A.offset + str_A.stride * (tile_row * bm) + (tile_shared * bk)));

    cp_matrix_vec bm bk tileA sA nthr tid;

    Trade.elim_trade _ _;
  };

  {
    unfold live_strided_chunks sB nthr tid;
    let tileB = gpu_matrix_extract_tile_ro' gB
      (SZ.v bk) (SZ.v bn) (SZ.v tile_shared) (SZ.v tile_col);

    Kuiper.Divides.lemma_divides_product_l (chunk et) str_B.stride (tile_shared * bk);
    Kuiper.Divides.lemma_divides_product_r (chunk et) tile_col bn;
    divides_helper (chunk et) str_B.offset str_B.stride (tile_shared * bk) (tile_col * bn);
    assert pure (chunk et /?+ (str_B.offset + str_B.stride * (tile_shared * bk) + (tile_col * bn)));

    cp_matrix_vec bk bn tileB sB nthr tid;

    Trade.elim_trade _ _;
  };

  ();
}
