module Kuiper.Poly.GEMM.Tiled.Common.Vec

#lang-pulse

open Kuiper
open Kuiper.Matrix
open Kuiper.Matrix.Reprs
open Kuiper.Matrix.Tiling
open Kuiper.Poly.GEMM.Copy.Vec
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }

open Kuiper.EMatrix

module SZ = Kuiper.SizeT
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
  (nthr : sz)
  (#_ : squash (chunk et * nthr /? (bm * bk))) // extra req
  (#_ : squash (chunk et * nthr /? (bk * bn))) // extra req
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
    pure (aligned 16 (core gB))
  requires
    live_tile_stride_cells sA nthr tid **
    live_tile_stride_cells sB nthr tid
  ensures
    live_tile_stride_cells sA nthr tid **
    live_tile_stride_cells sB nthr tid
{
  let tileA = gpu_matrix_extract_tile_ro' gA
    (SZ.v bm) (SZ.v bk) (SZ.v tile_row) (SZ.v tile_shared);
  // get_bdim() is not specialized, when the block dimensions are specialized
  // cp_matrix bm bk tileA sA (get_bdim()) tid;
  cp_matrix_vec bm bk tileA sA nthr tid;

  let tileB = gpu_matrix_extract_tile_ro' gB
    (SZ.v bk) (SZ.v bn) (SZ.v tile_shared) (SZ.v tile_col);
  // cp_matrix bk bn tileB sB (get_bdim()) tid;
  cp_matrix_vec bk bn tileB sB nthr tid;

  ambig_trade_elim ();
  ambig_trade_elim ();
  ();
}
