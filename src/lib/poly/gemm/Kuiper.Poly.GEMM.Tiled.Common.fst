module Kuiper.Poly.GEMM.Tiled.Common

#lang-pulse

open Kuiper
open Kuiper.Matrix
open Kuiper.Matrix.Reprs
open Kuiper.Matrix.Tiling
open Kuiper.Poly.GEMM.Copy

open Kuiper.EMatrix

module SZ = FStar.SizeT

inline_for_extraction noextract
fn populate_shmem
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : erased nat)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (#slA : mlayout bm bk) {| clayout slA |}
  (#slB : mlayout bk bn) {| clayout slB |}
  (sA : gpu_matrix et slA)
  (sB : gpu_matrix et slB)
  (#lA : mlayout rows shared) {| clayout lA |}
  (#lB : mlayout shared cols) {| clayout lB |}
  (gA : gpu_matrix et lA)
  (#eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (#fA #fB : perm)
  (#eB : ematrix et shared cols)
  (tile_row : szlt (rows/bm))
  (tile_shared : szlt (shared/bk))
  (tile_col : szlt (cols/bn))
  (#nthr : erased nat)
  (#_ : squash (SZ.fits (bm*bk + nthr-1)))
  (#_ : squash (SZ.fits (bk*bn + nthr-1)))
  (tid : szlt nthr)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    thread_id nthr tid **
    live_tile_stride_cells sA nthr tid **
    live_tile_stride_cells sB nthr tid
{
  let tileA = gpu_matrix_extract_tile_ro' gA
    (SZ.v bm) (SZ.v bk) (SZ.v tile_row) (SZ.v tile_shared);
  cp_matrix bm bk tileA sA (get_bdim()) tid;

  let tileB = gpu_matrix_extract_tile_ro' gB
    (SZ.v bk) (SZ.v bn) (SZ.v tile_shared) (SZ.v tile_col);
  cp_matrix bk bn tileB sB (get_bdim()) tid;

  ambig_trade_elim ();
  ambig_trade_elim ();
  ();
}
