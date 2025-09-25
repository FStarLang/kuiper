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
fn copy_tiles_out_of_matrices
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : erased nat)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
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
  (nthr : sz)
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
  // get_bdim() is not specialized, when the block dimensions are specialized
  // cp_matrix bm bk tileA sA (get_bdim()) tid;
  cp_matrix bm bk tileA sA nthr tid;

  let tileB = gpu_matrix_extract_tile_ro' gB
    (SZ.v bk) (SZ.v bn) (SZ.v tile_shared) (SZ.v tile_col);
  // cp_matrix bk bn tileB sB (get_bdim()) tid;
  cp_matrix bk bn tileB sB nthr tid;

  ambig_trade_elim ();
  ambig_trade_elim ();
  ();
}

inline_for_extraction noextract
fn copy_tiles_out_of_matrices_one_cell_per_thread
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : erased nat)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  // every thread loads a single element for either matrix,
  (#slA : mlayout bm bk) {| clayout slA |}
  (#slB : mlayout bk bn) {| clayout slB |}
  (sA : gpu_matrix et slA)
  (sB : gpu_matrix et slB)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  {| clayout lA, clayout lB |}
  (gA : gpu_matrix et lA)
  (#eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (#eB : ematrix et shared cols)
  (#fA #fB : perm)
  (tile_row : szlt (rows/bm))
  (tile_shared : szlt (shared/bk))
  (tile_col : szlt (cols/bn))
  (#nthr : erased nat {nthr == bm*bk /\ nthr == bk*bn})
  (tid : szlt nthr)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    live_cell sA (tid/bk) (tid%bk) **
    live_cell sB (tid/bn) (tid%bn)
  ensures
    gpu_matrix_pts_to_cell sA (tid/bk) (tid%bk)
      (macc
        (ematrix_subtile eA bm bk tile_row tile_shared)
        (tid / bk)
        (tid % bk)) **
    gpu_matrix_pts_to_cell sB (tid/bn) (tid%bn)
      (macc
        (ematrix_subtile eB bk bn tile_shared tile_col)
        (tid/bn)
        (tid%bn))
{
    let tileA = gpu_matrix_extract_tile_ro' gA
      (SZ.v bm) (SZ.v bk) (SZ.v tile_row) (SZ.v tile_shared);
    cp_matrix_one_cell_per_thread tileA sA tid;

    let tileB = gpu_matrix_extract_tile_ro' gB
      (SZ.v bk) (SZ.v bn) (SZ.v tile_shared) (SZ.v tile_col);
    cp_matrix_one_cell_per_thread tileB sB tid;

    ambig_trade_elim ();
    ambig_trade_elim ();
}

