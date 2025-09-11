module Kuiper.Poly.GEMM.Copy

#lang-pulse

open Kuiper
open Kuiper.Matrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.EMatrix

module SZ = FStar.SizeT

let live_cell
  (#et : Type0)
  (#rows #cols : nat)
  (#lm : mlayout rows cols)
  (sm : gpu_matrix et lm)
  (i : natlt rows)
  (j : natlt cols)
  : slprop
  = exists* v. gpu_matrix_pts_to_cell sm i j v

let div_ceil (a : nat) (b : pos) : erased int = (a + (b-1))/b

let live_tile_stride_cells
  (#et : Type0)
  (#rows #cols : nat)
  (#lm : mlayout rows cols)
  ([@@@mkey] m : gpu_matrix et lm)
  // (em : ematrix et rows cols)
  (nthr : nat)
  (tid : natlt nthr)
  : slprop
  =
  forall+ (it : natlt (div_ceil (rows*cols) nthr)).
    let flat_idx = tid + (it * nthr) <: nat in
    let i = flat_idx/cols <: nat in
    let j = flat_idx%cols <: nat in
      if (i < rows && j < cols)
      then live_cell m i j
      else emp

inline_for_extraction noextract
fn cp_matrix
  (#et : Type0) {| scalar et |}
  (rows cols: sz)
  (#lsrc #ldst : mlayout rows cols)
  {| clayout lsrc, clayout ldst |}
  (src : gpu_matrix et lsrc)
  (#esrc : ematrix et rows cols)
  (dst : gpu_matrix et ldst)
  (#fM : perm)
  (nthr : sz)
  (tid : szlt nthr)
  preserves
    gpu **
    // Where does rows*cols + nthr >= 1 come from?
    pure (SZ.fits (rows * cols + nthr - 1)) **
    src |-> Frac fM esrc **
    live_tile_stride_cells dst nthr tid
