module Kuiper.Kernel.GEMM.Copy

#lang-pulse

open Kuiper
open Kuiper.Matrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.EMatrix

module SZ = Kuiper.SizeT

let live_cell
  (#et : Type0)
  (#rows #cols : nat)
  (#lm : mlayout rows cols)
  (gm : gpu_matrix et lm)
  (i : natlt rows)
  (j : natlt cols)
  : slprop
  = exists* v. gpu_matrix_pts_to_cell gm i j v

let live_strided_chunks
  (#et : Type0)
  (#rows #cols : nat)
  (#lm : mlayout rows cols)
  ([@@@mkey] m : gpu_matrix et lm)
  // (em : ematrix et rows cols)
  (nthr : nat)
  (tid : natlt nthr)
  : slprop
  =
  forall+ (it : natlt (divup (rows*cols) nthr)).
    let flat_idx = tid + (it * nthr) <: nat in
    let i = flat_idx/cols <: nat in
    let j = flat_idx%cols <: nat in
      if i < rows && j < cols
      then live_cell m i j
      else emp

inline_for_extraction noextract
fn cp_matrix
  (#et : Type0) {| scalar et |}
  (rows cols: sz)
  (#lsrc #ldst : mlayout rows cols)
  {| clayout lsrc, clayout ldst |}
  (src : gpu_matrix et lsrc)
  (#f : perm)
  (#esrc : ematrix et rows cols)
  (dst : gpu_matrix et ldst)
  (nthr : sz)
  (tid : szlt nthr)
  preserves
    gpu **
    // Where does rows*cols + nthr >= 1 come from?
    pure (SZ.fits (rows * cols + nthr - 1)) **
    src |-> Frac f esrc **
    live_strided_chunks dst nthr tid

inline_for_extraction noextract
fn cp_matrix_one_cell_per_thread
  (#et : Type0) {| scalar et |}
  (#rows #cols : szp)
  (#lsrc #ldst : mlayout rows cols)
  {| clayout lsrc, clayout ldst |}
  (src : gpu_matrix et lsrc)
  (#esrc : ematrix et rows cols)
  (dst : gpu_matrix et ldst)
  (#f : perm)
  (#nthr : erased nat{nthr == rows * cols})
  (tid : szlt nthr)
  preserves
    gpu **
    src |-> Frac f esrc
  requires
    live_cell dst (tid/cols) (tid%cols)
  ensures
    gpu_matrix_pts_to_cell dst (tid/cols) (tid%cols) (macc esrc (tid/cols) (tid%cols))
