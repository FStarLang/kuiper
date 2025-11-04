module Kuiper.Poly.GEMM.Copy.Vec

#lang-pulse

open Kuiper
open Kuiper.Matrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.EMatrix
open Kuiper.Array.Vectorized

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

let live_chunk
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#lm : mlayout rows cols)
  ([@@@mkey] m : gpu_matrix et lm)
  // (em : ematrix et rows cols)
  (i : natlt rows)
  (j : natlt (cols - chunk et + 1))
  : slprop
=
  forall+ (k : natlt (chunk et)).
    live_cell m i (j + k)

let live_tile_stride_cells
  (#et : Type0) {| sized et, hvc: has_vec_cpy et |}
  (#rows #cols : nat)
  (#lm : mlayout rows cols)
  ([@@@mkey] m : gpu_matrix et lm)
  // (#_ : squash (chunk et /?+ cols))
  // ^ We will have this in any interesting client, but
  // since it's not needed here let's just skip it, to make
  // the type more defined.
  // (em : ematrix et rows cols)
  (nthr : nat)
  (tid : natlt nthr)
  : slprop
=
  // typeclass constraint not resolved
  forall+ (idx : natlt (rows*cols) {tid == idx/(chunk et #_ #hvc) % nthr /\ chunk et /?+ idx}).
    if (idx%cols < cols - chunk et + 1)
    // this fact about (idx%cols) should be provable here, we add it only to avoid a
    // hard VC at this point, punting the proof on to the user (where it's
    // hopefully easier)
    then live_chunk m (idx/cols) (idx%cols)
    else emp
 
//  // Number of chunks each thread will copy
//  forall+ (it : natlt (divup (rows*cols) (chunk et * nthr))).
//    let flat_idx = tid * chunk et + it * nthr * chunk et <: nat in
//    let i : nat = flat_idx / cols in
//    let j : nat = flat_idx % cols in
//    if i < rows
//       && j < cols - chunk et + 1
//       // this fact about j should be provable here, we add it only to avoid a
//       // hard VC at this point, punting the proof on to the user (where it's
//       // hopefully easier)
//    then live_chunk m i j
//    else emp

// NB: The scalar constraint is only here so we can use 'zero' as an initializer
// for a local array... would be gone if we had uninitialized local arrays.
inline_for_extraction noextract
fn cp_matrix_vec
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (rows cols: sz)
  // (#_ : squash (chunk et /? cols))
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
