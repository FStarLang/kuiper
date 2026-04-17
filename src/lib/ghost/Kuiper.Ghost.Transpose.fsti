module Kuiper.Ghost.Transpose

// To deleted when Kuiper.Matrix goes away, tensors are the new thing.

#lang-pulse

open Kuiper
open Kuiper.Matrix
module Repr = Kuiper.Matrix.Reprs
open Kuiper.EMatrix

unfold
let row2col
  (#et : Type)
  (#rows #cols : erased nat)
  (m : gpu_matrix et (Repr.row_major rows cols))
  : gpu_matrix et (Repr.col_major cols rows) =
  from_array (Repr.col_major cols rows) (core m)

unfold
let col2row
  (#et : Type)
  (#rows #cols : erased nat)
  (m : gpu_matrix et (Repr.col_major rows cols))
  : gpu_matrix et (Repr.row_major cols rows) =
  from_array (Repr.row_major cols rows) (core m)

ghost
fn ghost_transpose1
  (#et:Type)
  (#rows #cols : nat)
  (gA : gpu_matrix et (Repr.row_major rows cols))
  (#m : ematrix et rows cols)
  requires
    gA |-> m
  ensures
    row2col gA |-> mtranspose m

ghost
fn ghost_transpose2
  (#et:Type)
  (#rows #cols : nat)
  (gA : gpu_matrix et (Repr.col_major rows cols))
  (#m : ematrix et rows cols)
  requires
    gA |-> m
  ensures
    col2row gA |-> mtranspose m

ghost
fn ghost_transpose1_back
  (#et:Type)
  (#rows #cols : nat)
  (gA : gpu_matrix et (Repr.row_major rows cols))
  (#m : ematrix et cols rows)
  requires
    row2col gA |-> m
  ensures
    gA |-> mtranspose m

ghost
fn ghost_transpose2_back
  (#et:Type)
  (#rows #cols : nat)
  (gA : gpu_matrix et (Repr.col_major rows cols))
  (#m : ematrix et cols rows)
  requires
    col2row gA |-> m
  ensures
    gA |-> mtranspose m
