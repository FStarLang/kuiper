module Kuiper.Ghost.TensorTranspose

#lang-pulse

open Kuiper
open Kuiper.Array2
open Kuiper.EMatrix
open Kuiper.Tensor.Layout.Alg

unfold
let row2col
  (#et : Type)
  (#rows #cols : erased nat)
  (m : array2 et (l2_row_major rows cols))
  : array2 et (l2_col_major cols rows) =
  from_array (l2_col_major cols rows) (core m)

unfold
let col2row
  (#et : Type)
  (#rows #cols : erased nat)
  (m : array2 et (l2_col_major rows cols))
  : array2 et (l2_row_major cols rows) =
  from_array (l2_row_major cols rows) (core m)

ghost
fn ghost_transpose1
  (#et:Type)
  (#rows #cols : nat)
  (gA : array2 et (l2_row_major rows cols))
  (#m : ematrix et rows cols)
  requires
    gA |-> m
  ensures
    row2col gA |-> mtranspose m

ghost
fn ghost_transpose2
  (#et:Type)
  (#rows #cols : nat)
  (gA : array2 et (l2_col_major rows cols))
  (#m : ematrix et rows cols)
  requires
    gA |-> m
  ensures
    col2row gA |-> mtranspose m

ghost
fn ghost_transpose1_back
  (#et:Type)
  (#rows #cols : nat)
  (gA : array2 et (l2_row_major rows cols))
  (#m : ematrix et cols rows)
  requires
    row2col gA |-> m
  ensures
    gA |-> mtranspose m

ghost
fn ghost_transpose2_back
  (#et:Type)
  (#rows #cols : nat)
  (gA : array2 et (l2_col_major rows cols))
  (#m : ematrix et cols rows)
  requires
    col2row gA |-> m
  ensures
    gA |-> mtranspose m
