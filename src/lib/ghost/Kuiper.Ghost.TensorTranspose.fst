module Kuiper.Ghost.TensorTranspose

#lang-pulse

open Kuiper
open Kuiper.Array2
open Kuiper.EMatrix
open Kuiper.Tensor.Layout.Alg

open Kuiper.Injection

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
{
  array2_concr gA;
  assume_ (pure (Seq.equal
                  (to_seq (l2_row_major rows cols) m)
                  (to_seq (l2_col_major cols rows) (mtranspose m))));
  // FIXME: ^ should be obvious
  rewrite core gA |-> to_seq (l2_row_major rows cols) m
       as core gA |-> to_seq (l2_col_major cols rows) (mtranspose m);
  array2_abs (l2_col_major cols rows) (core gA);
}

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
{
  array2_concr gA;
  assume_ (pure (Seq.equal
                  (to_seq (l2_col_major rows cols) m)
                  (to_seq (l2_row_major cols rows) (mtranspose m))));
  // FIXME: ^ should be obvious
  rewrite core gA |-> to_seq (l2_col_major rows cols) m
       as core gA |-> to_seq (l2_row_major cols rows) (mtranspose m);
  array2_abs (l2_row_major cols rows) (core gA);
}

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
{
  ghost_transpose2 (row2col gA);
  rewrite
    col2row (row2col gA) |-> mtranspose m
  as
    gA |-> mtranspose m;
  ()
}

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
{
  ghost_transpose1 (col2row gA);
  rewrite
    row2col (col2row gA) |-> mtranspose m
  as
    gA |-> mtranspose m;
  ()
}
