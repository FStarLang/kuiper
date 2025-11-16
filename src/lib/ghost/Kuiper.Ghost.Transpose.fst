module Kuiper.Ghost.Transpose

#lang-pulse

open Kuiper
module Repr = Kuiper.Matrix.Reprs
open Kuiper.EMatrix
open Kuiper.Matrix.Common

open Kuiper.Injection

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
{
  gpu_matrix_concr gA;
  assert (pure (Seq.equal
                  (to_seq (Repr.row_major rows cols) m)
                  (to_seq (Repr.col_major cols rows) (mtranspose m))));
  rewrite core gA |-> to_seq (Repr.row_major rows cols) m
       as core gA |-> to_seq (Repr.col_major cols rows) (mtranspose m);
  gpu_matrix_abs (Repr.col_major cols rows) (core gA);
}

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
{
  gpu_matrix_concr gA;
  assert (pure (Seq.equal
                  (to_seq (Repr.col_major rows cols) m)
                  (to_seq (Repr.row_major cols rows) (mtranspose m))));
  rewrite core gA |-> to_seq (Repr.col_major rows cols) m
       as core gA |-> to_seq (Repr.row_major cols rows) (mtranspose m);
  gpu_matrix_abs (Repr.row_major cols rows) (core gA);
}

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
  (gA : gpu_matrix et (Repr.col_major rows cols))
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
