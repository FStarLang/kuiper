module Kuiper.GhostTranspose

#lang-pulse

open Kuiper
module Repr = Kuiper.Matrix.Reprs
open Kuiper.EMatrix
open Kuiper.Matrix.Common

inline_for_extraction noextract
fn ghost_transpose1
  (#et:Type)
  (#rows #cols : erased nat)
  (gA : gpu_matrix et (Repr.row_major rows cols))
  (#m : ematrix et rows cols)
  requires
    gA |-> m
  returns
    gA' : gpu_matrix et (Repr.col_major cols rows)
  ensures
    pure (core gA == core gA') **
    (gA' |-> mtranspose m)
{
  gpu_matrix_concr gA;
  assert (pure (Seq.equal
                  (to_seq (Repr.row_major rows cols) m)
                  (to_seq (Repr.col_major cols rows) (mtranspose m))));
  rewrite each to_seq (Repr.row_major rows cols) m
            as to_seq (Repr.col_major cols rows) (mtranspose m);
  let gA' = gpu_matrix_abs gA (Repr.col_major cols rows);
  gA'
}

inline_for_extraction noextract
fn ghost_transpose2
  (#et:Type)
  (#rows #cols : erased nat)
  (gA : gpu_matrix et (Repr.col_major rows cols))
  (#m : ematrix et rows cols)
  requires
    gA |-> m
  returns
    gA' : gpu_matrix et (Repr.row_major cols rows)
  ensures
    pure (core gA == core gA') **
    (gA' |-> mtranspose m)
{
  gpu_matrix_concr gA;
  assert (pure (Seq.equal
                  (to_seq (Repr.col_major rows cols) m)
                  (to_seq (Repr.row_major cols rows) (mtranspose m))));
  rewrite each to_seq (Repr.col_major rows cols) m
            as to_seq (Repr.row_major cols rows) (mtranspose m);
  let gA' = gpu_matrix_abs gA (Repr.row_major cols rows);
  gA'
}
