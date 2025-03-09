module Kuiper.GhostTranspose

#lang-pulse

open Kuiper
module SZ = FStar.SizeT
module Repr = Kuiper.Matrix.Reprs
open Kuiper.EMatrix

inline_for_extraction noextract
fn ghost_transpose1
  (#et:Type)
  (#rows #cols : erased nat)
  (gA : gpu_matrix et rows cols Repr.row_major)
  (#m : ematrix et rows cols)
  requires
    gA |-> m
  returns
    gA' : gpu_matrix et cols rows Repr.col_major
  ensures
    gA' |-> mtranspose m
{
  gpu_matrix_concr gA;
  assert (pure (Seq.equal
                  (to_seq Repr.row_major m)
                  (to_seq Repr.col_major (mtranspose m))));
  rewrite each to_seq Repr.row_major m
            as to_seq Repr.col_major (mtranspose m);
  let gA' = gpu_matrix_abs gA cols rows Repr.col_major;
  gA'
}

inline_for_extraction noextract
fn ghost_transpose2
  (#et:Type)
  (#rows #cols : erased nat)
  (gA : gpu_matrix et rows cols Repr.col_major)
  (#m : ematrix et rows cols)
  requires
    gA |-> m
  returns
    gA' : gpu_matrix et cols rows Repr.row_major
  ensures
    gA' |-> mtranspose m
{
  gpu_matrix_concr gA;
  assert (pure (Seq.equal
                  (to_seq Repr.col_major m)
                  (to_seq Repr.row_major (mtranspose m))));
  rewrite each to_seq Repr.col_major m
            as to_seq Repr.row_major (mtranspose m);
  let gA' = gpu_matrix_abs gA cols rows Repr.row_major;
  gA'
}
