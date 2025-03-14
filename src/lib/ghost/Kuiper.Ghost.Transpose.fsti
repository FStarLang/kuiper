module Kuiper.Ghost.Transpose

#lang-pulse

open Kuiper
open Kuiper.Matrix
module Repr = Kuiper.Matrix.Reprs
open Kuiper.EMatrix

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
