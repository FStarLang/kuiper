module Kuiper.GhostTranspose

#lang-pulse

open Kuiper
open Kuiper.Matrix.Poly
module Repr = Kuiper.Matrix.Reprs
open Kuiper.EMatrix

ghost
fn ghost_transpose1
  (#et:Type)
  (#rows #cols : nat)
  (gA : gpu_matrix et rows cols Repr.row_major)
  (#m : ematrix et rows cols)
  requires
    gA |-> m
  returns
    gA' : gpu_matrix et cols rows Repr.col_major
  ensures
    gA' |-> mtranspose m

ghost
fn ghost_transpose2
  (#et:Type)
  (#rows #cols : nat)
  (gA : gpu_matrix et rows cols Repr.col_major)
  (#m : ematrix et rows cols)
  requires
    gA |-> m
  returns
    gA' : gpu_matrix et cols rows Repr.row_major
  ensures
    gA' |-> mtranspose m
