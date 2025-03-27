module Kuiper.Matrix.Casts
#lang-pulse

open Kuiper

module EM = Kuiper.EMatrix
module M  = Kuiper.Matrix
module M4 = Kuiper.Matrix4

inline_for_extraction let x = 0 (* make sure interface is inlined *)

inline_for_extraction noextract
fn m2_to_m4
  (tile : erased nat)
  (mrows mcols : erased nat)
  (#et : Type0) {| scalar et |}
  (#lA : M4.mlayout4 mrows mcols tile tile)
  (gA : M.gpu_matrix et lA)
  (#eA : EM.ematrix et _ _)
  (#f : perm)
  requires
    gA |-> Fraction f eA
  returns
    gA4 : M4.gpu_matrix et lA
  ensures
    (gA4 |-> Fraction f eA) **
    pure (M4.core gA4 == M.core gA)

inline_for_extraction noextract
fn m4_to_m2
  (tile : erased nat)
  (mrows mcols : erased nat)
  (#et : Type0) {| scalar et |}
  (#lA : M4.mlayout4 mrows mcols tile tile)
  (gA4 : M4.gpu_matrix et lA)
  (#eA : EM.ematrix et _ _)
  (#f : perm)
  requires
    gA4 |-> Fraction f eA
  returns
    gA : M.gpu_matrix et lA
  ensures
    (gA |-> Fraction f eA) **
    pure (M4.core gA4 == M.core gA)
