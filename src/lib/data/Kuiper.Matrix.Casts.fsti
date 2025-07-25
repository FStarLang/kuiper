module Kuiper.Matrix.Casts
#lang-pulse

open Kuiper

module EM = Kuiper.EMatrix
module M  = Kuiper.Matrix
module M4 = Kuiper.Matrix4

inline_for_extraction let x = 0 (* make sure interface is inlined *)

inline_for_extraction noextract
fn m2_to_m4
  (trows tcols mrows mcols : erased nat)
  (#et : Type0) {| scalar et |}
  (#lA : M4.mlayout4 mrows mcols trows tcols)
  (gA : M.gpu_matrix et lA)
  (#eA : EM.ematrix et _ _)
  (#f : perm)
  requires
    gA |-> Frac f eA
  returns
    gA4 : M4.gpu_matrix et lA
  ensures
    (gA4 |-> Frac f eA) **
    pure (M4.core gA4 == M.core gA)

inline_for_extraction noextract
fn m4_to_m2
  (#trows #tcols #mrows #mcols : erased nat)
  (#et : Type0) {| scalar et |}
  (#lA : M4.mlayout4 mrows mcols trows tcols)
  (gA4 : M4.gpu_matrix et lA)
  (#eA : EM.ematrix et _ _)
  (#f : perm)
  requires
    gA4 |-> Frac f eA
  returns
    gA : M.gpu_matrix et lA
  ensures
    (gA |-> Frac f eA) **
    pure (M4.core gA4 == M.core gA)
