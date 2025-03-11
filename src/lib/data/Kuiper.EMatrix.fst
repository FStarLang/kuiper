module Kuiper.EMatrix
#lang-pulse

(* An "erased" matrix, for specification purposes only *)

open Kuiper
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality

let macc_pat m i j = ()

let ematrix_ext #et #rows #cols
  (m1 m2 : ematrix et rows cols)
  : Lemma (requires equal m1 m2)
          (ensures m1 == m2)
  = assert (F.feq_g m1.f m2.f)
