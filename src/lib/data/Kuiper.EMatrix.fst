module Kuiper.EMatrix
#lang-pulse

(* An "erased" matrix, for specification purposes only *)

open Kuiper
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality

let macc_pat m i j = ()

let equal (#et #rows #cols : _) m1 m2 =
  forall (i:natlt rows) (j:natlt cols). macc m1 i j == macc m2 i j

let lemma_equal_intro m1 m2 = ()

let ematrix_ext #et #rows #cols
  (m1 m2 : ematrix et rows cols)
  : Lemma (requires equal m1 m2)
          (ensures m1 == m2)
  = assert (F.feq_g m1.f m2.f)

let lemma_to_real_matrix_approximates (#et : Type0)
  {| scalar et, d : real_like et |}
  (#rows #cols : nat)
  (m : ematrix et rows cols)
  : Lemma (ensures m %~ to_real_matrix m)
          [SMTPat (to_real_matrix m)]
  = ()
