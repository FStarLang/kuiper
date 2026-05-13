module Kuiper.EMatrix4
#lang-pulse

open Kuiper
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality

let macc_pat m i j k l = ()

let equal (#et #d0 #d1 #d2 #d3 : _) m1 m2 =
  forall (i:natlt d0) (j:natlt d1) (k:natlt d2) (l:natlt d3). macc m1 i j k l == macc m2 i j k l

let lemma_equal_intro m1 m2 = ()

let ext #et #d0 #d1 #d2 #d3
  (m1 m2 : ematrix4 et d0 d1 d2 d3)
  : Lemma (requires equal m1 m2)
          (ensures m1 == m2)
  = assert (F.feq_g m1.f m2.f)

let lemma_to_real_matrix_approximates (#et : Type0)
  {| scalar et, d : real_like et |}
  (#d0 #d1 #d2 #d3 : nat)
  (m : ematrix4 et d0 d1 d2 d3)
  : Lemma (ensures m %~ to_real_matrix m)
          [SMTPat (to_real_matrix m)]
  = ()
