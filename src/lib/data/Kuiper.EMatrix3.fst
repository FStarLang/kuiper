module Kuiper.EMatrix3
#lang-pulse

open Kuiper
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality

let macc_pat m i j k = ()

let equal (#et #d0 #d1 #d2 : _) m1 m2 =
  forall (i:natlt d0) (j:natlt d1) (k:natlt d2). macc m1 i j k == macc m2 i j k

let lemma_equal_intro m1 m2 = ()

let ext #et #d0 #d1 #d2
  (m1 m2 : ematrix3 et d0 d1 d2)
  : Lemma (requires equal m1 m2)
          (ensures m1 == m2)
  = assert (F.feq_g m1.f m2.f)

let lemma_to_real_matrix_approximates (#et : Type0)
  {| scalar et, d : real_like et |}
  (#d0 #d1 #d2 : nat)
  (m : ematrix3 et d0 d1 d2)
  : Lemma (ensures m %~ to_real_matrix m)
          [SMTPat (to_real_matrix m)]
  = ()

let slice_upd_page_same #et #d0 #d1 #d2 m i p =
  assert (slice_page (upd_page m i p) i `EM.equal` p)

let slice_upd_page_other #et #d0 #d1 #d2 m i i' p =
  assert (slice_page (upd_page m i p) i' `EM.equal` slice_page m i')
