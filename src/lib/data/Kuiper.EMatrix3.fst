module Kuiper.EMatrix3
#lang-pulse

open Kuiper
open Kuiper.Index
open Kuiper.Chest
module EM = Kuiper.EMatrix

let macc_mkM #et #d0 #d1 #d2 f i j k = ()

let macc_mupd #et #d0 #d1 #d2 m i j k v i' j' k' = ()

let lemma_equal_intro #et #d0 #d1 #d2 m1 m2 =
  introduce forall (idx : abs (d0 @| d1 @| d2 @| INil)). acc m1 idx == acc m2 idx
  with (let (i, (j, (k, ()))) = idx in
        assert (macc m1 i j k == macc m2 i j k))

let lemma_approximates_intro #et #_ #_ #d0 #d1 #d2 m1 m2 =
  introduce forall (idx : abs (d0 @| d1 @| d2 @| INil)). acc m1 idx %~ acc m2 idx
  with (let (i, (j, (k, ()))) = idx in
        assert (macc m1 i j k %~ macc m2 i j k))

let slice_upd_page_same #et #d0 #d1 #d2 m i p =
  assert (slice_page (upd_page m i p) i `EM.equal` p)

let slice_upd_page_other #et #d0 #d1 #d2 m i i' p =
  assert (slice_page (upd_page m i p) i' `EM.equal` slice_page m i')
