module Kuiper.EMatrix
#lang-pulse

open Kuiper
open Kuiper.Shape
open Kuiper.Chest

let macc_mkM #et #rows #cols f i j = ()

let macc_mupd #et #rows #cols m i j v i' j' = ()

let lemma_equal_intro #et #rows #cols m1 m2 =
  introduce forall (idx : abs (rows @| cols @| INil)). acc m1 idx == acc m2 idx
  with (let (i, (j, ())) = idx in
        assert (acc2 m1 i j == acc2 m2 i j))

let lemma_approximates_intro #et #_ #_ #rows #cols m1 m2 =
  introduce forall (idx : abs (rows @| cols @| INil)). acc m1 idx %~ acc m2 idx
  with (let (i, (j, ())) = idx in
        assert (acc2 m1 i j %~ acc2 m2 i j))
