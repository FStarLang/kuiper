module Kuiper.EMatrix4
#lang-pulse

open Kuiper
open Kuiper.Shape
open Kuiper.Chest

let macc_mkM #et #d0 #d1 #d2 #d3 f i j k l = ()

let macc_mupd #et #d0 #d1 #d2 #d3 m i j k l v i' j' k' l' = ()

let lemma_equal_intro #et #d0 #d1 #d2 #d3 m1 m2 =
  introduce forall (idx : abs (d0 @| d1 @| d2 @| d3 @| INil)). acc m1 idx == acc m2 idx
  with (let (i, (j, (k, (l, ())))) = idx in
        assert (macc m1 i j k l == macc m2 i j k l))

let lemma_approximates_intro #et #_ #_ #d0 #d1 #d2 #d3 m1 m2 =
  introduce forall (idx : abs (d0 @| d1 @| d2 @| d3 @| INil)). acc m1 idx %~ acc m2 idx
  with (let (i, (j, (k, (l, ())))) = idx in
        assert (macc m1 i j k l %~ macc m2 i j k l))
