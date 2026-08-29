module Kuiper.EMatrix
open Kuiper.Container
#lang-pulse

open Kuiper
open Kuiper.Shape
open Kuiper.Chest

let macc_mkM #et #rows #cols f i j = ()

let chest_comb_acc2
  (#rows #cols : nat)
  (#t1 #t2 #t3 : Type)
  (f : t1 -> t2 -> t3)
  (c1 : chest2 t1 rows cols)
  (c2 : chest2 t2 rows cols)
  (i : natlt rows)
  (j : natlt cols)
  : Lemma (
      acc2 (chest_comb f c1 c2) i j
      == f (acc2 c1 i j) (acc2 c2 i j))
= Kuiper.Chest.acc_pat
    (fun (ij : abs (rows @| cols @| INil)) ->
      f (Kuiper.Chest.acc c1 ij) (Kuiper.Chest.acc c2 ij))
    (i, (j, ()))

let macc_mupd #et #rows #cols m i j v i' j' = ()

let lemma_equal_intro #et #rows #cols m1 m2 =
  introduce forall (idx : abs (rows @| cols @| INil)). acc m1 idx == acc m2 idx
  with (let (i, (j, ())) = idx in
        assert (acc2 m1 i j == acc2 m2 i j))

let lemma_approximates_intro #et #_ #_ #rows #cols m1 m2 =
  introduce forall (idx : abs (rows @| cols @| INil)). acc m1 idx %~ acc m2 idx
  with (let (i, (j, ())) = idx in
        assert (acc2 m1 i j %~ acc2 m2 i j))
