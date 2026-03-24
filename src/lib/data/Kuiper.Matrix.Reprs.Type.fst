module Kuiper.Matrix.Reprs.Type
#lang-pulse

open Kuiper
open Kuiper.Bijection
module SZ = Kuiper.SizeT

let full_layout_size_lt #rows #cols (l : mlayout rows cols)
  : Lemma (ensures l.len >= rows * cols)
= Kuiper.Enumerable.injection_implies_lte_cardinal (natlt rows & natlt cols) (natlt l.len) l.map

let full_layout_size #rows #cols (l : mlayout rows cols)
  : Lemma (requires is_full_layout l)
          (ensures  l.len == rows * cols)
          [SMTPat (is_full_layout l)]
= let b : bijection (natlt rows & natlt cols) (natlt l.len) = Kuiper.Bijection.bij_inj' l.map in
  Kuiper.Enumerable.bijection_implies_equal_cardinal (natlt rows & natlt cols) (natlt l.len) b

let clayout_fits (#rows #cols : nat) (#l : mlayout rows cols)
  (c : clayout l)
  : Lemma (SZ.fits (mlayout_size l))
  = () // This is now trivial, nice
