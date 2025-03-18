module Kuiper.EMatrix
#lang-pulse

(* An "erased" matrix, for specification purposes only *)

open Kuiper
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality

[@@erasable]
noeq
type ematrix (et:Type) (rows cols : erased nat) =
  | M : f:(natlt rows & natlt cols ^->> et)
     -> ematrix et rows cols

let mkM (#et:Type) (#rows #cols : nat)
  (f : natlt rows -> natlt cols -> GTot et)
  : ematrix et rows cols
  = M <| F.on_g _ <| fun (i, j) -> f i j

let macc (#et:Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  (i : natlt rows)
  (j : natlt cols)
  : GTot et
  = m.f (i, j)

let mupd (#et:Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  (i : nat{ i < rows })
  (j : nat{ j < cols })
  (v : et)
  : ematrix et rows cols
  = mkM fun i' j' ->
      if i' = i && j' = j
      then v
      else m.f (i', j')

val macc_pat (#et :Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  (i : natlt rows)
  (j : natlt cols)
  : Lemma (macc m i j == m.f (i, j))
          [SMTPat (m.f (i, j))]

let matrix_comb (#et:Type) (#rows #cols : nat)
  (f : binop et)
  (m1 m2 : ematrix et rows cols)
  : ematrix et rows cols
  = mkM fun i j -> f (macc m1 i j) (macc m2 i j)

let mtranspose (#et:Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  : ematrix et cols rows
  = mkM fun i j -> m.f (j, i)

val equal (#et #rows #cols : _) (m1 m2 : ematrix et rows cols) : prop

val lemma_equal_intro (#et #rows #cols : _)
  (m1 m2 : ematrix et rows cols)
  : Lemma (requires forall (i:natlt rows) (j:natlt cols). macc m1 i j == macc m2 i j)
          (ensures equal m1 m2)
          [SMTPat (equal m1 m2)]

val ematrix_ext #et #rows #cols
  (m1 m2 : ematrix et rows cols)
  : Lemma (requires equal m1 m2)
          (ensures m1 == m2)
          [SMTPat (equal m1 m2)]
