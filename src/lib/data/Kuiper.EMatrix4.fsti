module Kuiper.EMatrix4
#lang-pulse

(* An "erased" 4-D matrix, for specification purposes only *)

open Kuiper
open Kuiper.Container
open Kuiper.Chest
include Kuiper.Chest {
  to_real_chest as to_real_matrix,
  equal,
  chest_comb as matrix_comb,
}
open Kuiper.Index

[@@erasable]
type ematrix4 (et:Type) (d0 d1 d2 d3 : nat) =
  chest (d0 @| d1 @| d2 @| d3 @| INil) et

unfold let t = ematrix4

let mkM (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (f : natlt d0 -> natlt d1 -> natlt d2 -> natlt d3 -> GTot et)
  : ematrix4 et d0 d1 d2 d3
  = Chest.mk (d0 @| d1 @| d2 @| d3 @| INil)
      fun (i, (j, (k, (l, ())))) -> f i j k l

let const_matrix (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (v:et)
  : ematrix4 et d0 d1 d2 d3
  = Chest.const _ v

let macc (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (m : ematrix4 et d0 d1 d2 d3)
  (i : natlt d0)
  (j : natlt d1)
  (k : natlt d2)
  (l : natlt d3)
  : GTot et
  = Chest.acc m (i, (j, (k, (l, ()))))

let mupd (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (m : ematrix4 et d0 d1 d2 d3)
  (i : nat{ i < d0 })
  (j : nat{ j < d1 })
  (k : nat{ k < d2 })
  (l : nat{ l < d3 })
  (v : et)
  : ematrix4 et d0 d1 d2 d3
  = Chest.upd m (i, (j, (k, (l, ())))) v

(* ── Matrix-level bridges over the underlying chest ──────────────────────────
   The chest indexes matrices by the nested tuple
   [abs (d0 @| d1 @| d2 @| d3 @| INil)], so the chest-level [equal],
   [approximates] and [acc] facts quantify over that nested index. These lemmas
   re-expose them in the flat [macc m i j k l] form that the matrix API uses,
   bridging the trailing [unit] that SMT cannot erase on its own. *)

val macc_mkM (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (f : natlt d0 -> natlt d1 -> natlt d2 -> natlt d3 -> GTot et)
  (i : natlt d0) (j : natlt d1) (k : natlt d2) (l : natlt d3)
  : Lemma (macc (mkM f) i j k l == f i j k l)
          [SMTPat (macc (mkM f) i j k l)]

val macc_mupd (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (m : ematrix4 et d0 d1 d2 d3)
  (i : nat{i < d0}) (j : nat{j < d1}) (k : nat{k < d2}) (l : nat{l < d3}) (v : et)
  (i' : natlt d0) (j' : natlt d1) (k' : natlt d2) (l' : natlt d3)
  : Lemma (macc (mupd m i j k l v) i' j' k' l'
           == (if i' = i && j' = j && k' = k && l' = l then v else macc m i' j' k' l'))
          [SMTPat (macc (mupd m i j k l v) i' j' k' l')]

val lemma_equal_intro (#et #d0 #d1 #d2 #d3 : _)
  (m1 m2 : ematrix4 et d0 d1 d2 d3)
  : Lemma (requires forall (i:natlt d0) (j:natlt d1) (k:natlt d2) (l:natlt d3).
                      macc m1 i j k l == macc m2 i j k l)
          (ensures equal m1 m2)
          [SMTPat (equal m1 m2)]

val lemma_approximates_intro
  (#et:Type0) {| scalar et, real_like et |} (#d0 #d1 #d2 #d3 : nat)
  (m1 : ematrix4 et d0 d1 d2 d3)
  (m2 : ematrix4 real d0 d1 d2 d3)
  : Lemma (requires forall (i:natlt d0) (j:natlt d1) (k:natlt d2) (l:natlt d3).
                      macc m1 i j k l %~ macc m2 i j k l)
          (ensures m1 %~ m2)
          [SMTPat (m1 %~ m2)]
