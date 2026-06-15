module Kuiper.EMatrix3
#lang-pulse

(* An "erased" 3-D matrix, for specification purposes only *)

open Kuiper
open Kuiper.Container
open Kuiper.Chest
include Kuiper.Chest {
  to_real_chest as to_real_matrix,
  equal,
  chest_comb as matrix_comb,
}
open Kuiper.Index
module EM = Kuiper.EMatrix

[@@erasable]
type ematrix3 (et:Type) (d0 d1 d2 : nat) =
  chest (d0 @| d1 @| d2 @| INil) et

unfold let t = ematrix3

let mkM (#et:Type) (#d0 #d1 #d2 : nat)
  (f : natlt d0 -> natlt d1 -> natlt d2 -> GTot et)
  : ematrix3 et d0 d1 d2
  = Chest.mk (d0 @| d1 @| d2 @| INil)
      fun (i, (j, (k, ()))) -> f i j k

let const_matrix (#et:Type) (#d0 #d1 #d2 : nat)
  (v:et)
  : ematrix3 et d0 d1 d2
  = Chest.const _ v

let macc (#et:Type) (#d0 #d1 #d2 : nat)
  (m : ematrix3 et d0 d1 d2)
  (i : natlt d0)
  (j : natlt d1)
  (k : natlt d2)
  : GTot et
  = Chest.acc m (i, (j, (k, ())))

let mupd (#et:Type) (#d0 #d1 #d2 : nat)
  (m : ematrix3 et d0 d1 d2)
  (i : nat{ i < d0 })
  (j : nat{ j < d1 })
  (k : nat{ k < d2 })
  (v : et)
  : ematrix3 et d0 d1 d2
  = Chest.upd m (i, (j, (k, ()))) v

(* ── Matrix-level bridges over the underlying chest ──────────────────────────
   The chest indexes matrices by the nested tuple [abs (d0 @| d1 @| d2 @| INil)]
   (i.e. [natlt d0 & (natlt d1 & (natlt d2 & unit))]), so the chest-level
   [equal], [approximates] and [acc] facts quantify over that nested index.
   These lemmas re-expose them in the flat [macc m i j k] form that the matrix
   API uses, bridging the trailing [unit] that SMT cannot erase on its own. *)

val macc_mkM (#et:Type) (#d0 #d1 #d2 : nat)
  (f : natlt d0 -> natlt d1 -> natlt d2 -> GTot et)
  (i : natlt d0) (j : natlt d1) (k : natlt d2)
  : Lemma (macc (mkM f) i j k == f i j k)
          [SMTPat (macc (mkM f) i j k)]

val macc_mupd (#et:Type) (#d0 #d1 #d2 : nat)
  (m : ematrix3 et d0 d1 d2)
  (i : nat{i < d0}) (j : nat{j < d1}) (k : nat{k < d2}) (v : et)
  (i' : natlt d0) (j' : natlt d1) (k' : natlt d2)
  : Lemma (macc (mupd m i j k v) i' j' k'
           == (if i' = i && j' = j && k' = k then v else macc m i' j' k'))
          [SMTPat (macc (mupd m i j k v) i' j' k')]

val lemma_equal_intro (#et #d0 #d1 #d2 : _)
  (m1 m2 : ematrix3 et d0 d1 d2)
  : Lemma (requires forall (i:natlt d0) (j:natlt d1) (k:natlt d2). macc m1 i j k == macc m2 i j k)
          (ensures equal m1 m2)
          [SMTPat (equal m1 m2)]

val lemma_approximates_intro
  (#et:Type0) {| scalar et, real_like et |} (#d0 #d1 #d2 : nat)
  (m1 : ematrix3 et d0 d1 d2)
  (m2 : ematrix3 real d0 d1 d2)
  : Lemma (requires forall (i:natlt d0) (j:natlt d1) (k:natlt d2). macc m1 i j k %~ macc m2 i j k)
          (ensures m1 %~ m2)
          [SMTPat (m1 %~ m2)]

(* Extract / update a single "page" (the 2-D slice at batch index i). *)
let slice_page (#et:Type) (#d0 #d1 #d2 : nat)
  (m : ematrix3 et d0 d1 d2) (i : natlt d0)
  : EM.ematrix et d1 d2
  = chest_slice 0 i m

let upd_page (#et:Type) (#d0 #d1 #d2 : nat)
  (m : ematrix3 et d0 d1 d2) (i : natlt d0)
  (p : EM.ematrix et d1 d2)
  : ematrix3 et d0 d1 d2
  = chest_update_slice 0 i m p

val slice_upd_page_same (#et:Type) (#d0 #d1 #d2 : nat)
  (m : ematrix3 et d0 d1 d2) (i : natlt d0)
  (p : EM.ematrix et d1 d2)
  : Lemma (ensures slice_page (upd_page m i p) i == p)
          [SMTPat (slice_page (upd_page m i p) i)]

val slice_upd_page_other (#et:Type) (#d0 #d1 #d2 : nat)
  (m : ematrix3 et d0 d1 d2) (i i' : natlt d0)
  (p : EM.ematrix et d1 d2)
  : Lemma (requires i' <> i)
          (ensures slice_page (upd_page m i p) i' == slice_page m i')
          [SMTPat (slice_page (upd_page m i p) i')]
