module Kuiper.EMatrix
#lang-pulse

(* An "erased" matrix, for specification purposes only *)

open Kuiper
open Kuiper.Container
open Kuiper.Chest
open Kuiper.Shape
include Kuiper.Chest {
  to_real_chest as to_real_matrix,
  equal,
  chest_comb as ematrix_comb,
  chest2
}

let const_matrix (#et:Type) (#rows #cols : nat)
  (v:et)
  : chest2 et rows cols
  = Chest.const _ v

let matrix_comb (#et:Type) (#rows #cols : nat)
  (f : binop et)
  (m1 m2 : chest2 et rows cols)
  : chest2 et rows cols
  = Chest.chest_comb f m1 m2

let mtranspose (#et:Type) (#rows #cols : nat)
  (m : chest2 et rows cols)
  : chest2 et cols rows
  = mk2 fun i j -> acc2 m j i

let ematrix_row
  (#et : Type0)
  (#rows #cols : erased nat)
  (em : chest2 et rows cols)
  (i : natlt rows)
  : GTot (lseq et cols)
  = Seq.init_ghost cols (fun j -> acc2 em i j)

let ematrix_col
  (#et : Type0)
  (#rows #cols : erased nat)
  (em : chest2 et rows cols)
  (j : natlt cols)
  : GTot (lseq et rows)
  = Seq.init_ghost rows (fun i -> acc2 em i j)

let ematrix_upd_row
  (#et : Type0)
  (#rows #cols : erased nat)
  (em : chest2 et rows cols)
  (i : natlt rows)
  (new_row : lseq et cols)
  : chest2 et rows cols
  = mk2 fun i' j ->
      if i' = i
      then Seq.index new_row j
      else acc2 em i' j

let ematrix_upd_col
  (#et : Type0)
  (#rows #cols : erased nat)
  (em : chest2 et rows cols)
  (j : natlt cols)
  (new_col : lseq et rows)
  : chest2 et rows cols
  = mk2 fun i j' ->
      if j' = j
      then Seq.index new_col i
      else acc2 em i j'

(* ── Matrix-level bridges over the underlying chest ──────────────────────────
   The chest indexes matrices by the nested tuple [abs (rows @| cols @| INil)]
   (i.e. [natlt rows & (natlt cols & unit)]), so the chest-level [equal],
   [approximates] and [acc] facts quantify over that nested index. These
   lemmas re-expose them in the flat [acc2 m i j] form that the matrix API
   uses, bridging the trailing [unit] that SMT cannot erase on its own. *)

val macc_mkM (#et:Type) (#rows #cols : nat)
  (f : natlt rows -> natlt cols -> GTot et)
  (i : natlt rows) (j : natlt cols)
  : Lemma (acc2 (mk2 f) i j == f i j)
          [SMTPat (acc2 (mk2 f) i j)]

val macc_mupd (#et:Type) (#rows #cols : nat)
  (m : chest2 et rows cols)
  (i : nat{i < rows}) (j : nat{j < cols}) (v : et)
  (i' : natlt rows) (j' : natlt cols)
  : Lemma (acc2 (upd2 m i j v) i' j' == (if i' = i && j' = j then v else acc2 m i' j'))
          [SMTPat (acc2 (upd2 m i j v) i' j')]

val lemma_equal_intro (#et:Type) (#rows #cols : nat)
  (m1 m2 : chest2 et rows cols)
  : Lemma (requires forall (i:natlt rows) (j:natlt cols). acc2 m1 i j == acc2 m2 i j)
          (ensures equal m1 m2)
          [SMTPat (equal m1 m2)]

val lemma_approximates_intro
  (#et:Type0) {| scalar et, real_like et |} (#rows #cols : nat)
  (m1 : chest2 et rows cols)
  (m2 : chest2 real rows cols)
  : Lemma (requires forall (i:natlt rows) (j:natlt cols). acc2 m1 i j %~ acc2 m2 i j)
          (ensures m1 %~ m2)
          [SMTPat (m1 %~ m2)]
