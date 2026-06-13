module Kuiper.EMatrix
#lang-pulse

(* An "erased" matrix, for specification purposes only *)

open Kuiper
open Kuiper.Container
open Kuiper.Chest
include Kuiper.Chest {
  to_real_chest as to_real_matrix,
  equal,
  chest_approximates as ematrix_approximates,
  chest_comb as ematrix_comb,
}
open Kuiper.Index

[@@erasable]
type ematrix (et:Type) (rows cols : nat) =
  chest (rows @| cols @| INil) et

let mkM (#et:Type) (#rows #cols : nat)
  (f : natlt rows -> natlt cols -> GTot et)
  : ematrix et rows cols
  = Chest.mk (rows @| cols @| INil)
      fun (i, (j, ())) -> f i j

let const_matrix (#et:Type) (#rows #cols : nat)
  (v:et)
  : ematrix et rows cols
  = Chest.const _ v

let macc (#et:Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  (i : natlt rows)
  (j : natlt cols)
  : GTot et
  = Chest.acc m (i, (j, ()))

let mupd (#et:Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  (i : nat{ i < rows })
  (j : nat{ j < cols })
  (v : et)
  : ematrix et rows cols
  = Chest.upd m (i, (j, ())) v

let matrix_comb (#et:Type) (#rows #cols : nat)
  (f : binop et)
  (m1 m2 : ematrix et rows cols)
  : ematrix et rows cols
  = Chest.chest_comb f m1 m2

let mtranspose (#et:Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  : ematrix et cols rows
  = mkM fun i j -> macc m j i

let ematrix_row
  (#et : Type0)
  (#rows #cols : erased nat)
  (em : ematrix et rows cols)
  (i : natlt rows)
  : GTot (lseq et cols)
  = Seq.init_ghost cols (fun j -> macc em i j)

let ematrix_col
  (#et : Type0)
  (#rows #cols : erased nat)
  (em : ematrix et rows cols)
  (j : natlt cols)
  : GTot (lseq et rows)
  = Seq.init_ghost rows (fun i -> macc em i j)

let ematrix_upd_row
  (#et : Type0)
  (#rows #cols : erased nat)
  (em : ematrix et rows cols)
  (i : natlt rows)
  (new_row : lseq et cols)
  : ematrix et rows cols
  = mkM fun i' j ->
      if i' = i
      then Seq.index new_row j
      else macc em i' j

let ematrix_upd_col
  (#et : Type0)
  (#rows #cols : erased nat)
  (em : ematrix et rows cols)
  (j : natlt cols)
  (new_col : lseq et rows)
  : ematrix et rows cols
  = mkM fun i j' ->
      if j' = j
      then Seq.index new_col i
      else macc em i j'

(* ── Matrix-level bridges over the underlying chest ──────────────────────────
   The chest indexes matrices by the nested tuple [abs (rows @| cols @| INil)]
   (i.e. [natlt rows & (natlt cols & unit)]), so the chest-level [equal],
   [approximates] and [acc] facts quantify over that nested index. These
   lemmas re-expose them in the flat [macc m i j] form that the matrix API
   uses, bridging the trailing [unit] that SMT cannot erase on its own. *)

val macc_mkM (#et:Type) (#rows #cols : nat)
  (f : natlt rows -> natlt cols -> GTot et)
  (i : natlt rows) (j : natlt cols)
  : Lemma (macc (mkM f) i j == f i j)
          [SMTPat (macc (mkM f) i j)]

val macc_mupd (#et:Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  (i : nat{i < rows}) (j : nat{j < cols}) (v : et)
  (i' : natlt rows) (j' : natlt cols)
  : Lemma (macc (mupd m i j v) i' j' == (if i' = i && j' = j then v else macc m i' j'))
          [SMTPat (macc (mupd m i j v) i' j')]

val lemma_equal_intro (#et:Type) (#rows #cols : nat)
  (m1 m2 : ematrix et rows cols)
  : Lemma (requires forall (i:natlt rows) (j:natlt cols). macc m1 i j == macc m2 i j)
          (ensures equal m1 m2)
          [SMTPat (equal m1 m2)]

val lemma_approximates_intro
  (#et:Type0) {| scalar et, real_like et |} (#rows #cols : nat)
  (m1 : ematrix et rows cols)
  (m2 : ematrix real rows cols)
  : Lemma (requires forall (i:natlt rows) (j:natlt cols). macc m1 i j %~ macc m2 i j)
          (ensures m1 %~ m2)
          [SMTPat (m1 %~ m2)]
