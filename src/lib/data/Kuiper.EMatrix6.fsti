module Kuiper.EMatrix6
#lang-pulse

open Kuiper
module M = Kuiper.EMatrix4

(* An "erased" matrix, for specification purposes only *)

type ematrix6 (et:Type) (mrows mcols brows bcols trows tcols : nat) =
  M.ematrix4 et
    (mrows * brows)
    (mcols * bcols)
    trows
    tcols

let mkM (#et:Type) (#mrows #mcols #brows #bcols #trows #tcols : nat)
  (f : natlt mrows -> natlt mcols -> natlt brows -> natlt bcols -> natlt trows -> natlt tcols -> GTot et)
  : ematrix6 et mrows mcols brows bcols trows tcols
  = M.mkM <| fun i j k l -> f (i / brows) (j / bcols) (i % brows) (j % bcols) k l

let macc (#et:Type) (#mrows #mcols #brows #bcols #trows #tcols : nat)
  (m : ematrix6 et mrows mcols brows bcols trows tcols)
  (bi : natlt mrows)
  (bj : natlt mcols)
  (i  : natlt brows)
  (j  : natlt bcols)
  (k  : natlt trows)
  (l  : natlt tcols)
  : GTot et
  = M.macc m (bi * brows + i) (bj * bcols + j) k l

let mupd (#et:Type) (#mrows #mcols #brows #bcols #trows #tcols : nat)
  (m : ematrix6 et mrows mcols brows bcols trows tcols)
  (bi : natlt mrows)
  (bj : natlt mcols)
  (i  : natlt brows)
  (j  : natlt bcols)
  (k  : natlt trows)
  (l  : natlt tcols)
  (v : et)
  : ematrix6 et mrows mcols brows bcols trows tcols
  = mkM fun bi' i' bj' j' k' l' ->
      if bi' = bi && i' = i && bj' = bj && j' = j && k' = k && l' = l
      then v
      else macc m bi' i' bj' j' k' l'

// Needed?
val macc_pat (#et :Type) (#mrows #mcols #brows #bcols #trows #tcols : nat)
  (m : ematrix6 et mrows mcols brows bcols trows tcols)
  (bi : natlt mrows)
  (bj : natlt mcols)
  (i  : natlt brows)
  (j  : natlt bcols)
  (k  : natlt trows)
  (l  : natlt tcols)
  : Lemma (macc m bi bj i j k l == m.f ((bi * brows + i) * trows + k, (bj * bcols + j) * tcols + l))
          [SMTPat (macc m bi bj i j k l)]

let mtranspose (#et:Type) (#mrows #mcols #brows #bcols #trows #tcols : nat)
  (m : ematrix6 et mrows brows mcols bcols trows tcols)
  : ematrix6 et mcols bcols mrows brows trows tcols
  = mkM fun bi i bj j k l -> macc m bj j bi i k l

val equal
  (#et #mrows #mcols #brows #bcols #trows #tcols : _)
  (m1 m2 : ematrix6 et mrows mcols brows bcols trows tcols) : prop

val lemma_equal_intro
  (#et #mrows #mcols #brows #bcols #trows #tcols : _)
  (m1 m2 : ematrix6 et mrows mcols brows bcols trows tcols)
  : Lemma (requires forall bi bj i j. macc m1 bi bj i j == macc m2 bi bj i j)
          (ensures equal m1 m2)
          [SMTPat (equal m1 m2)]

val ematrix_ext
  (#et #mrows #mcols #brows #bcols #trows #tcols : _)
  (m1 m2 : ematrix6 et mrows mcols brows bcols trows tcols)
  : Lemma (requires equal m1 m2)
          (ensures m1 == m2)
          [SMTPat (equal m1 m2)]
