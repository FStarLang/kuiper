module Kuiper.EMatrix4
#lang-pulse

open Kuiper
module M = Kuiper.EMatrix

(* An "erased" matrix, for specification purposes only *)

type ematrix4 (et:Type) (mrows mcols brows bcols : nat) =
  M.ematrix et
    (mrows * brows)
    (mcols * bcols)

let mkM (#et:Type) (#mrows #brows #mcols #bcols : nat)
  (f : natlt mrows -> natlt mcols -> natlt brows -> natlt bcols -> GTot et)
  : ematrix4 et mrows mcols brows bcols
  = M.mkM <| fun i j -> f (i / brows) (j / bcols) (i % brows) (j % bcols)

let macc (#et:Type) (#mrows #brows #mcols #bcols : nat)
  (m : ematrix4 et mrows mcols brows bcols)
  (bi : natlt mrows)
  (bj : natlt mcols)
  (i  : natlt brows)
  (j  : natlt bcols)
  : GTot et
  = M.macc m (bi * brows + i) (bj * bcols + j)

let mupd (#et:Type) (#mrows #brows #mcols #bcols : nat)
  (m : ematrix4 et mrows mcols brows bcols)
  (bi : natlt mrows)
  (bj : natlt mcols)
  (i  : natlt brows)
  (j  : natlt bcols)
  (v : et)
  : ematrix4 et mrows mcols brows bcols
  = mkM fun bi' i' bj' j' ->
      if bi' = bi && i' = i && bj' = bj && j' = j
      then v
      else macc m bi' i' bj' j'

// Needed?
// val macc_pat (#et :Type) (#mrows #brows #mcols #bcols : nat)
//   (m : ematrix4 et mrows brows mcols bcols)
//   (bi : natlt mrows)
//   (i  : natlt brows)
//   (bj : natlt mcols)
//   (j  : natlt bcols)
//   : Lemma (macc m bi i bj j == m.f (bi * brows + i, bj * bcols + j))
//           [SMTPat (m.f (bi * brows + i, bj * bcols + j))]

let mtranspose (#et:Type) (#mrows #mcols #brows #bcols : nat)
  (m : ematrix4 et mrows brows mcols bcols)
  : ematrix4 et mcols bcols mrows brows
  = mkM fun bi i bj j -> macc m bj j bi i

val equal
  (#et #mrows #mcols #brows #bcols : _)
  (m1 m2 : ematrix4 et mrows mcols brows bcols) : prop

val lemma_equal_intro
  (#et #mrows #mcols #brows #bcols : _)
  (m1 m2 : ematrix4 et mrows mcols brows bcols)
  : Lemma (requires forall bi bj i j. macc m1 bi bj i j == macc m2 bi bj i j)
          (ensures equal m1 m2)
          [SMTPat (equal m1 m2)]

val ematrix_ext
  (#et #mrows #mcols #brows #bcols : _)
  (m1 m2 : ematrix4 et mrows mcols brows bcols)
  : Lemma (requires equal m1 m2)
          (ensures m1 == m2)
          [SMTPat (equal m1 m2)]
