module Kuiper.EMatrix4
#lang-pulse

open Kuiper
module M = Kuiper.EMatrix

let macc_pat (#et :Type) (#mrows #brows #mcols #bcols : nat)
  (m : ematrix4 et mrows brows mcols bcols)
  (bi : natlt mrows)
  (i  : natlt brows)
  (bj : natlt mcols)
  (j  : natlt bcols)
  : Lemma (macc m bi i bj j == m.f (bi * brows + i, bj * bcols + j))
  = ()

let equal
  (#et #mrows #brows #mcols #bcols : _)
  (m1 m2 : ematrix4 et mrows brows mcols bcols) : prop
  = forall bi i bj j. macc m1 bi i bj j == macc m2 bi i bj j

let lemma_equal_intro
  (#et #mrows #brows #mcols #bcols : _)
  (m1 m2 : ematrix4 et mrows brows mcols bcols)
  : Lemma (requires forall bi i bj j. macc m1 bi i bj j == macc m2 bi i bj j)
          (ensures equal m1 m2)
          [SMTPat (equal m1 m2)]
  = ()

let ematrix_ext
  (#et #mrows #brows #mcols #bcols : _)
  (m1 m2 : ematrix4 et mrows brows mcols bcols)
  : Lemma (requires equal m1 m2)
          (ensures m1 == m2)
          [SMTPat (equal m1 m2)]
  = let aux (i : natlt (mrows * brows))
            (j : natlt (mcols * bcols))
      : Lemma (M.macc m1 i j == M.macc m2 i j) =
      let bi, si = divmod brows i in
      let bj, sj = divmod bcols j in
      assert (M.macc m1 (bi * brows + si) (bj * bcols + sj) == macc m1 bi si bj sj);
      assert (M.macc m2 (bi * brows + si) (bj * bcols + sj) == macc m2 bi si bj sj);
      ()
    in
    Classical.forall_intro_2 aux;
    M.ematrix_ext m1 m2
