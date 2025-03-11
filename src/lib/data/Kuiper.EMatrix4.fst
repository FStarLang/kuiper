module Kuiper.EMatrix4
#lang-pulse

open Kuiper
module M = Kuiper.EMatrix

let macc_pat (#et:Type) (#mrows #brows #mcols #bcols : nat)
  (m : ematrix4 et mrows mcols brows bcols)
  (bi : natlt mrows)
  (bj : natlt mcols)
  (i  : natlt brows)
  (j  : natlt bcols)
  : Lemma (macc m bi bj i j == m.f (bi * brows + i, bj * bcols + j))
  = ()

let equal
  (#et #mrows #mcols #brows #bcols : _)
  (m1 m2 : ematrix4 et mrows mcols brows bcols) : prop
  = forall bi bj i j. macc m1 bi bj i j == macc m2 bi bj i j

let lemma_equal_intro
  (#et #mrows #mcols #brows #bcols : _)
  (m1 m2 : ematrix4 et mrows mcols brows bcols)
  : Lemma (requires forall bi bj i j. macc m1 bi bj i j == macc m2 bi bj i j)
          (ensures equal m1 m2)
          [SMTPat (equal m1 m2)]
  = ()

let ematrix_ext
  (#et #mrows #mcols #brows #bcols : _)
  (m1 m2 : ematrix4 et mrows mcols brows bcols)
  : Lemma (requires equal m1 m2)
          (ensures m1 == m2)
          [SMTPat (equal m1 m2)]
  = let aux (i : natlt (mrows * brows))
            (j : natlt (mcols * bcols))
      : Lemma (M.macc m1 i j == M.macc m2 i j) =
      let bi, si = divmod brows i in
      let bj, sj = divmod bcols j in
      assert (M.macc m1 (bi * brows + si) (bj * bcols + sj) == macc m1 bi bj si sj);
      assert (M.macc m2 (bi * brows + si) (bj * bcols + sj) == macc m2 bi bj si sj);
      ()
    in
    Classical.forall_intro_2 aux;
    M.ematrix_ext m1 m2
