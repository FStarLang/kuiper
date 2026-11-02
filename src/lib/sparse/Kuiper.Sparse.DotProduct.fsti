module Kuiper.Sparse.DotProduct

#lang-pulse

(* Sparse algebra is stated over real numbers. In particular, skipping
   implicit or masked zeros is not an equality on computed floats. *)
open Kuiper
open Kuiper.Sparse.Common
open Kuiper.Seq.Common { (@+) }
open Kuiper.EMatrix
open Kuiper.Spec.GEMM
module SZ = FStar.SizeT
module KSeq = Kuiper.Seq.Common


let rec _dprod_acc
  (acc : real)
  (#n : nat)
  (s t : lseq real n)
  (to : natle n)
  : real
= if to = 0
    then acc
    else add (_dprod_acc acc s t (to - 1))
             ((s @! to - 1) `mul` (t @! to - 1))

let dprod_acc
  (acc : real)
  (#n : nat)
  (s t : lseq real n)
  : real
= _dprod_acc acc s t n

let _dprod
  (#n : nat)
  (s t : lseq real n)
  (to : natle n)
  : real
= _dprod_acc zero s t to

let dprod
  (#n : nat)
  (s t : lseq real n)
  : real
= _dprod s t n

val _dprod_acc_mask_lemma
  (acc : real)
  (#n : nat)
  (k : natle n)
  (s : lseq real (n - k))
  (t : lseq real n)
  (to : natle n { k <= to })
: Lemma
  (requires true)
  (ensures
    _dprod_acc acc (Seq.create k zero @+ s) t to ==
    _dprod_acc acc s (Seq.slice t k n) (to - k)
  )

noextract
let _sparse_dprod_acc
  (#nnz #n : nat)
  (acc : real)
  (elems : lseq real nnz)
  (pos : lseq nat nnz{in_bounds 0 n pos})
  (t : lseq real n)
  (to : natle nnz)
  : real
= _dprod_acc acc elems (seq_make_sparse pos t) to

noextract
let sparse_dprod_acc
  (#n : nat )
  (acc : real)
  (#nnz : nat)
  (elems : lseq real nnz)
  (pos : lseq nat nnz{in_bounds 0 n pos})
  (t : lseq real n)
  : real
= _sparse_dprod_acc acc elems pos t nnz

let _sparse_dprod
  (#nnz #n : nat )
  (elems : lseq real nnz)
  (pos : lseq nat nnz{valid_pos n pos})
  (t : lseq real n)
  (to : natle nnz)
  : real
= _sparse_dprod_acc zero elems pos t to

let sparse_dprod
  (#nnz #n : nat )
  (elems : lseq real nnz)
  (pos : lseq nat nnz{valid_pos n pos})
  (t : lseq real n)
  : real
= _sparse_dprod elems pos t nnz

val sparse_dprod_lemma
  (#n #nnz : nat)
  (elems : lseq real nnz)
  (pos : lseq nat nnz)
  (t : lseq real n)
  : Lemma
    (requires valid_pos n pos)
    (ensures
      sparse_dprod elems pos t ==
      dprod (seq_unsparse _ _ elems pos) t
    )

val sparse_dprod_slice_lemma
  (acc : real)
  (#n #nnz : nat)
  (elems : lseq real nnz)
  (pos : lseq nat nnz)
  (t : lseq real n)
  (to : natle nnz)
  (to_ : natle to)
  : Lemma
    (requires in_bounds 0 n pos)
    (ensures
      _sparse_dprod_acc acc elems pos t to_ ==
      _sparse_dprod_acc acc
        (Seq.slice elems 0 to <: lseq real to)
        (Seq.slice pos 0 to)
        t to_
    )

val sparse_dprod_accum
  (acc : real)
  (#nnz : nat)
  (elems : lseq real nnz)
  (pos : lseq nat nnz)
  (#n : nat)
  (t : lseq real n)
  (from to : natle nnz { from <= to })
  : Lemma
    (requires in_bounds 0 n pos)
    (ensures
      sparse_dprod_acc
        (sparse_dprod_acc
          acc
          (Seq.slice elems 0 from <: lseq real from)
          (Seq.slice pos 0 from)
          t)
        (Seq.slice elems from to <: lseq real (to - from))
        (Seq.slice pos from to) t ==
      sparse_dprod_acc acc
        (Seq.slice elems 0 to <: lseq real to)
        (Seq.slice pos 0 to)
        t
    )

val dprod_is_matmul_single
  (#rows #shared #columns : nat)
  (m1 : chest2 real rows shared)
  (m2 : chest2 real shared columns)
  (row : natlt rows)
  (col : natlt columns)
: Lemma
  (requires true)
  (ensures
    dprod (ematrix_row m1 row) (ematrix_col m2 col) ==
    matmul_single m1 m2 row col
  )
