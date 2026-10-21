module Kuiper.Sparse.DotProduct

#lang-pulse
open Kuiper
open Kuiper.Sparse.Common
open Kuiper
open Kuiper.Seq.Common { (@+) }
open Kuiper.EMatrix
open Kuiper.Spec.GEMM
module SZ = FStar.SizeT
module KSeq = Kuiper.Seq.Common


let rec _dprod_acc
  (#et:_) {| scalar et |}
  (acc : et)
  (#n : nat)
  (s t : lseq et n)
  (to : natle n)
  : et
= if to = 0
    then acc
    else add (_dprod_acc acc s t (to - 1))
             ((s @! to - 1) `mul` (t @! to - 1))

let dprod_acc
  (#et:_) {| scalar et |}
  (acc : et)
  (#n : nat)
  (s t : lseq et n)
  : et
= _dprod_acc acc s t n

let _dprod
  (#et:_) {| scalar et |}
  (#n : nat)
  (s t : lseq et n)
  (to : natle n)
  : et
= _dprod_acc zero s t to

let dprod
  (#et:_) {| scalar et |}
  (#n : nat)
  (s t : lseq et n)
  : et
= _dprod s t n

val _dprod_acc_mask_lemma
  (#et:_) {| scalar et |}
  (acc : et)
  (#n : nat)
  (k : natle n)
  (s : lseq et (n - k))
  (t : lseq et n)
  (to : natle n { k <= to })
: Lemma
  (requires true)
  (ensures
    _dprod_acc acc (Seq.create k zero @+ s) t to ==
    _dprod_acc acc s (Seq.slice t k n) (to - k)
  )

noextract
let _sparse_dprod_acc
  (#et : Type0) {| scalar et |}
  (#nnz #n : nat)
  (acc : et)
  (elems : lseq et nnz)
  (pos : lseq nat nnz{in_bounds 0 n pos})
  (t : lseq et n)
  (to : natle nnz)
  : et
= _dprod_acc acc elems (seq_make_sparse pos t) to

noextract
let sparse_dprod_acc
  (#et : Type0) {| scalar et |}
  (#n : nat )
  (acc : et)
  (#nnz : nat)
  (elems : lseq et nnz)
  (pos : lseq nat nnz{in_bounds 0 n pos})
  (t : lseq et n)
  : et
= _sparse_dprod_acc acc elems pos t nnz

let _sparse_dprod
  (#et : Type0) {| scalar et |}
  (#nnz #n : nat )
  (elems : lseq et nnz)
  (pos : lseq nat nnz{valid_pos n pos})
  (t : lseq et n)
  (to : natle nnz)
  : et
= _sparse_dprod_acc zero elems pos t to

let sparse_dprod
  (#et : Type0) {| scalar et |}
  (#nnz #n : nat )
  (elems : lseq et nnz)
  (pos : lseq nat nnz{valid_pos n pos})
  (t : lseq et n)
  : et
= _sparse_dprod elems pos t nnz

val sparse_dprod_lemma
  (#et : Type0) {| scalar et |}
  (#n #nnz : nat)
  (elems : lseq et nnz)
  (pos : lseq nat nnz)
  (t : lseq et n)
  : Lemma
    (requires valid_pos n pos)
    (ensures
      sparse_dprod elems pos t ==
      dprod (seq_unsparse _ _ elems pos) t
    )

val sparse_dprod_slice_lemma
  (#et : Type0) {| scalar et |}
  (acc : et)
  (#n #nnz : nat)
  (elems : lseq et nnz)
  (pos : lseq nat nnz)
  (t : lseq et n)
  (to : natle nnz)
  (to_ : natle to)
  : Lemma
    (requires in_bounds 0 n pos)
    (ensures
      _sparse_dprod_acc acc elems pos t to_ ==
      _sparse_dprod_acc acc
        (Seq.slice elems 0 to <: lseq et to)
        (Seq.slice pos 0 to)
        t to_
    )

val sparse_dprod_accum
  (#et : Type0) {| scalar et |}
  (acc : et)
  (#nnz : nat)
  (elems : lseq et nnz)
  (pos : lseq nat nnz)
  (#n : nat)
  (t : lseq et n)
  (from to : natle nnz { from <= to })
  : Lemma
    (requires in_bounds 0 n pos)
    (ensures
      sparse_dprod_acc
        (sparse_dprod_acc
          acc
          (Seq.slice elems 0 from <: lseq et from)
          (Seq.slice pos 0 from)
          t)
        (Seq.slice elems from to <: lseq et (to - from))
        (Seq.slice pos from to) t ==
      sparse_dprod_acc acc
        (Seq.slice elems 0 to <: lseq et to)
        (Seq.slice pos 0 to)
        t
    )

val dprod_is_matmul_single
  (#et : Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (row : natlt rows)
  (col : natlt columns)
: Lemma
  (requires true)
  (ensures
    dprod (ematrix_row m1 row) (ematrix_col m2 col) ==
    matmul_single m1 m2 row col
  )