module Kuiper.Example.Sparse.DotProduct

#lang-pulse
open Kuiper
open Kuiper.Sparse
module SZ = FStar.SizeT
module KSeq = Kuiper.Seq.Common

let rec _dprod
  (#et:_) {| scalar et |}
  (#n : nat)
  (s t : lseq et n)
  (to : natle n)
  : GTot et
=
  if to = 0
    then zero
    else
      add
        (_dprod s t (to - 1))
        ((s @! to - 1) `mul` (t @! to - 1))


let dprod
  (#et:_) {| scalar et |}
  (#n : nat)
  (s t : lseq et n)
  : GTot et
= _dprod s t n


noextract
let rec _sparse_dprod
  (#et : Type0) {| scalar et |}
  (#nnz #n : nat) //{nnz <= n})
  (elems : lseq et nnz)
  (pos : lseq nat nnz{valid_pos n pos})
  (t : lseq et n)
  (to : natle nnz)
  : GTot et
=
  if to = 0
    then zero
    else
      add
        (_sparse_dprod elems pos t (to - 1))
        (mul (elems @! to - 1) (t @! (pos @! to - 1)))

noextract
let sparse_dprod
  (#et : Type0) {| scalar et |}
  (#nnz #n : nat )//{nnz <= n})
  (elems : lseq et nnz)
  (pos : lseq nat nnz{valid_pos n pos})
  (t : lseq et n)
  : GTot et
= _sparse_dprod elems pos t nnz

let rec dprod_all_zeros
  (#et:Type) {| scalar et |}
  (#n : nat)
  (s t : lseq et n)
  (from to : nat{from <= to /\ to <= n})
  : Lemma
    (requires forall i. from <= i /\ i < to ==> s @! i == zero)
    (ensures _dprod s t from == _dprod s t to)
=
  if from = to
    then ()
    else dprod_all_zeros s t from (to - 1)

#push-options "--split_queries always"
let rec _sparse_dprod_lemma
  (#et : Type0) {| scalar et |}
  (#n : nat) (#nnz : nat)
  (elems : lseq et nnz)
  (pos : lseq nat nnz)
  (t : lseq et n)
  (to : natlt nnz)
  : Lemma
    (requires valid_pos n pos)
    (ensures
      _sparse_dprod elems pos t (to + 1) ==
      _dprod (unsparse _ _ elems pos) t ((pos @! to) + 1)
    )
=
  bounded_from_sorted_in_bounds 0 n pos;
  let s = unsparse _ _ elems pos in

  if to = 0
  then dprod_all_zeros s t 0 (pos @! to)
  else (
    _sparse_dprod_lemma elems pos t (to - 1);
    dprod_all_zeros s t ((pos @! to - 1) + 1) (pos @! to)
  )
#pop-options

let sparse_dprod_lemma
  (#et : Type0) {| scalar et |}
  (#n : nat) (#nnz : nat)
  (elems : lseq et nnz)
  (pos : lseq nat nnz)
  (t : lseq et n)
  : Lemma
    (requires valid_pos n pos)
    (ensures
      sparse_dprod elems pos t ==
      dprod (unsparse _ _ elems pos) t
    )
=
  let s = unsparse _ _ elems pos in
  if nnz = 0
    then dprod_all_zeros s t 0 n
    else (
      _sparse_dprod_lemma elems pos t (nnz - 1);
      dprod_all_zeros s t ((pos @! nnz - 1) + 1) n
    )

inline_for_extraction noextract
fn sarray_product_dense
  (#et : eqtype) {| scalar et |}
  (#l : erased nat)
  (a : sarray et l)
  (v : gpu_array et l)
  (#s #t : erased (lseq et l))
  preserves gpu ** a |-> s ** v |-> t
  returns
    dp: et
  ensures
    pure (dp == dprod s t)
{
  unfold a |-> s; // que pasa con esto?
  unfold v |-> t;
  unfold sarray_pts_to a s;
  unfold sarray_pts_to' a s;

  with v_elems. assert a.elems |-> v_elems;
  with v_pos. assert a.pos |-> v_pos;

  let mut i = 0sz;
  let mut dp : et = zero;

  let pos : erased (lseq nat a.nnz) = cast_pos v_pos;

  // esto prueba que nnz <= l
  // TODO abstraer en otro lema
  bounded_from_sorted_in_bounds 0 l (cast_pos v_pos);

  while (!i <^ a.nnz)
    invariant
      live dp ** live i **
        pure (
          !i <= a.nnz /\
          !dp = _sparse_dprod v_elems (cast_pos v_pos) t !i
        )
    decreases (a.nnz - !i)
  {
    let p = gpu_array_read a.pos !i;
    let x = gpu_array_read a.elems !i;
    let y = gpu_array_read v p;

    dp := !dp `add` (x `mul` y);
    i := !i `SZ.add` 1sz;
  };

  sparse_dprod_lemma v_elems pos t;

  fold sarray_pts_to' a s;
  fold sarray_pts_to a s;
  fold v |-> t;

  !dp;
}

// podemos usar esto para probar que dprod es un left-fold
let rec seq_fold_left_lemma
  (#a #b : Type)
  (f: b -> a -> b)
  (#n : nat)
  (acc: b) (v: lseq a n)
  : Lemma
    (requires n > 0)
    (ensures
      KSeq.seq_fold_left f acc v ==
      f (KSeq.seq_fold_left f acc (KSeq.seq_take (n - 1) v)) (v @! n - 1)
    )
=
  let open KSeq in
  if n = 1
    then ()
    else seq_fold_left_lemma f #(n - 1) (f acc (v @! 0)) (seq_drop 1 v)

let sarray_product_dense_u32 #len = sarray_product_dense #u32 #_ #len
