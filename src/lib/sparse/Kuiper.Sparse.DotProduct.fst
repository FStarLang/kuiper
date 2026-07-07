module Kuiper.Sparse.DotProduct

#lang-pulse
open Kuiper
open Kuiper.Sparse.Common
open Kuiper
module SZ = FStar.SizeT
module KSeq = Kuiper.Seq.Common

//TODO ver lib/algo/Kuiper.DotProd


let scale
  (#et:_) {| scalar et |}
  (#n : nat)
  (k : et)
  (s : lseq et n)
  : (lseq et n)
=
  Seq.init n (fun i -> k `mul`(s @! i))

let _comb
  (#et:_) {| scalar et |}
  (#n : nat)
  (#acc_l : nat)
  (acc : lseq et acc_l)
  (k : et)
  (s : lseq et n)
  (to : natle n)
  : Ghost (lseq et acc_l) (requires n <= acc_l) (ensures fun _ -> true)
=
  Seq.init_ghost acc_l (fun i ->
    if i < to
      then (acc @! i) `add` (k `mul`(s @! i))
      else acc @! i
  )

let _comb_lemma
  (#et:_) {| scalar et |}
  (#n : nat)
  (#acc_l : nat)
  (acc : lseq et acc_l)
  (k : et)
  (s : lseq et n)
  (to : natlt n)
: Lemma
  (requires n <= acc_l)
  (ensures
    Seq.upd (_comb acc k s to) to ((acc @! to) `add` (k `mul` (s @! to))) ==
    _comb acc k s (to + 1)
  )
=
  assert Seq.equal
    (Seq.upd (_comb acc k s to) to ((acc @! to) `add` (k `mul` (s @! to))))
    (_comb acc k s (to + 1))

let comb
  (#et:_) {| scalar et |}
  (#n : nat)
  (#acc_l : nat)
  (acc : lseq et acc_l)
  (k : et)
  (s : lseq et n)
  : Ghost (lseq et acc_l) (requires n <= acc_l) (ensures fun _ -> true)
=
  _comb acc k s n

let rec _dprod_acc
  (#et:_) {| scalar et |}
  (acc : et)
  (#n : nat)
  (s t : lseq et n)
  (to : natle n)
  : et
=
  if to = 0
    then acc
    else
      add
        (_dprod_acc acc s t (to - 1))
        ((s @! to - 1) `mul` (t @! to - 1))

let dprod_acc
  (#et:_) {| scalar et |}
  (acc : et)
  (#n : nat)
  (s t : lseq et n)
  : et
= _dprod_acc acc s t n

let rec _dprod_acc_lemma0
  (#et:_) {| scalar et |}
  (#n1 #n2 : nat)
  (acc : et)
  (s1 t1 : lseq et n1)
  (s2 t2 : lseq et n2)
  (to : natle n1)
: Lemma
  (requires true)
  (ensures
    _dprod_acc acc s1 t1 to ==
    _dprod_acc acc #(n1 + n2) (Seq.append s1 s2) (Seq.append t1 t2) to
  )
=
  if to = 0
    then ()
    else _dprod_acc_lemma0 acc s1 t1 s2 t2 (to - 1)

let rec _dprod_acc_lemma
  (#et:_) {| scalar et |}
  (#n1 #n2 : nat)
  (acc : et)
  (s1 t1 : lseq et n1)
  (s2 t2 : lseq et n2)
  (to : natle n2)
: Lemma
  (requires true)
  (ensures
    _dprod_acc (dprod_acc acc s1 t1) s2 t2 to ==
    _dprod_acc acc #(n1 + n2) (Seq.append s1 s2) (Seq.append t1 t2) (n1 + to)
  )
=
  if to = 0
    then _dprod_acc_lemma0 acc s1 t1 s2 t2 n1
    else _dprod_acc_lemma acc s1 t1 s2 t2 (to - 1)

let dprod_acc_lemma
  (#et:_) {| scalar et |}
  (acc : et)
  (#n1 : nat)
  (s1 t1 : lseq et n1)
  (#n2 : nat)
  (s2 t2 : lseq et n2)
: Lemma
  (requires true)
  (ensures
    dprod_acc (dprod_acc acc s1 t1) s2 t2 ==
    dprod_acc acc #(n1 + n2) (Seq.append s1 s2) (Seq.append t1 t2)
  )
=
  _dprod_acc_lemma acc s1 t1 s2 t2 n2

let dprod_acc_lemma'
  (#et:_) {| scalar et |}
  (#n : nat)
  (acc : et)
  (s t : lseq et n)
  (to : natle n)
: Lemma
  (requires true)
  (ensures
    dprod_acc
      (dprod_acc acc #to (fst (Seq.split s to)) (fst (Seq.split t to)))
      #(n - to)
      (snd (Seq.split s to)) (snd (Seq.split t to)) ==
    dprod_acc acc s t
  )
=
  let s1, s2 = Seq.split s to in
  let t1, t2 = Seq.split t to in

  Seq.lemma_split s to;
  Seq.lemma_split t to;

  dprod_acc_lemma acc #to s1 t1 #(n - to) s2 t2

let _dprod
  (#et:_) {| scalar et |}
  (#n : nat)
  (s t : lseq et n)
  (to : natle n)
  : et
=
  _dprod_acc zero s t to


let dprod
  (#et:_) {| scalar et |}
  (#n : nat)
  (s t : lseq et n)
  : et
= _dprod s t n

noextract
let _sparse_dprod_acc
  (#et : Type0) {| scalar et |}
  (#nnz #n : nat) //{nnz <= n})
  (acc : et)
  (elems : lseq et nnz)
  (pos : lseq nat nnz{valid_pos n pos})
  (t : lseq et n)
  (to : natle nnz)
  : et
=
  _dprod_acc acc elems (seq_make_sparse pos t) to

noextract
let sparse_dprod_acc
  (#et : Type0) {| scalar et |}
  (#n : nat )//{nnz <= n})
  (acc : et)
  (#nnz : nat)
  (elems : lseq et nnz)
  (pos : lseq nat nnz{valid_pos n pos})
  (t : lseq et n)
  : et
= _sparse_dprod_acc acc elems pos t nnz

let _sparse_dprod
  (#et : Type0) {| scalar et |}
  (#nnz #n : nat )//{nnz <= n})
  (elems : lseq et nnz)
  (pos : lseq nat nnz{valid_pos n pos})
  (t : lseq et n)
  (to : natle nnz)
  : et
= _sparse_dprod_acc zero elems pos t to

let sparse_dprod
  (#et : Type0) {| scalar et |}
  (#nnz #n : nat )//{nnz <= n})
  (elems : lseq et nnz)
  (pos : lseq nat nnz{valid_pos n pos})
  (t : lseq et n)
  : et
= _sparse_dprod elems pos t nnz

let rec dprod_acc_all_zeros
  (#et:Type) {| scalar et |}
  (acc : et)
  (#n : nat)
  (s t : lseq et n)
  (from to : nat{from <= to /\ to <= n})
  : Lemma
    (requires forall i. from <= i /\ i < to ==> s @! i == zero)
    (ensures _dprod_acc acc s t from == _dprod_acc acc s t to)
=
  if from = to
    then ()
    else dprod_acc_all_zeros acc s t from (to - 1)

#push-options "--split_queries always"
let rec _sparse_dprod_acc_lemma
  (#et : Type0) {| scalar et |}
  (acc : et)
  (#n : nat) (#nnz : nat)
  (elems : lseq et nnz)
  (pos : lseq nat nnz)
  (t : lseq et n)
  (to : natlt nnz)
  : Lemma
    (requires valid_pos n pos)
    (ensures
      _sparse_dprod_acc acc elems pos t (to + 1) ==
      _dprod_acc acc (unsparse _ _ elems pos) t ((pos @! to) + 1)
    )
=
  bounded_from_sorted_in_bounds 0 n pos;
  let s = unsparse _ _ elems pos in

  if to = 0
  then dprod_acc_all_zeros acc s t 0 (pos @! to)
  else (
    _sparse_dprod_acc_lemma acc elems pos t (to - 1);
    dprod_acc_all_zeros acc s t ((pos @! to - 1) + 1) (pos @! to)
  )
#pop-options

let sparse_dprod_acc_lemma
  (#et : Type0) {| scalar et |}
  (acc : et)
  (#n : nat) (#nnz : nat)
  (elems : lseq et nnz)
  (pos : lseq nat nnz)
  (t : lseq et n)
  : Lemma
    (requires valid_pos n pos)
    (ensures
      sparse_dprod_acc acc elems pos t ==
      dprod_acc acc (unsparse _ _ elems pos) t
    )
=
  let s = unsparse _ _ elems pos in
  if nnz = 0
    then dprod_acc_all_zeros acc s t 0 n
    else (
      _sparse_dprod_acc_lemma acc elems pos t (nnz - 1);
      dprod_acc_all_zeros acc s t ((pos @! nnz - 1) + 1) n
    )

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
  sparse_dprod_acc_lemma zero elems pos t

open Kuiper.EMatrix
open Kuiper.Spec.GEMM

let rec __dprod_is_matmul_single
  (#et : Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (row : natlt rows)
  (col : natlt columns)
  (to : natle shared)
: Lemma
  (requires true)
  (ensures
    _dprod (ematrix_row m1 row) (ematrix_col m2 col) to ==
    __matmul_single m1 m2 row col to
  )
=
  if to = 0
    then ()
    else (
      matmul_single_lemma m1 m2 row col to;
      __dprod_is_matmul_single m1 m2 row col (to - 1)
    )

let dprod_is_matmul_single
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
=
  __dprod_is_matmul_single m1 m2 row col shared

(*
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

*)