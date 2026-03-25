module Kuiper.Sparse.DotProduct

#lang-pulse
open Kuiper
open Kuiper.Sparse
module SZ = FStar.SizeT
module KSeq = Kuiper.Seq.Common
module DP = Kuiper.Poly.DotProduct

let pmul
  (#et:_) {| scalar et |}
  (#l : nat)
  (s t : lseq et l)
  : GTot (lseq et l)
=
  DP.pmul s t


let sum
  (#et:_) {| scalar et |}
  (#l : nat)
  (s : lseq et l)
  : GTot et
=
  DP.sum s

let dprod
  (#et:_) {| scalar et |}
  (#l : nat)
  (s t : lseq et l)
  : GTot et
=
  sum (pmul s t)

(* producto interno sparse x dense *)

let seq_project
  (#a:_)
  (#nnz #l : nat)
  (pos : lseq nat nnz{valid_pos l pos})
  (s : lseq a l)
  : GTot (lseq a nnz)
=
  let open FStar.Seq in
  // me gustaría escribir:
  // map_seq (index s) pos
  init_ghost nnz fun i ->
    index s (index pos i)


noextract
let rec sum_all_zeros
  (#et : _) {| scalar et |}
  (l : nat)
  (k : et)
  : Lemma
    (requires true)
    (ensures KSeq.seq_fold_left add k (Seq.create #et l zero) == k)
=
  let open FStar.Seq in
  if l = 0
    then ()
    else (
      sum_all_zeros #et (l - 1) k;
      assert create #et (l -1) zero `equal` tail (create #et l zero)
    )


noextract
let shift
  (#l a b : nat)
  (s : lseq nat l)
  : Ghost (lseq nat l)
    (requires a > 0 /\ b > 0 /\ in_bounds a b s)
    (ensures fun s' -> in_bounds (a - 1) (b -1) s')
=
  Seq.init_ghost l fun i -> let (k : nat) = (s @! i) - 1 in k

noextract
let shift_tail
  (#nnz l : nat{nnz > 0 /\ l > 0})
  (pos : lseq nat nnz{valid_pos l pos})
  : Ghost (lseq nat (nnz -1))
    (requires true)
    (ensures valid_pos (l - 1))
=
  assert (pos @! 0 >= 0);
  shift 1 l (Seq.tail pos)


noextract
let rec shift_tail_mem
  (#nnz l : nat{nnz > 0 /\ l > 0})
  (pos : lseq nat nnz{valid_pos l pos})
  (i : nat)
  : Lemma
    (requires true)
    (ensures Seq.mem (i + 1) (Seq.tail pos) <==> Seq.mem i (shift_tail l pos))
=
  let open FStar.Seq in

  let pos' = tail pos in
  if len pos' = 0
    then ()
    else (
      assert shift_tail #(nnz - 1) l (tail pos) `equal` tail (shift_tail l pos);
      shift_tail_mem #(nnz - 1) l (tail pos) i
    )


noextract
let shift_tail_unsparse
  (#et:_) {| scalar et |}
  (#nnz l : nat{nnz > 0 /\ l > 0})
  (pos : lseq nat nnz{valid_pos l pos})
  (elems : lseq et nnz)
  : Lemma
    (requires pos @! 0 == 0)
    (ensures
      Seq.tail (unsparse nnz l elems pos) ==
      unsparse (nnz - 1) (l - 1) (Seq.tail elems) (shift_tail l pos)
    )
=
  let open FStar.Seq in

  let pos' = shift_tail l pos in
  let s1 = tail (unsparse nnz l elems pos) in
  let s2 = unsparse (nnz - 1) (l - 1) (tail elems) pos' in

  introduce forall i. s1 @! i == s2 @! i
  with (
    shift_tail_mem l pos i;
    if mem i pos'
      then assert index_mem i pos' == index_mem (i + 1) pos - 1
      else ()
  );
  assert s1 `equal` s2

noextract
let rec shift_mem
  (#nnz l : nat{nnz > 0 /\ l > 0})
  (pos : lseq nat nnz{valid_pos l pos})
  (i : nat)
  : Lemma
    (requires in_bounds 1 l pos)
    (ensures Seq.mem (i + 1) pos <==> Seq.mem i (shift 1 l pos))
=
  let open FStar.Seq in

  if nnz = 1
    then ()
    else (
      assert tail (shift 1 l pos) `equal` shift #(nnz - 1) 1 l (tail pos);
      shift_mem #(nnz - 1) l (tail pos) i
    )

noextract
let shift_unsparse
  (#et:_) {| scalar et |}
  (#nnz l : nat{nnz > 0 /\ nnz <= l})
  (pos : lseq nat nnz{valid_pos l pos})
  (elems : lseq et nnz)
  : Lemma
    (requires in_bounds 1 l pos)
    (ensures
      Seq.tail (unsparse nnz l elems pos) ==
      unsparse nnz (l - 1) elems (shift 1 l pos))
=
  let open FStar.Seq in

  let s1 = tail (unsparse nnz l elems pos) in
  let s2 = unsparse nnz (l - 1) elems (shift 1 l pos) in
  introduce forall i. s1 @! i == s2 @! i
  with  shift_mem l pos i;
  assert s1 `equal` s2

noextract
let rec lemma_sparse_dprod
  (#et : eqtype) {| scalar et |}
  (#nnz #l : nat)
  (elems : lseq et nnz)
  (pos   : lseq nat nnz{valid_pos l pos})
  (s : lseq et l)
  (k : et)
  : Lemma
    (requires true)
    (ensures
      KSeq.seq_fold_left add k (elems `pmul` seq_project pos s) ==
      KSeq.seq_fold_left add k (unsparse nnz l elems pos `pmul` s))
=
  let open FStar.Seq in

  bounded_from_sorted_in_bounds 0 l pos;

  let p1 = elems `pmul` seq_project pos s in
  let p2 = unsparse _ _ elems pos `pmul` s in

  if l = 0
    then ()
    else (
      if nnz = 0
        then (
          assert p2 `equal` create l zero;
          sum_all_zeros #et l k
        )
        else (
          if mem 0 pos
            then (
              let (k' : et) = k `add` (p1 @! 0) in
              assert k' == k `add` (p2 @! 0);
              let pos' = shift_tail l pos in
              shift_tail_unsparse l pos elems;
              assert tail p1 `equal` (tail elems `pmul` seq_project  #_ #(nnz - 1) #(l - 1) pos' (tail s));
              assert tail p2 `equal` (unsparse (nnz - 1) (l -1) (tail elems) pos' `pmul` tail s);
              lemma_sparse_dprod #_ #_ #(nnz - 1) #(l - 1) (tail elems) pos' (tail s) k'
            )
            else (
              let pos' = shift 1 l pos in
              shift_unsparse l pos elems;
              assert p1 `equal` (elems `pmul` seq_project #_ #nnz #(l - 1) pos' (tail s));
              assert tail p2 `equal` (unsparse nnz (l - 1) elems pos' `pmul` tail s);
              lemma_sparse_dprod #_ #_ #nnz #(l - 1) elems pos' (tail s) k
            )
        )
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

  while (!i <^ a.nnz)
    invariant
      (exists* i_v dp_v.
        i |-> i_v **
        dp |-> dp_v **
        pure (
          i_v <= a.nnz /\
          KSeq.seq_fold_left add !dp (seq_drop i_v (v_elems `pmul` seq_project pos t)) ==
          dprod v_elems (seq_project pos t)
        )
      )
    decreases (a.nnz - !i)
  {
    let p = gpu_array_read a.pos !i;
    let x = gpu_array_read a.elems !i;
    let y = gpu_array_read v p;

    dp := !dp `add` (x `mul` y);
    i := !i `SZ.add` 1sz;
  };

  lemma_sparse_dprod v_elems pos t zero;

  fold sarray_pts_to' a s;
  fold sarray_pts_to a s;
  fold v |-> t;

  !dp;
}

let product_dense_u32 #l = sarray_product_dense #u32 #_ #l


(* sarray_product: producto intero sparse x sparse *)

inline_for_extraction noextract
fn sarray_product_quadratic
  (#et : eqtype) {| scalar et |}
  (#l : erased nat)
  (a b : sarray et l)
  (#s #t : erased (lseq et l))
  preserves gpu ** a |-> s ** b |-> t
  requires
    emp
  returns
    dp: et
  ensures
    emp //pure (dp == dprod s t)
{
  unfold sarray_pts_to a s;
  unfold sarray_pts_to' a s;
  unfold sarray_pts_to b t;
  unfold sarray_pts_to' b t;

  let mut dp : et = zero;

  let mut i = 0sz;
  while (!i <^ a.nnz)
    invariant live i ** live dp
    decreases (a.nnz - !i)
  {
    let mut j = 0sz;
    let p_a = gpu_array_read a.pos !i;
    while (!j <^ b.nnz)
      invariant live j ** live dp
      decreases (b.nnz - !j)
    {
      let p_b = gpu_array_read b.pos !j;
      if (p_a = p_b) {
        let x = gpu_array_read a.elems !i;
        let y = gpu_array_read b.elems !j;
        dp := !dp `add` (x `mul` y);
        j := !j `SZ.add` 1sz;
      } else {
        j := !j `SZ.add` 1sz;
      };
    };
    i := !i `SZ.add` 1sz;
  };


  fold sarray_pts_to' a s;
  fold sarray_pts_to a s;
  fold sarray_pts_to' b t;
  fold sarray_pts_to b t;

  !dp;
}

let product_sparse_quadratic_u32 #l = sarray_product_quadratic #u32 #_ #l

inline_for_extraction noextract
fn sarray_product
  (#et : eqtype) {| scalar et |}
  (#l : erased nat)
  (a b : sarray et l)
  (#s #t : erased (lseq et l))
  preserves gpu ** a |-> s ** b |-> t
  requires
    emp
  returns
    dp: et
  // ensures
  //   pure (dp == dprod s t)
{
  unfold sarray_pts_to a s;
  unfold sarray_pts_to' a s;
  unfold sarray_pts_to b t;
  unfold sarray_pts_to' b t;

  let mut dp : et = zero;

  let mut i = 0sz;
  let mut j = 0sz;
  while (!i <^ a.nnz && !j <^ b.nnz)
    invariant live i ** live j
    invariant live dp
    decreases (a.nnz - !i + (b.nnz - !j))
  {
    // estas lecturas podrian hacerse una sola vez
    let p_a = gpu_array_read a.pos !i;
    let p_b = gpu_array_read b.pos !j;
    if ((p_a <^ p_b)) {
      i := !i `SZ.add` 1sz
    } else if ((p_b <^ p_a)) {
      j := !j `SZ.add` 1sz;
    } else {
      let x = gpu_array_read a.elems !i;
      let y = gpu_array_read b.elems !j;
      dp := !dp `add` (x `mul` y);
      i := !i `SZ.add` 1sz;
      j := !j `SZ.add` 1sz;
    };
  };


  fold sarray_pts_to' a s;
  fold sarray_pts_to a s;
  fold sarray_pts_to' b t;
  fold sarray_pts_to b t;

  !dp;
}

let product_sparse_u32 #l = sarray_product #u32 #_ #l
