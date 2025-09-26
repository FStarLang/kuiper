module Kuiper.Sparse.Example

#lang-pulse
open Kuiper
open Kuiper.Sparse
module SZ = FStar.SizeT


(* sarray_id: lee y escribe el arreglo sin modificarlo *)

inline_for_extraction noextract
fn sarray_id
  (#et : Type0) {| scalar et |}
  (l : erased nat)
  (a : sarray et l)
  (#s0 : erased (lseq et l))
  preserves gpu
  preserves sarray_pts_to a s0
{
  let mut i = 0sz;
  while ((!i <^ a.nnz))
    invariant a |-> s0 ** live i
  {
    unfold sarray_pts_to a s0;
    with v_elems. assert a.elems |-> v_elems;

    let v = gpu_array_read a.elems !i;
    gpu_array_write a.elems !i v;

    with i_v.
      assert i |-> i_v;
      assert pure (Seq.equal v_elems (Seq.upd v_elems i_v v));

    i := !i `SZ.add` 1sz;
    
    fold sarray_pts_to a s0;
  }
}

let _id_u32 = sarray_id #u32 #_

(* scale_sarray: producto escalar *)

let scale_seq
  (#et:_) {| d : scalar et |}
  (k : et)
  (#l : nat)
  (s : lseq et l)
  : seq et
=
  Seq.map_seq (mul k) s

let scale_unsparse
  (#et:_) {| scalar et |}
  (k : et)
  (#nnz #l : nat)
  (elems : lseq et nnz)
  (pos   : lseq nat nnz{valid_pos l pos})
  : Lemma
    (requires true)
    (ensures scale_seq k (unsparse nnz l elems pos) == unsparse nnz l (scale_seq k elems) pos)
=
  let open FStar.Seq in
  let s = scale_seq k (unsparse nnz l elems pos) in
  let s' = unsparse nnz l (scale_seq k elems) pos in
  assert (s `equal` s')

inline_for_extraction noextract
fn sarray_scale
  (#et : eqtype) {| scalar et |}
  (k : et)
  (#l : erased nat)
  (a : sarray et l)
  (#s : erased (lseq et l))
  preserves gpu
  requires a |-> s
  ensures  a |-> scale_seq k s
{
  unfold sarray_pts_to a s;

  let mut i = 0sz;

  with v_elems. assert a.elems |-> v_elems;
  with v_pos. assert a.pos |-> v_pos;
  
  while ((!i <^ a.nnz))
    invariant
      (exists* i_v v_elems'. 
        i |-> i_v **
        a.elems |-> v_elems' **
        pure FStar.Seq.(
          len v_elems' == a.nnz /\
          forall (j : nat{j < a.nnz}).
            (j <  i_v ==> index v_elems' j == k `mul` index v_elems j) /\
            (j >= i_v ==> index v_elems' j == index v_elems j)))
  {
    let v = gpu_array_read a.elems !i;
    gpu_array_write a.elems !i (k `mul` v);
    i := !i `SZ.add` 1sz;
  };

  with v_elems'. assert a.elems |-> v_elems';

  assert pure FStar.Seq.(v_elems' `equal` scale_seq k v_elems);

  scale_unsparse k #a.nnz #l v_elems (cast_pos v_pos);

  fold sarray_pts_to a (scale_seq k s);
}

let _scale_u32 = sarray_scale #u32 #_


