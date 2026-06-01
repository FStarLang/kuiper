module Kuiper.Sparse.Array.Iterator

#lang-pulse
open Kuiper
open Kuiper.Sparse.Common
open Kuiper.Sparse.Array

// This is here to force extraction.
let _ = 1ul

(* iterador sobre array esparso *)

inline_for_extraction
type sarray_iterator
  (#et : Type0) (#l : erased nat)
  (a : sarray et l) =
{
  i   : (i   : sz{i <= a.nnz}); // índice en elems
}

inline_for_extraction noextract
fn sarray_iterator_init
  (#et : Type0) {| scalar et |}
  (#l : erased nat)
  (a : sarray et l)
  (#f : perm)
  (#s : erased (seq et))
  (#v_elems : erased (seq et){Seq.length v_elems == a.nnz})
  (#v_pos   : erased (seq sz){Seq.length v_pos   == a.nnz})
  preserves gpu
  preserves sarray_pts_to' a #f s v_elems v_pos
  returns i : sarray_iterator #et #l a
  ensures pure (
    forall (j : natlt (Seq.length s)).
      i.i < a.nnz /\ j < v_pos @! i.i ==> s @! j == zero
  )
{
    let i : sarray_iterator a = { i = 0sz };
    unfold sarray_pts_to' a #f s v_elems v_pos;
    fold sarray_pts_to' a #f s v_elems v_pos;
    i;
}

inline_for_extraction noextract
let sarray_iterator_end
  (#et : Type0) (#l : erased nat)
  (#a : sarray et l)
  (i : sarray_iterator a)
  : bool =
  i.i = a.nnz

inline_for_extraction noextract
fn sarray_iterator_get
  (#et : Type0) {| scalar et |}
  (#l : erased nat)
  (#a : sarray et l)
  (#s : erased (seq et))
  (i : sarray_iterator a)
  preserves gpu ** a |-> s
  requires
    pure (not (sarray_iterator_end i))
  returns v : sz & et
{
  unfold sarray_pts_to a s;
  with v_elems v_pos.
    assert sarray_pts_to' a s v_elems v_pos;
    unfold sarray_pts_to' a s v_elems v_pos;

  let v = slice_read a.elems i.i;
  let p = slice_read a.pos i.i;

  fold sarray_pts_to' a s v_elems v_pos;
  fold sarray_pts_to a s;
  (p, v)
}

inline_for_extraction noextract
fn sarray_iterator_next
  (#et : Type0) {| scalar et |}
  (#l : erased nat)
  (#a : sarray et l)
  (i : sarray_iterator a)
  (#f : perm)
  (#s : erased (seq et))
  (#v_elems : erased (seq et){len v_elems == a.nnz})
  (#v_pos   : erased (seq sz){len v_pos   == a.nnz})
  (#_ : squash (not (sarray_iterator_end i)))
  preserves gpu
  preserves sarray_pts_to' a #f s v_elems v_pos
  requires
    pure (not (sarray_iterator_end i))
  returns i' : sarray_iterator a
  ensures pure (
    forall (j : natlt (len s)).
    v_pos @! i.i < j /\
      (if i'.i = a.nnz then true else j < v_pos @! i'.i)
      ==> s @! j == zero
  )
{
  unfold sarray_pts_to' a #f s v_elems v_pos;
    let i' : sarray_iterator a = {i = i.i +^ 1sz};
    fold sarray_pts_to' a #f s v_elems v_pos;
    i'
}

inline_for_extraction noextract
fn sarray_iterator_test
  (#et : eqtype) {| ets: scalar et |}
  (#l : erased nat)
  (a : sarray et l)
  (#s : erased (seq et))
  preserves gpu ** a |-> s
  ensures emp
{
  unfold sarray_pts_to a;

  with v_elems v_pos.
    assert sarray_pts_to' a s v_elems v_pos;

  let mut it : sarray_iterator #et #l a = sarray_iterator_init a #_ #s;

  fold sarray_pts_to a s;

  while (not (sarray_iterator_end !it))
    invariant
      live it
  {
    let r = sarray_iterator_get !it;

    unfold sarray_pts_to a s;

    it := sarray_iterator_next #et #ets #l #a !it #1.0R #s;

    fold sarray_pts_to a s;
  };
}

let sarray_iterator_test_u32 #l = sarray_iterator_test #u32 #_ #l
