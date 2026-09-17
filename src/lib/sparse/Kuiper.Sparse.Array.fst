module Kuiper.Sparse.Array

#lang-pulse
open Kuiper
open Kuiper.Sparse.Common
module SZ = FStar.SizeT

// This is here to force extraction.
let _ = 1ul

(* Sparse array *)

noeq
inline_for_extraction
type sarray (et : Type0)
  (l : erased nat) =
  // ^ longitud "virtual" del array
{ nnz   : sz; // número de no-zeros len   : (len : sz {SZ.v len == reveal l}); // longitud "real" del array virtual
  elems : larray et nnz; // elementos (no zero)
  pos   : larray sz nnz; // posición de cada elemento
}


unfold
let pure_sarray_pts_to
  (#et:Type0) {| d : scalar et |}
  (l #nnz : nat)
  (s : seq et)
  (v_elems : lseq et nnz)
  (v_pos   : lseq sz nnz)
: prop
=
  valid_pos l (cast_pos #nnz v_pos <: lseq nat nnz)
  /\ s == unsparse nnz l v_elems (cast_pos v_pos)

unfold
let sarray_pts_to'
  (#et:Type0) {| d : scalar et |} (#l : nat)
  (a : sarray et l)
  (#[full_default ()] f : perm)
  (s : seq et)
  (v_elems : lseq et a.nnz)
  (v_pos   : lseq sz a.nnz)
  : slprop
=
    a.elems |-> Frac f v_elems **
    a.pos   |-> Frac f v_pos   **
    pure (
      pure_sarray_pts_to l s v_elems v_pos
    )

let sarray_pts_to
  (#et:Type0) {| d : scalar et |} #l
  (a : sarray et l)
  (#[full_default ()] f : perm)
  (s : seq et)
  : slprop
=
  exists* (v_elems : lseq et a.nnz) (v_pos : lseq sz a.nnz).
    sarray_pts_to' a #f s v_elems v_pos

inline_for_extraction noextract
unfold
instance has_pts_to_sarray
  (#et: Type0) (#l : nat) {| scalar et |}
  : has_pts_to (sarray et l) (seq et) =
{
  pts_to = sarray_pts_to;
}

ghost
fn array_pts_to_eq
  (#a:Type u#0)
  (arr : array a)
  (#f1 f2 : perm)
  (#v1 #v2 : seq a)
  preserves
    arr |-> Frac f1 v1 **
    arr |-> Frac f2 v2
  ensures
    pure (v1 == v2)
{
  to_mask arr #f1;
  to_mask arr #f2;
  mask_gather arr;
  mask_share_gen arr f1 f2;
  assert pure (Seq.equal v1 v2);
  from_mask arr #f1;
  from_mask arr #f2;
  with v1'. assert arr |-> Frac f1 v1';
  assert pure (Seq.equal v1' v1);
  with v2'. assert arr |-> Frac f2 v2';
  assert pure (Seq.equal v2' v2);
  rewrite arr |-> Frac f1 v1' as arr |-> Frac f1 v1;
  rewrite arr |-> Frac f2 v2' as arr |-> Frac f2 v2;
}

ghost
fn sarray_pts_to_eq
  (#et:Type0) {| scalar et |}
  (#l : nat)
  (a : sarray et l)
  (#f1 f2 : perm)
  (#v1 #v2 : seq et)
  requires
    sarray_pts_to a #f1 v1 **
    sarray_pts_to a #f2 v2
  ensures
    sarray_pts_to a #f1 v2 **
    sarray_pts_to a #f2 v2
{
  unfold sarray_pts_to a #f1 v1;
  unfold sarray_pts_to a #f2 v2;

  array_pts_to_eq a.elems f2;
  array_pts_to_eq a.pos f2;

  fold sarray_pts_to a #f1 v2;
  fold sarray_pts_to a #f2 v2;
}

ghost
fn rec array_share_n
  (#a: Type0) // si agrego el universo u fstar se rompe
  (arr:array a)
  (n : pos)
  (#p :perm)
  (#s:Ghost.erased (Seq.seq a))
  requires arr |-> Frac p s
  ensures forall+ (_ : natlt n). arr |-> Frac (p /. n) s
  decreases n
{
  if (n = 1)
  {
    forevery_singleton_intro (fun (_ : natlt n) -> arr |-> Frac (p /. n) s);
  }
  else
  {
    to_mask arr;

    mask_share_gen arr (p /. n) ((n - 1) *. (p /. n));

    from_mask arr #(p /. n);
    with v. assert arr |-> Frac (p /. n) v;
    assert pure (Seq.equal v s);

    from_mask arr;
    with v. assert arr |-> Frac ((n - 1) *. (p /. n)) v;
    assert pure (Seq.equal v s);

    array_share_n arr (n - 1) #((n - 1) *. (p /. n));

    forevery_ext _ (fun _ -> arr |-> Frac (p /. n) s);
    forevery_natlt_push n (fun _ -> arr |-> Frac (p /. n) s);
  }
}

ghost
fn sarray_share_n
  (#et:Type0) {| scalar et |}
  (#l : nat)
  (a : sarray et l)
  (n : pos)
  (#f : perm)
  (#s : seq et)
  requires
    a |-> Frac f s
  ensures
    forall+ (_ : natlt n). a |-> Frac (f /. n) s
{
  unfold sarray_pts_to a #f s;
  with v_elems. assert a.elems |-> Frac f v_elems;
  with v_pos.   assert a.pos   |-> Frac f v_pos;

  array_share_n a.elems n #f;
  array_share_n a.pos n #f;

  forevery_zip (fun _ -> a.elems |-> Frac (f /. n) _) _;

  forevery_map #(natlt n)
    (fun _ ->
      a.elems |-> Frac (f /. n) v_elems **
      a.pos   |-> Frac (f /. n) v_pos)
    (fun _ -> a |-> Frac (f /. n) s)
    fn _ { fold sarray_pts_to a #(f /. n) s };
}

ghost
fn sarray_share
  (#et:Type0) {| scalar et |}
  (#l : nat)
  (a : sarray et l)
  (#f : perm)
  (#s : seq et)
  requires
    sarray_pts_to a #f s
  ensures
    sarray_pts_to a #(f /. 2) s **
    sarray_pts_to a #(f /. 2) s
{
  sarray_share_n a 2;
  forevery_natlt_pop 2 _;
  forevery_natlt_pop 1 _;
  forevery_elim_empty _;
}

ghost
fn rec sarray_gather_n_aux
  (#et:Type0) {| scalar et |}
  (#l : nat)
  (a : sarray et l)
  (n : pos)
  (#f : perm)
  (#s : seq et)
  requires
    forall+ (_ : natlt n). a |-> Frac f s
  ensures
    a |-> Frac (f *. n) s
  decreases n
{
  if (n = 1)
  {
    forevery_singleton_elim #(natlt n) _;
    rewrite each f as (f *. n);
  }
  else
  {
    forevery_natlt_pop n _;
    unfold sarray_pts_to a #f s;

    sarray_gather_n_aux a (n - 1) #f #s;
    unfold sarray_pts_to a #(f *. (n - 1)) s;

    gather a.elems;
    gather a.pos;

    fold sarray_pts_to a #(f *. n) s;
  }
}

ghost
fn sarray_gather_n
  (#et:Type0) {| scalar et |}
  (#l : nat)
  (a : sarray et l)
  (n : pos)
  (#f : perm)
  (#s : seq et)
  requires
    forall+ (_ : natlt n). a |-> Frac (f /. n) s
  ensures
    a |-> Frac f s
{
  sarray_gather_n_aux a n;
  rewrite each (f /. n *. n) as f;
}

ghost
fn sarray_gather
  (#et:Type0) {| scalar et |}
  (#l : nat)
  (a : sarray et l)
  (#f : perm)
  (#s : seq et)
  requires
    sarray_pts_to a #(f /. 2) s **
    sarray_pts_to a #(f /. 2) s
  ensures
    sarray_pts_to a #f s
{
  forevery_intro_empty #(natlt 0) (fun _ -> sarray_pts_to a #(f /. 2) s);
  forevery_natlt_push_shift 1 _;
  forevery_natlt_push_shift 2 _;
  sarray_gather_n a 2;
}
