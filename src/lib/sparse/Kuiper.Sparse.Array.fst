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
  elems : gpu_array et nnz; // elementos (no zero)
  pos   : gpu_array sz nnz; // posición de cada elemento
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
  (#[Tactics.exact (`1.0R)] f : perm)
  (s : seq et)
  (v_elems : lseq et a.nnz)
  (v_pos   : lseq sz a.nnz)
  : slprop
=
    a.elems |-> Frac f v_elems **
    a.pos   |-> Frac f v_pos **
    pure (
      pure_sarray_pts_to l s v_elems v_pos
    )

let sarray_pts_to
  (#et:Type0) {| d : scalar et |} #l
  (a : sarray et l)
  (#[Tactics.exact (`1.0R)] f : perm)
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

  gpu_slice_pts_to_eq a.elems 0 a.nnz f2;
  gpu_slice_pts_to_eq a.pos 0 a.nnz f2;

  with v_elems.
    assert gpu_pts_to_slice a.elems #f1 0 a.nnz v_elems;
    assert gpu_pts_to_slice a.elems #f2 0 a.nnz v_elems;
  with v_pos.
    assert gpu_pts_to_slice a.pos #f1 0 a.nnz v_pos;
    assert gpu_pts_to_slice a.pos #f2 0 a.nnz v_pos;

  fold sarray_pts_to a #f1 v2;
  fold sarray_pts_to a #f2 v2;
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
  with v_elems. assert gpu_pts_to_slice a.elems #f 0 a.nnz v_elems;
  with v_pos. assert gpu_pts_to_slice a.pos #f 0 a.nnz v_pos;

  gpu_slice_share a.elems 0 a.nnz n #f;
  gpu_slice_share a.pos 0 a.nnz n #f;

  forevery_zip (fun _ -> gpu_pts_to_slice a.elems #(f /. n) 0 a.nnz _) _;

  forevery_map #(natlt n)
    (fun _ ->
      gpu_pts_to_slice a.elems #(f /. n) 0 a.nnz v_elems **
      gpu_pts_to_slice a.pos #(f /. n) 0 a.nnz v_pos)
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
fn sarray_gather_n
  (#et:Type0) {| scalar et |}
  (#l : nat)
  (a : sarray et l)
  (n : pos)
  (#f : perm)
  (#s : seq et)
  requires
    forall+ (_ : natlt n). sarray_pts_to a #(f /. n) s
  ensures
    sarray_pts_to a #f s
{
  forevery_natlt_pop n _;

  unfold sarray_pts_to a #(f /. n) s;
  with v_elems.   assert gpu_pts_to_array a.elems   #(f /. n) v_elems;
  with v_pos. assert gpu_pts_to_array a.pos #(f /. n) v_pos;

  ghost
  fn aux (_ : natlt (n-1))
    norewrite
    preserves
      gpu_pts_to_array a.elems #(f /. n) v_elems **
      gpu_pts_to_array a.pos #(f /. n) v_pos
    requires
      sarray_pts_to a #(f /. n) s
    ensures
      gpu_pts_to_array a.elems #(f /. n) v_elems **
      gpu_pts_to_array a.pos #(f /. n) v_pos
  {
    unfold sarray_pts_to a #(f /. n) s;

    gpu_slice_pts_to_eq a.elems 0 a.nnz (f /. n) #_ #v_elems;
    gpu_slice_pts_to_eq a.pos 0 a.nnz (f /. n) #_ #v_pos;
  };

  forevery_map_extra _ _ _ aux;
  forevery_natlt_push n _;

  forevery_unzip #(natlt n) _ _;

  gpu_slice_gather a.elems   _ _ n;
  gpu_slice_gather a.pos _ _ n;

  fold sarray_pts_to a #f s;
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
