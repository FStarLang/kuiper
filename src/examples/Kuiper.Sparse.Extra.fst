
// Estas definiciones deberían hacerse (y demostrarse) en otro lado 

module Kuiper.Sparse.Extra

#lang-pulse
open Kuiper

ghost
fn forevery_map_extra
  (#a:Type0) {| enumerable a |}
  (k : slprop)
  (p1 p2 : a -> slprop)
  (f : (x:a -> stt_ghost unit emp_inames (k ** p1 x) (fun _ -> k ** p2 x)))
  requires
    k ** (forall+ (x:a). p1 x)
  ensures
    k ** (forall+ (x:a). p2 x)
{ admit() }

ghost
fn forevery_extract_if_eqtype
  (#a:eqtype) {| enumerable a |}
  (z : a)
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    p z **
    (forall+ (x:a).
      if x = z then emp else p x)
{ admit() }

ghost
fn forevery_unextract_if_eqtype
  (#a:eqtype) {| enumerable a |}
  (z : a)
  (p : a -> slprop)
  requires
    p z **
    (forall+ (x:a).
      if x = z then emp else p x)
  ensures
   forall+ (x:a). p x
{ admit() }

ghost
fn gpu_slice_gather'
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (m n:nat)
  (k: nat { k > 0 })
  (#f : perm) // FIXME: if we use 'f, it gets type 'real' instead of 'perm'
  requires
    forall+ (_ : natlt k).
      gpu_pts_to_slice arr #(f /. k) m n 'v
  ensures gpu_pts_to_slice arr #f m n 'v
{
  forevery_tostar #(natlt k) _;
  rewrite each Kuiper.Enumerable.cardinal (natlt k) #_ as k;
  gpu_slice_gather arr m n k;
}

[@@allow_ambiguous]
ghost
fn gpu_array_pts_to_eq
  (#a:Type)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f1 #f2 : perm)
  (#s1 #s2 : seq a)
  preserves
    gpu_pts_to_array arr #f1 s1 **
    gpu_pts_to_array arr #f2 s2
  ensures
    pure (s1 == s2)
{
  admit()
}