module Kuiper.FArray
#lang-pulse

open Kuiper
module A = Kuiper.VArray
module SZ = Kuiper.SizeT
module T = FStar.Tactics.V2

let farray (et : Type0) (#len : nat) (l : flayout len) : Type0 =
  A.varray (farray_aview et l)

let is_global_farray (#et : Type0) (#len : nat) (#l : flayout len) (a : farray et l) : prop =
  A.is_global_varray a

let from_array (#et : Type0) (#len : erased nat) (l : flayout len) (a : gpu_array et (flayout_size l)) : farray et l =
  A.from_array (farray_aview et l) a

let core (#et : Type0) (#len : erased nat) (#l : flayout len) (a : farray et l) : gpu_array et (flayout_size l) =
  A.core a

let lem_core_from_array (#et : Type) (#len : erased nat) (#l : flayout len) (a : farray et l)
  : Lemma (ensures from_array l (core a) == a /\ (is_global_array (core a) <==> is_global_farray a))
          [SMTPat (core a)]
  = ()

let lem_from_array_core (#et : Type) (#len : erased nat) (l : flayout len) (p : gpu_array et (flayout_size l))
  : Lemma (ensures core (from_array l p) == p /\ (is_global_farray (from_array l p) <==> is_global_array p))
          [SMTPat (from_array l p)]
  = ()

let farray_pts_to
  (#et : Type) (#len : nat) (#l : flayout len)
  ([@@@mkey] a : farray et l)
  (#[T.exact (`1.0R)] f : perm)
  (s : lseq et len)
  : slprop
  = A.varray_pts_to a #f s

instance is_send_across_global_farray
  (#et : Type0) (#len : nat) (#l : flayout len)
  (a : farray et l { is_global_farray a })
  (#f : perm) (s : lseq et len)
  : is_send_across gpu_of (farray_pts_to a #f s)
  = solve

ghost
fn farray_pts_to_ref
  (#et : Type) (#len : nat) (#l : flayout len)
  (a : farray et l)
  (#f : perm) (#s : erased (lseq et len))
  preserves a |-> Frac f s
  ensures pure (SZ.fits (flayout_size l))
{
  unfold farray_pts_to a #f s;
  A.varray_pts_to_ref a;
  fold farray_pts_to a #f s;
}

ghost
fn farray_share_n
  (#et : Type0) (#len : nat) (#l : flayout len)
  (a : farray et l) (k : pos)
  (#f : perm) (#s : lseq et len)
  requires a |-> Frac f s
  ensures  forall+ (_:natlt k). a |-> Frac (f /. k) s
{
  unfold farray_pts_to a #f s;
  A.varray_share_n a k;
  forevery_map
    (fun (i:natlt k) -> A.varray_pts_to a #(f /. k) s)
    (fun (i:natlt k) -> farray_pts_to a #(f /. k) s)
    fn i { fold farray_pts_to a #(f /. k) s };
}

ghost
fn farray_gather_n
  (#et : Type0) (#len : nat) (#l : flayout len)
  (a : farray et l) (k : pos)
  (#f : perm) (#s : lseq et len)
  requires forall+ (_:natlt k). a |-> Frac (f /. k) s
  ensures  a |-> Frac f s
{
  forevery_map
    (fun (i:natlt k) -> farray_pts_to a #(f /. k) s)
    (fun (i:natlt k) -> A.varray_pts_to a #(f /. k) s)
    fn i { unfold farray_pts_to a #(f /. k) s };
  A.varray_gather_n a k;
  fold farray_pts_to a #f s;
}

inline_for_extraction noextract
fn farray_read
  (#et : Type0)
  (#len : erased nat)
  (#l : flayout len) {| cflayout l |}
  (a : farray et l)
  (i : szlt len)
  (#f : perm)
  (#s : erased (lseq et len))
  preserves a |-> Frac f s
  returns v : et
  ensures pure (v == Seq.index s i)
{
  unfold farray_pts_to a #f s;
  let v = A.varray_read a i;
  fold farray_pts_to a #f s;
  v
}

inline_for_extraction noextract
fn farray_write
  (#et : Type0)
  (#len : erased nat)
  (#l : flayout len) {| cflayout l |}
  (a : farray et l)
  (i : szlt len)
  (v : et)
  (#s : erased (lseq et len))
  requires a |-> s
  ensures  a |-> (Seq.upd s i v <: lseq et len)
{
  unfold farray_pts_to a s;
  A.varray_write a i v;
  fold farray_pts_to a (Seq.upd s i v <: lseq et len);
}

(* ============ CELL-LEVEL OPERATIONS ============ *)

let farray_pts_to_cell
  (#et : Type) (#len : nat) (#l : flayout len)
  ([@@@mkey] a : farray et l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey] i : natlt len)
  (v : et)
  : slprop
  = A.varray_pts_to_cell a #f i v

let farray_pts_to_cell_eq
  (#et : Type) (#len : nat) (#l : flayout len)
  (a : farray et l) (i : natlt len) (f : perm) (v : et)
  : Lemma (farray_pts_to_cell a #f i v
           ==
           gpu_pts_to_cell (core a) #f (l.fmap.f i) v)
  = A.varray_pts_to_cell_eq a i f v

ghost
fn farray_explode
  (#et : Type0) (#len : nat) (#l : flayout len)
  (a : farray et l)
  (#f : perm)
  (#s : lseq et len)
  requires a |-> Frac f s
  ensures
    forall+ (i : natlt len).
      farray_pts_to_cell a #f i (Seq.index s i)
{
  unfold farray_pts_to a #f s;
  A.varray_explode a;
  forevery_ext
    (fun (i : natlt len) ->
      A.varray_pts_to_cell a #f i ((farray_aview et l).ctn.acc s i))
    (fun (i : natlt len) ->
      farray_pts_to_cell a #f i (Seq.index s i));
}

ghost
fn farray_implode
  (#et : Type0) (#len : nat) (#l : flayout len)
  (a : farray et l)
  (#f : perm)
  (#s : lseq et len)
  requires
    pure (SZ.fits (flayout_size l))
  requires
    forall+ (i : natlt len).
      farray_pts_to_cell a #f i (Seq.index s i)
  ensures
    a |-> Frac f s
{
  forevery_ext
    (fun (i : natlt len) ->
      farray_pts_to_cell a #f i (Seq.index s i))
    (fun (i : natlt len) ->
      A.varray_pts_to_cell a #f i ((farray_aview et l).ctn.acc s i));
  A.varray_implode a;
  fold farray_pts_to a #f s;
}
