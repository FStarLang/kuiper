module Kuiper.Tensor
#lang-pulse

open Kuiper
open Kuiper.Index
module A = Kuiper.VArray
module SZ = Kuiper.SizeT
module T = FStar.Tactics.V2

let tensor (et : Type0) (#r : nat) (#d : idesc r) (l : tlayout d) : Type0 =
  A.varray (tensor_aview et l)

let is_global_tensor
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l) : prop
  = A.is_global_varray a

inline_for_extraction noextract
let from_array
  (#et : Type0) (#r : nat) (#d : idesc r)
  (l : tlayout d)
  (a : gpu_array et (tlayout_size l))
  : tensor et l
  = A.from_array (tensor_aview et l) a

inline_for_extraction noextract
let core
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  : gpu_array et (tlayout_size l)
  = A.core a

let lem_core_from_array
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]
  = ()

let lem_from_array_core
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (p : gpu_array et (tlayout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]
  = ()

let lem_is_global_iff_core
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  : Lemma (ensures is_global_tensor a <==> is_global_array (core a))
          [SMTPat (is_global_tensor a)]
  = ()

let tensor_pts_to
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  ([@@@mkey] a : tensor et l)
  (#[T.exact (`1.0R)] f : perm)
  (s : chest d et)
  : slprop
  = A.varray_pts_to a #f s

instance is_send_across_global_tensor
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l { is_global_tensor a })
  (#f : perm) (s : chest d et)
  : is_send_across gpu_of (tensor_pts_to a #f s)
  = solve

ghost
fn tensor_pts_to_ref
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm) (#s : chest d et)
  preserves
    a |-> Frac f s
  ensures
    pure (SZ.fits (tlayout_size l))
{
  unfold tensor_pts_to a #f s;
  A.varray_pts_to_ref a;
  fold tensor_pts_to a #f s;
}

ghost
fn tensor_share_n
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l) (k : pos)
  (#f : perm) (#s : chest d et)
  requires
    a |-> Frac f s
  ensures
    forall+ (_:natlt k). a |-> Frac (f /. k) s
{
  unfold tensor_pts_to a #f s;
  A.varray_share_n a k;
  forevery_map
    (fun (i:natlt k) -> A.varray_pts_to a #(f /. k) s)
    (fun (i:natlt k) -> tensor_pts_to a #(f /. k) s)
    fn i { fold tensor_pts_to a #(f /. k) s };
}

ghost
fn tensor_gather_n
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l) (k : pos)
  (#f : perm) (#s : chest d et)
  requires
    forall+ (_:natlt k). a |-> Frac (f /. k) s
  ensures
    a |-> Frac f s
{
  forevery_map
    (fun (i:natlt k) -> tensor_pts_to a #(f /. k) s)
    (fun (i:natlt k) -> A.varray_pts_to a #(f /. k) s)
    fn i { unfold tensor_pts_to a #(f /. k) s };
  A.varray_gather_n a k;
  fold tensor_pts_to a #f s;
}

inline_for_extraction noextract
fn tensor_read
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d) {| ctlayout l |}
  (a : tensor et l)
  (i : conc d)
  (#f : perm)
  (#s : chest d et)
  preserves
    a |-> Frac f s
  returns
    v : et
  ensures
    pure (v == acc s ((abs_conc_bij d).gg i))
{
  unfold tensor_pts_to a #f s;
  let v = A.varray_read a i;
  fold tensor_pts_to a #f s;
  v
}

inline_for_extraction noextract
fn tensor_write
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d) {| ctlayout l |}
  (a : tensor et l)
  (i : conc d)
  (v : et)
  (#s : chest d et)
  requires
    a |-> s
  ensures
    a |-> upd s ((abs_conc_bij d).gg i) v
{
  unfold tensor_pts_to a s;
  A.varray_write a i v;
  fold tensor_pts_to a;
}

(* ============ CELL-LEVEL OPERATIONS ============ *)

let tensor_pts_to_cell
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  ([@@@mkey] a : tensor et l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey] i : abs d)
  (v : et)
  : slprop
  = A.varray_pts_to_cell a #f i v

let tensor_pts_to_cell_eq
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l) (i : abs d) (f : perm) (v : et)
  : Lemma (tensor_pts_to_cell a #f i v
           ==
           gpu_pts_to_cell (core a) #f (l.imap.f i) v)
  = A.varray_pts_to_cell_eq a i f v

ghost
fn tensor_explode
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm)
  (#s : chest d et)
  requires
    a |-> Frac f s
  ensures
    forall+ (i : abs d).
      tensor_pts_to_cell a #f i (acc s i)

{
  unfold tensor_pts_to a #f s;
  A.varray_explode a;

  forevery_rw_type _ (abs d) _;
  forevery_ext
    (fun (i : abs d) ->
      A.varray_pts_to_cell a #f i ((tensor_aview et l).ctn.acc s i))
    (fun (i : abs d) ->
      tensor_pts_to_cell a #f i (acc s i));
  ();
}

ghost
fn tensor_implode
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm)
  (#s : chest d et)
  requires
    pure (SZ.fits (tlayout_size l))
  requires
    forall+ (i : abs d).
      tensor_pts_to_cell a #f i (acc s i)
  ensures
    a |-> Frac f s
{
  forevery_ext
    (fun (i : abs d) ->
      tensor_pts_to_cell a #f i (acc s i))
    (fun (i : abs d) ->
      A.varray_pts_to_cell a #f i ((tensor_aview et l).ctn.acc s i));
  forevery_rw_type _ (tensor_aview et l).iview.ait _;
  A.varray_implode a;
  fold tensor_pts_to a #f s;
}
