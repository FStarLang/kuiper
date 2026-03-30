module Kuiper.Tensor
#lang-pulse

open Kuiper
open Kuiper.Injection
open Kuiper.Index
open Kuiper.Chest
include Kuiper.TensorLayout
open FStar.Tactics.Typeclasses { no_method }
module V = Kuiper.View
module SZ = Kuiper.SizeT
module T = FStar.Tactics.V2

inline_for_extraction noextract
val tensor (et : Type0) (#r : nat) (#d : idesc r) (l : tlayout d) : Type0

val is_global_tensor
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l) : prop

inline_for_extraction noextract
val from_array
  (#et : Type0) (#r : nat) (#d : idesc r)
  (l : tlayout d)
  (a : gpu_array et (tlayout_size l))
  : tensor et l

inline_for_extraction noextract
val core
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  : gpu_array et (tlayout_size l)

val lem_core_from_array
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]

val lem_from_array_core
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (p : gpu_array et (tlayout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]

val lem_is_global_iff_core
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  : Lemma (ensures is_global_tensor a <==> is_global_array (core a))
          [SMTPat (is_global_tensor a)]

val tensor_pts_to
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  ([@@@mkey] a : tensor et l)
  (#[T.exact (`1.0R)] f : perm)
  (s : chest d et)
  : slprop

instance
val is_send_across_global_tensor
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l { is_global_tensor a })
  (#f : perm) (s : chest d et)
  : is_send_across gpu_of (tensor_pts_to a #f s)

unfold
instance has_pts_to_tensor
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  : has_pts_to (tensor et l) (chest d et) = {
  pts_to = tensor_pts_to;
}

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

(* Helper for below *)
inline_for_extraction noextract
let cimap
  (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (c : ctlayout l)
  : conc d -> szlt l.ulen
  = c.cimap

inline_for_extraction noextract
instance ctensor_ciview
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (c : ctlayout l)
  : Kuiper.IView.ciview (tensor_aview et l).iview =
{
  clen = c.culen;
  sch = {
    cit = conc d;
    bij = abs_conc_bij d;
  };
  step = {
    cimap = mk_cinj (cimap c);
    (* ^ Using c.cimap directly fails *)
    compat = ez;
  };
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

val tensor_pts_to_cell
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  ([@@@mkey] a : tensor et l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey] i : abs d)
  (v : et)
  : slprop

val tensor_pts_to_cell_eq
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l) (i : abs d) (f : perm) (v : et)
  : Lemma (tensor_pts_to_cell a #f i v
           ==
           gpu_pts_to_cell (core a) #f (l.imap.f i) v)

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

ghost
fn tensor_implode
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm)
  (#s : chest d et)
  requires
    forall+ (i : abs d).
      tensor_pts_to_cell a #f i (acc s i)
  ensures
    a |-> Frac f s
