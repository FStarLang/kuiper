module Kuiper.Tensor
#lang-pulse

open Kuiper
open Kuiper.Injection
open Kuiper.Index
open Kuiper.Chest
include Kuiper.TensorLayout
open FStar.Tactics.Typeclasses { no_method }
open Pulse.Lib.Trade
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

let cimap_inj
  (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (c : ctlayout l)
  (x y : conc d)
  : Lemma (requires cimap c x == cimap c y)
          (ensures x == y)
          [SMTPat (cimap c x); SMTPat (cimap c y)]
          // ^ Bad pattern probably, used below.
  = down_up x;
    down_up y;
    ()

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
    pure (v == acc s (up i))

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
    a |-> upd s (up i) v

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

(* Generic extraction of slices *)

// Move some of this to TensorLayout.
let tlayout_slice_imap
  (#n:nat) (d : idesc n) (l : tlayout d)
  (i : natlt n) (j : natlt (d @! i))
  (idx : abs (modulo_i i d))
  : GTot (natlt l.ulen) =
    let idx' = (abs_bring_forward_bij i d).gg (j, idx) in
    l.imap.f idx'

let tlayout_slice
  (#n:nat) (d : idesc n) (l : tlayout d)
  (i : natlt n) (j : natlt (d @! i)) // Fixing the ith-dimension to j
  : tlayout (modulo_i i d) =
  {
    ulen = l.ulen;
    imap = {
      f = tlayout_slice_imap d l i j;
      is_inj = (fun x y -> ());
    };
  }

inline_for_extraction noextract
let ctlayout_slice_cimap
  (#n:nat) (d : idesc n) (l : tlayout d)
  {| c : ctlayout l |}
  (i : szlt n) (j : szlt (d @! i))
  (idx : conc (modulo_i i d))
  : Tot (x : szlt l.ulen{SZ.v x == tlayout_slice_imap d l i j (up idx)}) =
    let idx' = (c_conc_bring_forward_bij i d).cgg (j, idx) in
    let res = c.cimap idx' in
    calc (==) {
      SZ.v res;
      == {}
      SZ.v (c.cimap ((c_conc_bring_forward_bij i d).cgg (j, idx)));
      == {}
      l.imap.f (up ((c_conc_bring_forward_bij i d).cgg (j, idx)));
      == { bring_forward_commute2 i d j idx }
      l.imap.f ((abs_bring_forward_bij i d).gg (SZ.v j, up idx));
      == {}
      tlayout_slice_imap d l i j (up idx);
    };
    res

inline_for_extraction noextract
instance ctlayout_slice
  (#n:nat) (d : idesc n) (l : tlayout d)
  {| c : ctlayout l |}
  (i : szlt n) (j : szlt (d @! i))
  : ctlayout (tlayout_slice d l i j) =
  {
    culen = c.culen;
    all_fit = ();
    cimap = (fun idx -> ctlayout_slice_cimap d l i j idx);
  }

inline_for_extraction noextract
val sliceof
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : natlt r) (j : natlt (d @! i))
  : tensor et (tlayout_slice d l i j)

#push-options "--warn_error -271" // implicit subtraction in pattern, OK
val tensor_slice_cell_eq
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : natlt r) (j : natlt (d @! i))
  (k : abs (modulo_i i d)) (f : perm) (v : et)
  : Lemma (tensor_pts_to_cell (sliceof a i j) #f k v
           ==
           tensor_pts_to_cell a #f ((abs_bring_forward_bij i d).gg (j, k)) v)
           [SMTPat (tensor_pts_to_cell (sliceof a i j) #f k v)]
#pop-options

let chest_slice
  (#et : Type0) (#r : nat) (#d : idesc r)
  (i : natlt r) (j : natlt (d @! i))
  (s : chest d et)
  : chest (modulo_i i d) et
  = mk _ (fun (idx : abs (modulo_i i d)) ->
            acc s ((abs_bring_forward_bij i d).gg (j, idx)))

let chest_update_slice
  (#et : Type0) (#r : nat) (#d : idesc r)
  (i : natlt r) (j : natlt (d @! i))
  (s : chest d et) (s' : chest (modulo_i i d) et)
  : chest d et
  = mk _ (fun (idx : abs d) ->
            let (j', k) = (abs_bring_forward_bij i d).ff idx in
            if j' = j then acc s' k else acc s idx)

ghost
fn tensor_extract_slice
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : natlt r) (j : natlt (d @! i))
  (#f : perm) (#s : chest d et)
  requires
    a |-> Frac f s
  ensures
    sliceof a i j |-> Frac f (chest_slice i j s) **
    (forall* (s' : chest (modulo_i i d) et).
      sliceof a i j |-> Frac f s' @==>
      a |-> Frac f (chest_update_slice i j s s'))

ghost
fn tensor_extract_slice_ro
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : natlt r) (j : natlt (d @! i))
  (#f : perm) (#s : chest d et)
  requires
    a |-> Frac f s
  ensures
    factored
      (sliceof a i j |-> Frac f (chest_slice i j s))
      (a |-> Frac f s)

ghost
fn tensor_restore_slice
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : natlt r) (j : natlt (d @! i))
  (#f : perm) (#s : chest d et)
  requires
    factored
      (sliceof a i j |-> Frac f (chest_slice i j s))
      (a |-> Frac f s)
  ensures
    a |-> Frac f s
