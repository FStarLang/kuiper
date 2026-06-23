module Kuiper.Tensor
#lang-pulse

open Kuiper
open Kuiper.Injection
open Kuiper.Index
open Kuiper.Chest
include Kuiper.Tensor.Layout
open FStar.Tactics.Typeclasses { no_method }
open Pulse.Lib.Trade
module SZ = Kuiper.SizeT
module T = FStar.Tactics.V2

inline_for_extraction noextract
val tensor (et : Type0) (#r : nat) (#d : idesc r) (l : tlayout d) : Type0

val is_global
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l) : prop

inline_for_extraction noextract
val from_array
  (#et : Type0) (#r : erased nat) (#d : idesc r)
  (l : tlayout d)
  (a : larray et (tlayout_ulen l))
  : tensor et l

inline_for_extraction noextract
val core
  (#et : Type0) (#r : erased nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  : larray et (tlayout_ulen l)

val lem_core_from_array
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]

val lem_from_array_core
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (p : larray et (tlayout_ulen l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]

val lem_is_global_iff_core
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  : Lemma (ensures is_global a <==> is_global_array (core a))
          [SMTPat (is_global a)]

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
  (a : tensor et l { is_global a })
  (#f : perm) (s : chest d et)
  : is_send_across gpu_of (tensor_pts_to a #f s)

unfold
instance has_pts_to_tensor
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  : has_pts_to (tensor et l) (chest d et) = {
  pts_to = tensor_pts_to;
}

(* This is slightly odd, the user needs to give the total size
instead of each dimension. *)
inline_for_extraction noextract
fn alloc0
  (#et:Type) {| sized et |}
  (#r : nat) (#d : idesc r)
  (s : szp{SZ.v s == sizeof d})
  (l : tlayout d { is_full l })
  preserves
    cpu
  returns
    p : tensor et l
  ensures
    exists* em. on gpu_loc (p |-> em)
  ensures
    pure (is_global p) **
    pure (is_full_array (core p))

inline_for_extraction noextract
fn free
  (#et:Type)
  (#r : nat) (#d : idesc r)
  (#l : tlayout d { is_full l })
  (p : tensor et l)
  (#em : chest d et)
  preserves
    cpu
  requires
    pure (is_full_array (core p)) **
    on gpu_loc (p |-> em)
  ensures emp

ghost
fn tensor_pts_to_ref
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm) (#s : chest d et)
  preserves
    a |-> Frac f s
  ensures
    pure (SZ.fits (tlayout_ulen l))

ghost
fn tensor_pts_to_ref_located
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (#loc : loc_id)
  (#f : perm) (#s : chest d et)
  preserves
    on loc (a |-> Frac f s)
  ensures
    pure (SZ.fits (tlayout_ulen l))

ghost
fn tensor_pts_to_eq
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f1 f2 : perm)
  (#s1 #s2 : chest d et)
  requires
    tensor_pts_to a #f1 s1 **
    tensor_pts_to a #f2 s2
  ensures
    tensor_pts_to a #f1 s2 **
    tensor_pts_to a #f2 s2

ghost
fn tensor_concr
  (#et:Type)
  (#r : nat) (#d : idesc r)
  (#l : tlayout d { is_full l })
  (g : tensor et l)
  (#s : chest d et)
  (#f : perm)
  requires
    g |-> Frac f s
  ensures
    core g |-> Frac f (to_seq l s)

ghost
fn tensor_abs
  (#et:Type)
  (#r : nat) (#d : idesc r)
  (l : tlayout d { is_full l })
  (p : larray et (tlayout_ulen l))
  (#f : perm)
  (#s : chest d et)
  requires
    p |-> Frac f (to_seq l s)
  ensures
    from_array l p |-> Frac f s

ghost
fn tensor_abs'
  (#et:Type)
  (#r : nat) (#d : idesc r)
  (l : tlayout d { is_full l })
  (p : larray et (tlayout_ulen l))
  (#f : perm)
  (#s : lseq et (tlayout_ulen l))
  requires
    p |-> Frac f s
  ensures
    from_array l p |-> Frac f (from_seq l s)

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

// Needs to be exposed
inline_for_extraction noextract
instance ctensor_ciview
  (#et : Type0) (#r : erased nat) (#d : idesc r)
  (#l : tlayout d)
  (c : ctlayout l)
  : Kuiper.IView.ciview (tensor_aview et l).iview =
{
  len_fits = ();
  sch = {
    cit = conc d;
    bij = abs_conc_bij d;
  };
  step = {
    cimap = mk_cinj c.cimap #(fun x y -> down_up x; down_up y);
    compat = ez;
  };
}

inline_for_extraction noextract
fn tensor_read
  (#et : Type0) (#r : erased nat) (#d : idesc r)
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
  (#et : Type0) (#r : erased nat) (#d : idesc r)
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

[@@pulse_unfold; FStar.Tactics.Typeclasses.noinst]
instance cell_pts_to (#et : Type) (#r : nat) (#d : idesc r) (#l : tlayout d)
  : has_pts_to (cell (tensor et l) (abs d)) et
= {
  pts_to = (fun (Cell ar i) #f v -> tensor_pts_to_cell ar #f i v);
}

val tensor_pts_to_cell_eq
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l) (i : abs d) (f : perm) (v : et)
  : Lemma (Cell a i |-> Frac f v
           ==
           pts_to_cell (core a) #f (l.imap.f i) v)

instance
val is_send_across_global_tensor_cell
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l { is_global a })
  (#f : perm) (i : abs d) (v : et)
  : is_send_across gpu_of (tensor_pts_to_cell a #f i v)

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
      Cell a i |-> Frac f (acc s i)

ghost
fn tensor_implode
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm)
  (#s : chest d et)
  requires
    pure (SZ.fits (tlayout_ulen l))
  requires
    forall+ (i : abs d).
      Cell a i |-> Frac f (acc s i)
  ensures
    a |-> Frac f s

ghost
fn tensor_ilower
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm)
  (#s : chest d et)
  requires
    a |-> Frac f s
  ensures
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (i : abs d).
      pts_to_cell (core a) #f (l.imap.f i) (acc s i))

ghost
fn tensor_iraise
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm)
  (#s : chest d et)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (i : abs d).
      pts_to_cell (core a) #f (l.imap.f i) (acc s i))
  ensures
    a |-> Frac f s

inline_for_extraction noextract
fn tensor_read_cell
  (#et : Type0) (#r : erased nat) (#d : idesc r)
  (#l : tlayout d) {| ctlayout l |}
  (a : tensor et l)
  (i : conc d)
  (#f : perm)
  (#s : erased et)
  preserves
    Cell a (up i) |-> Frac f s
  returns
    v : et
  ensures
    pure (v == s)

inline_for_extraction noextract
fn tensor_write_cell
  (#et : Type0) (#r : erased nat) (#d : idesc r)
  (#l : tlayout d) {| ctlayout l |}
  (a : tensor et l)
  (i : conc d)
  (v : et)
  (#s : erased et)
  requires
    Cell a (up i) |-> s
  ensures
    Cell a (up i) |-> v

(* Generic extraction of slices *)

// Move some of this to Tensor.Layout.
let tlayout_slice_imap
  (#n:nat) (d : idesc n) (l : tlayout d)
  (i : natlt n) (j : natlt (d @! i))
  (idx : abs (modulo_i i d))
  : GTot (natlt l.ulen) =
    let idx' = (abs_bring_forward_bij i d).gg (j, idx) in
    l.imap.f idx'

let tlayout_slice
  (#n : nat) (#d : idesc n) (l : tlayout d)
  (i : natlt n) (j : natlt (d @! i)) // Fixing the ith-dimension to j
  : tlayout (modulo_i i d) =
  {
    ulen = l.ulen;
    imap = {
      f = tlayout_slice_imap d l i j;
      is_inj = (fun x y -> ());
    };
  }

(* Note: the codomain of this instance
   has existentially quantified r'/d' so we do
   not force the unifier to prove equalities
   involving integer subtraction or modulo_i. *)
inline_for_extraction noextract
instance val ctlayout_slice
  (#n : erased nat) (#d : idesc n) (l : tlayout d)
  {| ctlayout l |}
  (i : erased nat{i < n}) (j : erased nat{j < (d @! i)})
  {| ix : concrete_sz i |} {| jx : concrete_sz j |}
  (#r' : erased nat) (#d' : idesc r')
  (#_ : reveal r' == n-1)
  (#_ : d' == modulo_i i d)
  : ctlayout #r' #d' (tlayout_slice l i j)

inline_for_extraction noextract
val sliceof
  (#et : Type0) (#r : erased nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : erased nat{i < r}) (j : erased nat{j < d @! i})
  : tensor et (tlayout_slice l i j)

val lem_sliceof_core
  (#et : Type0) (#r : erased nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : erased nat{i < r}) (j : erased nat{j < d @! i})
  : Lemma (core (sliceof a i j) == core a)
          [SMTPat (sliceof a i j)]

val lem_is_global_iff_sliceof
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : natlt r) (j : natlt (d @! i))
  : Lemma (ensures is_global (sliceof a i j) <==> is_global a)
          [SMTPat (is_global (sliceof a i j))]

#push-options "--warn_error -271" // implicit subtraction in pattern, OK
val tensor_slice_cell_eq
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : natlt r) (j : natlt (d @! i))
  (k : abs (modulo_i i d)) (f : perm) (v : et)
  : Lemma (Cell (sliceof a i j) k |-> Frac f v
           ==
           Cell a ((abs_bring_forward_bij i d).gg (j, k)) |-> Frac f v)
           [SMTPat (Cell (sliceof a i j) k |-> Frac f v)]
#pop-options

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
