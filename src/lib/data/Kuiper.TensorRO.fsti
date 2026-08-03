module Kuiper.TensorRO
#lang-pulse

include Kuiper.Shape
include Kuiper.Chest
include Kuiper.Tensor.Layout

open Kuiper
open Kuiper.Injection
open Kuiper.Bijection
open Kuiper.Shape
open Kuiper.Chest
open FStar.Tactics.Typeclasses { no_method }
open Pulse.Lib.Trade
open Kuiper.Shareable

module SZ = Kuiper.SizeT
module T = FStar.Tactics.V2

[@@erasable]
noeq
type vtlayout (#r : erased nat) (d : shape r) = {
  (* Underlying length of base array (Kuiper.Array) *)
  ulen : nat;
  (* Function (not an injection!) from (abstract) index space into base array. *)
  imap : abs d -> GTot (natlt ulen);
}

[@@Tactics.Typeclasses.fundeps [0;1]]
inline_for_extraction
class cvtlayout (#r : erased nat) (#d : shape r) (l : vtlayout d) = {
  ulen_fits : squash (SZ.fits l.ulen);

  [@@@no_method]
  all_fit : squash (all_fit d);

  [@@@no_method]
  cimap : i:conc d -> r:SZ.t{SZ.v r == l.imap (up i)};
}

let vtlayout_ulen (#d : shape 'r) (l : vtlayout d) : GTot nat = l.ulen

let vlayout1 d = vtlayout (d @| INil)
let vlayout2 d1 d2 = vtlayout (d1 @| d2 @| INil)
let vlayout3 d1 d2 d3 = vtlayout (d1 @| d2 @| d3 @| INil)
let vlayout4 d1 d2 d3 d4 = vtlayout (d1 @| d2 @| d3 @| d4 @| INil)


inline_for_extraction noextract
val rotensor (et : Type0) (#r : nat) (#d : shape r) (l : vtlayout d) : Type0

inline_for_extraction noextract
let roarray1 (et : Type0) (#d0 : nat) (l : vlayout1 d0) : Type0 = rotensor et l
inline_for_extraction noextract
let roarray2 (et : Type0) (#d0 #d1 : nat) (l : vlayout2 d0 d1) : Type0 = rotensor et l
inline_for_extraction noextract
let roarray3 (et : Type0) (#d0 #d1 #d2 : nat) (l : vlayout3 d0 d1 d2) : Type0 = rotensor et l
inline_for_extraction noextract
let roarray4 (et : Type0) (#d0 #d1 #d2 #d3 : nat) (l : vlayout4 d0 d1 d2 d3) : Type0 = rotensor et l

val is_global
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l) : prop

inline_for_extraction noextract
let global_rotensor (et : Type0) (#r : nat) (#d : shape r) (l : vtlayout d) : Type0 =
  a : rotensor et l { is_global a }

inline_for_extraction noextract
val from_array
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (l : vtlayout d)
  (a : larray et (vtlayout_ulen l))
  : rotensor et l

inline_for_extraction noextract
val core
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  : larray et (vtlayout_ulen l)

inline_for_extraction noextract
let relay
  (#et : Type0)
  (#r1 : erased nat) (#d1 : shape r1) (#l1 : vtlayout d1)
  (a : rotensor et l1)
  (#r2 : erased nat) (#d2 : shape r2) (l2 : vtlayout d2{l2.ulen == l1.ulen})
  : rotensor et l2
  = from_array l2 (core a)

val lem_core_from_array
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]

val lem_from_array_core
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (p : larray et (vtlayout_ulen l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]

val lem_is_global_iff_core
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  : Lemma (ensures is_global a <==> is_global_array (core a))
          [SMTPat (is_global a)]

val tensor_pts_to
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  ([@@@mkey] a : rotensor et l)
  (#[T.exact (`1.0R)] f : perm)
  (s : chest d et)
  : slprop

instance
val is_send_across_global_tensor
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l { is_global a })
  (#f : perm) (s : chest d et)
  : is_send_across gpu_of (tensor_pts_to a #f s)

unfold
instance has_pts_to_rotensor
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  : has_pts_to (rotensor et l) (chest d et) = {
  pts_to = tensor_pts_to;
}

ghost
fn tensor_pts_to_ref
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  (#f : perm) (#s : chest d et)
  preserves
    a |-> Frac f s
  ensures
    pure (SZ.fits (vtlayout_ulen l))

ghost
fn tensor_pts_to_ref_located
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  (#loc : loc_id)
  (#f : perm) (#s : chest d et)
  preserves
    on loc (a |-> Frac f s)
  ensures
    pure (SZ.fits (vtlayout_ulen l))

ghost
fn tensor_pts_to_eq
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  (#f1 f2 : perm)
  (#s1 #s2 : chest d et)
  requires
    tensor_pts_to a #f1 s1 **
    tensor_pts_to a #f2 s2
  ensures
    tensor_pts_to a #f1 s2 **
    tensor_pts_to a #f2 s2

ghost
fn tensor_share_n
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l) (k : pos)
  (#f : perm) (#s : chest d et)
  requires
    a |-> Frac f s
  ensures
    forall+ (_:natlt k). a |-> Frac (f /. k) s

ghost
fn tensor_gather_n
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l) (k : pos)
  (#f : perm) (#s : chest d et)
  requires
    forall+ (_:natlt k). a |-> Frac (f /. k) s
  ensures
    a |-> Frac f s

ghost
fn tensor_gather_n_underspec
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l) (k : pos)
  (#f : perm)
  requires
    forall+ (_:natlt k).
      exists* (s : chest d et). tensor_pts_to a #(f /. k) s
  ensures
    exists* (s : chest d et). tensor_pts_to a #f s

// Needs to be exposed
// GM: Why? Note to self, write it down.
// inline_for_extraction noextract
// instance ctensor_ciview
//   (#et : Type0) (#r : erased nat) (#d : shape r)
//   (#l : vtlayout d)
//   (c : ctlayout l)
//   : Kuiper.IView.ciview (tensor_aview et l).iview =
// {
//   len_fits = ();
//   sch = {
//     cit = conc d;
//     bij = abs_conc_bij d;
//   };
//   step = {
//     cimap = mk_cinj c.cimap #(fun x y -> down_up x; down_up y);
//   };
// }

inline_for_extraction noextract
fn tensor_read
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : vtlayout d) {| cvtlayout l |}
  (a : rotensor et l)
  (i : conc d)
  (#f : perm)
  (#s : chest d et)
  preserves
    a |-> Frac f s
  returns
    v : et
  ensures
    pure (v == acc s (up i))

// inline_for_extraction noextract
// fn tensor_write
//   (#et : Type0) (#r : erased nat) (#d : shape r)
//   (#l : vtlayout d) {| ctlayout l |}
//   (a : rotensor et l)
//   (i : conc d)
//   (v : et)
//   (#s : chest d et)
//   requires
//     a |-> s
//   ensures
//     a |-> upd s (up i) v

(* Syntax *)
inline_for_extraction noextract
unfold let op_Array_Access
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : vtlayout d) {| cvtlayout l |}
  (a : rotensor et l)
  (i : conc d)
  (#f : perm)
  (#s : chest d et)
  = tensor_read #et #r #d #l a i #f #s

(* Syntax *)
// inline_for_extraction noextract
// unfold let op_Array_Assignment
//   (#et : Type0) (#r : erased nat) (#d : shape r)
//   (#l : vtlayout d) {| ctlayout l |}
//   (a : rotensor et l)
//   (i : conc d)
//   (v : et)
//   (#s : chest d et)
//   = tensor_write #et #r #d #l a i v #s

val tensor_pts_to_cell
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  ([@@@mkey] a : rotensor et l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey] i : abs d)
  (v : et)
  : slprop

[@@pulse_unfold; FStar.Tactics.Typeclasses.noinst]
instance cell_pts_to (#et : Type) (#r : nat) (#d : shape r) (#l : vtlayout d)
  : has_pts_to (cell (rotensor et l) (abs d)) et
= {
  pts_to = (fun (Cell ar i) #f v -> tensor_pts_to_cell ar #f i v);
}

val tensor_pts_to_cell_eq
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l) (i : abs d) (f : perm) (v : et)
  : Lemma (Cell a i |-> Frac f v
           ==
           pts_to_cell (core a) #f (l.imap i) v)

instance
val is_send_across_global_tensor_cell
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l { is_global a })
  (#f : perm) (i : abs d) (v : et)
  : is_send_across gpu_of (tensor_pts_to_cell a #f i v)

inline_for_extraction noextract
fn tensor_read_cell
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : vtlayout d) {| cvtlayout l |}
  (a : rotensor et l)
  (i : conc d)
  (#f : perm)
  (#s : erased et)
  preserves
    Cell a (up i) |-> Frac f s
  returns
    v : et
  ensures
    pure (v == s)

// inline_for_extraction noextract
// fn tensor_write_cell
//   (#et : Type0) (#r : erased nat) (#d : shape r)
//   (#l : vtlayout d) {| ctlayout l |}
//   (a : rotensor et l)
//   (i : conc d)
//   (v : et)
//   (#s : erased et)
//   requires
//     Cell a (up i) |-> s
//   ensures
//     Cell a (up i) |-> v

(* Generic extraction of slices *)

// Move some of this to rotensor.Layout.
let tlayout_slice_imap
  (#n:nat) (d : shape n) (l : vtlayout d)
  (i : natlt n) (j : natlt (d @! i))
  (idx : abs (modulo_i i d))
  : GTot (natlt l.ulen) =
    let idx' = (abs_bring_forward_bij i d).gg (j, idx) in
    l.imap idx'

let vtlayout_slice
  (#n : nat) (#d : shape n) (l : vtlayout d)
  (i : natlt n) (j : natlt (d @! i)) // Fixing the ith-dimension to j
  : vtlayout (modulo_i i d) =
  {
    ulen = l.ulen;
    imap = tlayout_slice_imap d l i j;
      // is_inj = (fun x y -> ());
    // };
  }

(* Note: the codomain of this instance
   has existentially quantified r'/d' so we do
   not force the unifier to prove equalities
   involving integer subtraction or modulo_i. *)
inline_for_extraction noextract
instance val ctlayout_slice
  (#n : erased nat) (#d : shape n) (l : vtlayout d)
  {| cvtlayout l |}
  (i : erased nat{i < n}) (j : erased nat{j < (d @! i)})
  {| ix : concrete_sz i |} {| jx : concrete_sz j |}
  (#r' : erased nat) (#d' : shape r')
  (#_ : reveal r' == n-1)
  (#_ : d' == modulo_i i d)
  : cvtlayout #r' #d' (vtlayout_slice l i j)

inline_for_extraction noextract
val sliceof
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  (i : erased nat{i < r}) (j : erased nat{j < d @! i})
  : rotensor et (vtlayout_slice l i j)

val lem_sliceof_core
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  (i : erased nat{i < r}) (j : erased nat{j < d @! i})
  : Lemma (core (sliceof a i j) == core a)
          [SMTPat (sliceof a i j)]

val lem_is_global_iff_sliceof
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  (i : natlt r) (j : natlt (d @! i))
  : Lemma (ensures is_global (sliceof a i j) <==> is_global a)
          [SMTPat (is_global (sliceof a i j))]

#push-options "--warn_error -271" // implicit subtraction in pattern, OK
val tensor_slice_cell_eq
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  (i : natlt r) (j : natlt (d @! i))
  (k : abs (modulo_i i d)) (f : perm) (v : et)
  : Lemma (Cell (sliceof a i j) k |-> Frac f v
           ==
           Cell a ((abs_bring_forward_bij i d).gg (j, k)) |-> Frac f v)
           [SMTPat (Cell (sliceof a i j) k |-> Frac f v)]
#pop-options

(* NOTE (non-injective / broadcast layouts):
   The read/write and read-only slice-borrow operations that used to live here
   (tensor_extract_slice / tensor_extract_slice_ro / tensor_restore_slice) are
   *unsound* for the general non-injective [vtlayout].  Their trades reconstruct a
   parent chest via [chest_update_slice], which for a broadcast view can demand that
   a single underlying cell hold two different values (the sliced dimension aliases
   other dimensions), so no underlying array realizes the resulting chest.  Even the
   read-only borrow cannot be discharged with the whole-array (broadcast-capable)
   ownership used by [tensor_pts_to], since handing out [sliceof a i j] transfers the
   *entire* backing array and leaves nothing for the trade to pin the non-sliced
   cells with.  The pure/layout-level slicing below ([sliceof], [vtlayout_slice],
   [ctlayout_slice], [tensor_slice_cell_eq], ...) is sound and retained; the ownership
   transfer is intentionally omitted, mirroring the removal of the injective-only
   [ref_of_tensor_cell] family. *)

let vtlayout_bij
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (l : vtlayout d1)
  : vtlayout d2
  = {
      ulen = l.ulen;
      imap = (fun x -> l.imap (f.gg x));
  }

inline_for_extraction noextract
instance val ctlayout_bij
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2 { all_fit d2 })
  (f : abs d1 =~ abs d2)
  (fconc: conc d2 -> conc d1)
  (fconc_correct: (x: conc d2) -> up (fconc x) == f.gg (up x))
  (l : vtlayout d1) {| c: cvtlayout l |}
  : cvtlayout #r2 #d2 (vtlayout_bij f l)

ghost
fn tensor_apply_bij
  (#et : Type0)
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (#l : vtlayout d1) // {| is_full l |} (* Why would it have to be full? *)
  (a : rotensor et l)
  (#fp : perm) (#m : Chest.t d1 et)
  requires
    a |-> Frac fp m
  ensures
    from_array (vtlayout_bij f l) (core a) |-> Frac fp (Chest.mk d2 (fun i -> Chest.acc m (i <~| f)))

ghost
fn tensor_unapply_bij
  (#et : Type0)
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (#l : vtlayout d1)
  (a : rotensor et l)
  (#fp : perm) (#m : Chest.t d1 et)
  requires
    from_array (vtlayout_bij f l) (core a) |->
      Frac fp (Chest.mk d2 (fun i -> Chest.acc m (i <~| f)))
  ensures
    a |-> Frac fp m

// TODO: it should be possible to have just "pts_to_shareable" for any
// types that have pts_to, no? or does that not hold?
instance tensor_pts_to_shareable
  (#et : Type) (#r: erased nat) (#d: shape r) (#l: vtlayout d)
  (t: rotensor et l) (s: chest d et):
  shareable (fun fr -> tensor_pts_to t #fr s) = {
  _share_n = (fun (n: pos) (#fr : perm) -> tensor_share_n t n #fr);
  _gather_n = (fun (n: pos) (#fr : perm) -> tensor_gather_n t n #fr);
}

let drop_fst
  (#r : nat)
  (#d : shape r)
  (#e : nat)
  : abs (e @| d) -> GTot (abs d)
  = function (_, i) -> i

let extended_layout
  (#r : nat) (#d : shape r)
  (l : vtlayout d)
  (e : nat) // new dim
  : vtlayout (e @| d)
  = {
      ulen = l.ulen;
      imap = (fun i -> l.imap (drop_fst i));
    }

inline_for_extraction noextract
instance val cvtlayout_extended
  (#r : erased nat) (#d : shape r)
  (l : vtlayout d) {| cvtlayout l |}
  (e : erased nat{SZ.fits e})
  : cvtlayout (extended_layout l e)

ghost
fn rotensor_add_dim
  (#et : Type0)
  (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  (#f : perm) (#m : Chest.t d et)
  (e : nat) // new dim
  requires
    a |-> Frac f m
  ensures
    from_array (extended_layout l e) (core a) |-> Frac f (Chest.mk (e @| d) (function (_, i) -> acc m i))

(* Value-returning counterpart of [rotensor_add_dim].  The new dimension [e] is
   spec-only (the returned backing array is independent of it, only the erased
   shape/layout grows), hence [erased]: callers usually have [e] as a ghost
   quantity, and feeding an [erased] value to a *concrete* [nat] here would force
   a [reveal] (a ghost coercion) on this *stateful* application, which is
   rejected -- a stateful computation cannot have a ghost effect. *)
inline_for_extraction noextract
fn rotensor_add_dim_view
  (#et : Type0)
  (#r : erased nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  (#f : perm) (#m : Chest.t d et)
  (e : erased nat)
  requires
    a |-> Frac f m
  returns
    b : rotensor et (extended_layout l e)
  ensures
    rewrites_to b (from_array (extended_layout l e) (core a)) **
    b |-> Frac f (Chest.mk (e @| d) (function (_, i) -> acc m i))

ghost
fn rotensor_remove_dim
  (#et : Type0)
  (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  (#f : perm) (#m : Chest.t d et)
  (e : pos)
  requires
    from_array (extended_layout l e) (core a) |->
      Frac f (Chest.mk (e @| d) (function (_, i) -> acc m i))
  ensures
    a |-> Frac f m

(* ------------------------------------------------------------------------ *)
(* Ownership-safe conversion  Tensor <-> TensorRO.

   A *full* (bijective) writable layout [l] is viewed read-only through the
   corresponding [vtlayout] (same backing array, same index map): the borrow is
   zero-copy.  The read-only view exposes the very same chest [s]; it can then
   be reshaped into a zero-copy broadcast with [rotensor_add_dim] /
   [tensor_apply_bij].  The borrow hands back a trade that recovers the
   *unchanged* writable tensor once the read-only view is no longer used (a
   read-only view never mutates the backing store, so this is always sound).

   The public entry points are:

     [tensor_to_rotensor t]      borrow, *returning* the read-only view value;
     [rotensor_to_tensor t a]    give the view back, recovering [t].

   [tensor_borrow_ro] is the underlying ghost primitive: it proves the same
   thing but, being ghost, cannot return the (informative) view value, so
   callers must spell the view out as [rotensor_of_tensor t].  It is kept
   public because it is the only variant usable from ghost code. *)
(* ------------------------------------------------------------------------ *)

module KT = Kuiper.Tensor

(* Reinterpret a writable (injective) layout as a read-only [vtlayout].
   [vtlayout] permits non-injective maps, but a faithful borrow uses the
   original injective map; broadcasting happens afterwards. *)
let vtlayout_of_tlayout (#r : erased nat) (#d : shape r) (l : tlayout d) : vtlayout d =
  { ulen = l.ulen; imap = l.imap.f }

val lem_vtlayout_of_tlayout_ulen (#r : nat) (#d : shape r) (l : tlayout d)
  : Lemma (ensures vtlayout_ulen (vtlayout_of_tlayout l) == tlayout_ulen l)
          [SMTPat (vtlayout_ulen (vtlayout_of_tlayout l))]

(* A concrete writable layout is concrete read-only: reuse its runtime index
   map.  This lets callers [tensor_read] a borrowed read-only view. *)
inline_for_extraction noextract
instance val cvtlayout_of_ctlayout
  (#r : erased nat) (#d : shape r)
  (l : tlayout d) {| c : ctlayout l |}
  : cvtlayout (vtlayout_of_tlayout l)

(* The read-only view of a writable tensor: the same backing array under the
   same index map.  Named once here so that neither the specifications below
   nor client code has to repeat the [from_array .. (KT.core t)] expression. *)
inline_for_extraction noextract
let rotensor_of_tensor
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : tlayout d)
  (t : KT.tensor et l)
  : rotensor et (vtlayout_of_tlayout l)
  = from_array (vtlayout_of_tlayout l) (KT.core t)

(* Ghost primitive.  Prefer [tensor_to_rotensor] in non-ghost code: it returns
   the view instead of forcing clients to name it. *)
ghost
fn tensor_borrow_ro
  (#et : Type0)
  (#r : nat) (#d : shape r)
  (#l : tlayout d { is_full l })
  (t : KT.tensor et l)
  (#f : perm) (#s : chest d et)
  requires
    KT.tensor_pts_to t #f s
  ensures
    factored
      (rotensor_of_tensor t |-> Frac f s)
      (KT.tensor_pts_to t #f s)

(* Borrow [t] as a read-only tensor and *return* that view.  The returned [a]
   is (provably, via [rewrites_to]) [rotensor_of_tensor t], i.e. no copy and no
   allocation happens: at extraction this is the identity on the backing
   pointer.  The accompanying trade gives [t] back, see [rotensor_to_tensor]. *)
inline_for_extraction noextract
fn tensor_to_rotensor
  (#et : Type0)
  (#r : erased nat) (#d : shape r)
  (#l : tlayout d { is_full l })
  (t : KT.tensor et l)
  (#f : perm) (#s : chest d et)
  requires
    KT.tensor_pts_to t #f s
  returns
    a : rotensor et (vtlayout_of_tlayout l)
  ensures
    rewrites_to a (rotensor_of_tensor t) **
    factored
      (a |-> Frac f s)
      (KT.tensor_pts_to t #f s)

(* Give the read-only view [a] of [t] back, recovering the writable tensor with
   its (necessarily unchanged) contents.  [a] must hold the same chest and the
   same fraction it was borrowed with, so any intermediate broadcast view has
   to be undone first (e.g. with [rotensor_remove_dim] / [tensor_unapply_bij]).
   This is the named counterpart of [tensor_to_rotensor]; clients need not
   invoke the generic trade eliminator. *)
ghost
fn rotensor_to_tensor
  (#et : Type0)
  (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (t : KT.tensor et l)
  (a : rotensor et (vtlayout_of_tlayout l))
  (#f : perm) (#s : chest d et)
  requires
    factored
      (a |-> Frac f s)
      (KT.tensor_pts_to t #f s)
  ensures
    KT.tensor_pts_to t #f s
