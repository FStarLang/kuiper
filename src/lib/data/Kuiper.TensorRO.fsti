module Kuiper.TensorRO
#lang-pulse

include Kuiper.Shape
include Kuiper.Chest
include Kuiper.Tensor.Layout

open Kuiper
open Kuiper.Tensor
open Kuiper.Injection
open Kuiper.Bijection
open Kuiper.Shape
open Kuiper.Chest
open FStar.Tactics.Typeclasses { no_method }
open Pulse.Lib.Trade
open Kuiper.Shareable

module SZ = Kuiper.SizeT
module T = FStar.Tactics.V2

inline_for_extraction noextract
val rotensor (et : Type0) (#r : nat) (#d : shape r) (l : tlayout d) : Type0

inline_for_extraction noextract
let roarray1 (et : Type0) (#d0 : nat) (l : layout1 d0) : Type0 = rotensor et l
inline_for_extraction noextract
let roarray2 (et : Type0) (#d0 #d1 : nat) (l : layout2 d0 d1) : Type0 = rotensor et l
inline_for_extraction noextract
let roarray3 (et : Type0) (#d0 #d1 #d2 : nat) (l : layout3 d0 d1 d2) : Type0 = rotensor et l
inline_for_extraction noextract
let roarray4 (et : Type0) (#d0 #d1 #d2 #d3 : nat) (l : layout4 d0 d1 d2 d3) : Type0 = rotensor et l

val is_global
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l) : prop

inline_for_extraction noextract
let global_rotensor (et : Type0) (#r : nat) (#d : shape r) (l : tlayout d) : Type0 =
  a : rotensor et l { is_global a }

inline_for_extraction noextract
val from_array
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (l : tlayout d)
  (a : larray et (tlayout_ulen l))
  : rotensor et l

inline_for_extraction noextract
val core
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l)
  : larray et (tlayout_ulen l)

inline_for_extraction noextract
let relay
  (#et : Type0)
  (#r1 : erased nat) (#d1 : shape r1) (#l1 : tlayout d1)
  (a : rotensor et l1)
  (#r2 : erased nat) (#d2 : shape r2) (l2 : tlayout d2{l2.ulen == l1.ulen})
  : rotensor et l2
  = from_array l2 (core a)

val lem_core_from_array
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]

val lem_from_array_core
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (p : larray et (tlayout_ulen l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]

val lem_is_global_iff_core
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l)
  : Lemma (ensures is_global a <==> is_global_array (core a))
          [SMTPat (is_global a)]

val tensor_pts_to
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  ([@@@mkey] a : rotensor et l)
  (#[T.exact (`1.0R)] f : perm)
  (s : chest d et)
  : slprop

instance
val is_send_across_global_tensor
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l { is_global a })
  (#f : perm) (s : chest d et)
  : is_send_across gpu_of (tensor_pts_to a #f s)

unfold
instance has_pts_to_rotensor
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  : has_pts_to (rotensor et l) (chest d et) = {
  pts_to = tensor_pts_to;
}

(* This is slightly odd, the user needs to give the total size
instead of each dimension. *)
inline_for_extraction noextract
fn alloc0
  (#et:Type) {| sized et |}
  (#r : nat) (#d : shape r)
  (s : szp{SZ.v s == sizeof d})
  (l : tlayout d { is_full l })
  preserves
    cpu
  returns
    p : rotensor et l
  ensures
    exists* em. on gpu_loc (p |-> em)
  ensures
    pure (is_global p) **
    pure (is_full_array (core p))

inline_for_extraction noextract
fn free
  (#et:Type)
  (#r : nat) (#d : shape r)
  (#l : tlayout d { is_full l })
  (p : rotensor et l)
  (#em : chest d et)
  preserves
    cpu
  requires
    pure (is_full_array (core p)) **
    on gpu_loc (p |-> em)
  ensures emp

ghost
fn tensor_pts_to_ref
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l)
  (#f : perm) (#s : chest d et)
  preserves
    a |-> Frac f s
  ensures
    pure (SZ.fits (tlayout_ulen l))

ghost
fn tensor_pts_to_ref_located
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l)
  (#loc : loc_id)
  (#f : perm) (#s : chest d et)
  preserves
    on loc (a |-> Frac f s)
  ensures
    pure (SZ.fits (tlayout_ulen l))

ghost
fn tensor_pts_to_eq
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
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
fn tensor_concr
  (#et:Type)
  (#r : nat) (#d : shape r)
  (#l : tlayout d { is_full l })
  (g : rotensor et l)
  (#s : chest d et)
  (#f : perm)
  requires
    g |-> Frac f s
  ensures
    core g |-> Frac f (to_seq l s)

ghost
fn tensor_abs
  (#et:Type)
  (#r : nat) (#d : shape r)
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
  (#r : nat) (#d : shape r)
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
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l) (k : pos)
  (#f : perm) (#s : chest d et)
  requires
    a |-> Frac f s
  ensures
    forall+ (_:natlt k). a |-> Frac (f /. k) s

ghost
fn tensor_gather_n
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l) (k : pos)
  (#f : perm) (#s : chest d et)
  requires
    forall+ (_:natlt k). a |-> Frac (f /. k) s
  ensures
    a |-> Frac f s

ghost
fn tensor_gather_n_underspec
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l) (k : pos)
  (#f : perm)
  requires
    forall+ (_:natlt k).
      exists* (s : chest d et). tensor_pts_to a #(f /. k) s
  ensures
    exists* (s : chest d et). tensor_pts_to a #f s

// Needs to be exposed
inline_for_extraction noextract
instance ctensor_ciview
  (#et : Type0) (#r : erased nat) (#d : shape r)
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
  };
}

inline_for_extraction noextract
fn tensor_read
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : tlayout d) {| ctlayout l |}
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
//   (#l : tlayout d) {| ctlayout l |}
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
  (#l : tlayout d) {| ctlayout l |}
  (a : rotensor et l)
  (i : conc d)
  (#f : perm)
  (#s : chest d et)
  = tensor_read #et #r #d #l a i #f #s

(* Syntax *)
// inline_for_extraction noextract
// unfold let op_Array_Assignment
//   (#et : Type0) (#r : erased nat) (#d : shape r)
//   (#l : tlayout d) {| ctlayout l |}
//   (a : rotensor et l)
//   (i : conc d)
//   (v : et)
//   (#s : chest d et)
//   = tensor_write #et #r #d #l a i v #s

val tensor_pts_to_cell
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  ([@@@mkey] a : rotensor et l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey] i : abs d)
  (v : et)
  : slprop

[@@pulse_unfold; FStar.Tactics.Typeclasses.noinst]
instance cell_pts_to (#et : Type) (#r : nat) (#d : shape r) (#l : tlayout d)
  : has_pts_to (cell (rotensor et l) (abs d)) et
= {
  pts_to = (fun (Cell ar i) #f v -> tensor_pts_to_cell ar #f i v);
}

val tensor_pts_to_cell_eq
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l) (i : abs d) (f : perm) (v : et)
  : Lemma (Cell a i |-> Frac f v
           ==
           pts_to_cell (core a) #f (l.imap.f i) v)

instance
val is_send_across_global_tensor_cell
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l { is_global a })
  (#f : perm) (i : abs d) (v : et)
  : is_send_across gpu_of (tensor_pts_to_cell a #f i v)

ghost
fn tensor_explode
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l)
  (#f : perm)
  (#s : chest d et)
  requires
    a |-> Frac f s
  ensures
    forall+ (i : abs d).
      Cell a i |-> Frac f (acc s i)

ghost
fn tensor_implode
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l)
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
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l)
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
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l)
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
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : tlayout d) {| ctlayout l |}
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
//   (#l : tlayout d) {| ctlayout l |}
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
  (#n:nat) (d : shape n) (l : tlayout d)
  (i : natlt n) (j : natlt (d @! i))
  (idx : abs (modulo_i i d))
  : GTot (natlt l.ulen) =
    let idx' = (abs_bring_forward_bij i d).gg (j, idx) in
    l.imap.f idx'

let tlayout_slice
  (#n : nat) (#d : shape n) (l : tlayout d)
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
  (#n : erased nat) (#d : shape n) (l : tlayout d)
  {| ctlayout l |}
  (i : erased nat{i < n}) (j : erased nat{j < (d @! i)})
  {| ix : concrete_sz i |} {| jx : concrete_sz j |}
  (#r' : erased nat) (#d' : shape r')
  (#_ : reveal r' == n-1)
  (#_ : d' == modulo_i i d)
  : ctlayout #r' #d' (tlayout_slice l i j)

inline_for_extraction noextract
val sliceof
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l)
  (i : erased nat{i < r}) (j : erased nat{j < d @! i})
  : rotensor et (tlayout_slice l i j)

val lem_sliceof_core
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l)
  (i : erased nat{i < r}) (j : erased nat{j < d @! i})
  : Lemma (core (sliceof a i j) == core a)
          [SMTPat (sliceof a i j)]

val lem_is_global_iff_sliceof
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l)
  (i : natlt r) (j : natlt (d @! i))
  : Lemma (ensures is_global (sliceof a i j) <==> is_global a)
          [SMTPat (is_global (sliceof a i j))]

#push-options "--warn_error -271" // implicit subtraction in pattern, OK
val tensor_slice_cell_eq
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l)
  (i : natlt r) (j : natlt (d @! i))
  (k : abs (modulo_i i d)) (f : perm) (v : et)
  : Lemma (Cell (sliceof a i j) k |-> Frac f v
           ==
           Cell a ((abs_bring_forward_bij i d).gg (j, k)) |-> Frac f v)
           [SMTPat (Cell (sliceof a i j) k |-> Frac f v)]
#pop-options

ghost
fn tensor_extract_slice
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l)
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
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l)
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
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l)
  (i : natlt r) (j : natlt (d @! i))
  (#f : perm) (#s : chest d et)
  requires
    factored
      (sliceof a i j |-> Frac f (chest_slice i j s))
      (a |-> Frac f s)
  ensures
    a |-> Frac f s

let tlayout_bij
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (l : tlayout d1)
  : tlayout d2
  = {
      ulen = l.ulen;
      imap = inj_bij' f `Kuiper.Injection.inj_comp` l.imap;
  }

inline_for_extraction noextract
instance val ctlayout_bij
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2 { all_fit d2 })
  (f : abs d1 =~ abs d2)
  (fconc: conc d2 -> conc d1)
  (fconc_correct: (x: conc d2) -> up (fconc x) == f.gg (up x))
  (l : tlayout d1) {| c: ctlayout l |}
  : ctlayout #r2 #d2 (tlayout_bij f l)

ghost
fn tensor_apply_bij
  (#et : Type0)
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (#l : tlayout d1) {| is_full l |}
  (a : rotensor et l)
  (#fp : perm) (#m : Chest.t d1 et)
  requires
    a |-> Frac fp m
  ensures
    from_array (tlayout_bij f l) (core a) |-> Frac fp (Chest.mk d2 (fun i -> Chest.acc m (i <~| f)))

(* Rank-2 conveniences over the (natlt rows & natlt cols) index pair. *)

ghost
fn tensor_explode2
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : rotensor et l)
  (#f : perm)
  (#s : chest2 et rows cols)
  requires
    a |-> Frac f s
  ensures
    forall+ (ij : natlt rows & natlt cols).
      Cell a (idx2 (fst ij) (snd ij)) |-> Frac f (acc s (idx2 (fst ij) (snd ij)))

ghost
fn tensor_implode2
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : rotensor et l)
  (#f : perm)
  (#s : chest2 et rows cols)
  requires
    pure (SZ.fits (tlayout_ulen l))
  requires
    forall+ (ij : natlt rows & natlt cols).
      Cell a (idx2 (fst ij) (snd ij)) |-> Frac f (acc s (idx2 (fst ij) (snd ij)))
  ensures
    a |-> Frac f s

ghost
fn tensor_ilower2
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : rotensor et l)
  (#f : perm)
  (#s : chest2 et rows cols)
  requires
    a |-> Frac f s
  ensures
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (r : natlt rows) (c : natlt cols).
      Cell a (idx2 r c) |-> Frac f (acc s (idx2 r c)))

ghost
fn tensor_iraise2
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : rotensor et l)
  (#f : perm)
  (#s : chest2 et rows cols)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (r : natlt rows) (c : natlt cols).
      Cell a (idx2 r c) |-> Frac f (acc s (idx2 r c)))
  ensures
    a |-> Frac f s

// TODO: it should be possible to have just "pts_to_shareable" for any
// types that have pts_to, no? or does that not hold?
instance tensor_pts_to_shareable
  (#et : Type) (#r: erased nat) (#d: shape r) (#l: tlayout d)
  (t: rotensor et l) (s: chest d et):
  shareable (fun fr -> tensor_pts_to t #fr s) = {
  _share_n = (fun (n: pos) (#fr : perm) -> tensor_share_n t n #fr);
  _gather_n = (fun (n: pos) (#fr : perm) -> tensor_gather_n t n #fr);
}

val ref_of_tensor_cell
  (#et : Type0)
  (#r : nat) (#s : shape r) (#l : tlayout s)
  (a : rotensor et l)
  (i : abs s)
  : GTot (ref et)

inline_for_extraction noextract
fn get_ref_of_tensor_cell
  (#et : Type0)
  (#r : nat) (#s : shape r) (#l : tlayout s)
  (a : rotensor et l) {| ctlayout l |}
  (i : conc s)
  returns
    r : ref et
  ensures
    rewrites_to r (ref_of_tensor_cell a (up i))

ghost
fn tensor_cell_to_ref
  (#et : Type0)
  (#r : nat) (#s : shape r) (#l : tlayout s)
  (a : rotensor et l)
  (i : abs s)
  (#f : perm)
  (#v : erased et)
  requires
    Cell a i |-> Frac f v
  ensures
    ref_of_tensor_cell a i |-> Frac f v

ghost
fn tensor_cell_from_ref
  (#et : Type0)
  (#r : nat) (#s : shape r) (#l : tlayout s)
  (a : rotensor et l)
  (i : abs s)
  (#f : perm)
  (#v : erased et)
  requires
    ref_of_tensor_cell a i |-> Frac f v
  ensures
    Cell a i |-> Frac f v

let drop_fst
  (#r : nat)
  (#d : shape r)
  (#e : nat)
  : abs (e @| d) -> GTot (abs d)
  = function (_, i) -> i

let extended_layout
  (#r : nat) (#d : shape r)
  (l : tlayout d)
  (e : nat) // new dim
  : tlayout (e @| d)
  = {
      ulen = l.ulen;
      imap = {
        f = (fun i -> l.imap.f (drop_fst i));
        is_inj = (fun _ _ -> admit()); // clearly not true, needs layout rework
      }
    }

ghost
fn rotensor_add_dim
  (#et : Type0)
  (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : rotensor et l)
  (#f : perm) (#m : Chest.t d et)
  (e : nat) // new dim
  requires
    a |-> Frac f m
  ensures
    from_array (extended_layout l e) (core a) |-> Frac f (Chest.mk (e @| d) (function (_, i) -> acc m i))
