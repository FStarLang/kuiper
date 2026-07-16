module Kuiper.Tensor.Layout.Slice
#lang-pulse

open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor
open Pulse.Lib.Trade

module SZ = Kuiper.SizeT
module T = FStar.Tactics.V2

(* Generic extraction of slices *)

// Move some of this to Tensor.Layout.
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
  (a : tensor et l)
  (i : erased nat{i < r}) (j : erased nat{j < d @! i})
  : tensor et (tlayout_slice l i j)

val lem_sliceof_core
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : erased nat{i < r}) (j : erased nat{j < d @! i})
  : Lemma (core (sliceof a i j) == core a)
          [SMTPat (sliceof a i j)]

val lem_is_global_iff_sliceof
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : natlt r) (j : natlt (d @! i))
  : Lemma (ensures is_global (sliceof a i j) <==> is_global a)
          [SMTPat (is_global (sliceof a i j))]

#push-options "--warn_error -271" // implicit subtraction in pattern, OK
val tensor_slice_cell_eq
  (#et : Type0) (#r : nat) (#d : shape r)
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
  (#et : Type0) (#r : nat) (#d : shape r)
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
  (#et : Type0) (#r : nat) (#d : shape r)
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
  (#et : Type0) (#r : nat) (#d : shape r)
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