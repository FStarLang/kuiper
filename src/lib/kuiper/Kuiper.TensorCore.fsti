module Kuiper.TensorCore

#lang-pulse

open Kuiper
open Pulse.Lib.Trade
include Kuiper.TensorCore.Base

ghost
fn array_fragment_pts_to_ref
  (#et : Type0)
  (#knd : fragment_kind)
  (#m #n #k : nat)
  (#l : fragment_layout)
  ([@@@mkey] farr: array (fragment et knd m n k l))
  (#f : perm)
  (#ems : seq (value_for et knd m n k))
  preserves array_fragment_pts_to farr #f ems
  ensures   pure (Seq.length ems == Pulse.Lib.Array.length farr)

ghost
fn array_fragment_extract
  (#et:Type0)
  (#knd : fragment_kind)
  (#m #n #k : nat)
  (#l : fragment_layout)
  (farr: array (fragment et knd m n k l))
  (#f : perm)
  (#ems : seq (value_for et knd m n k))
  (i : natlt (Seq.length ems))
  requires
    array_fragment_pts_to farr #f ems
  ensures
    exists* (s : (lseq (fragment et knd m n k l) (Seq.length ems))).
      farr |-> Frac f s **
      (s @! i) |-> (ems @! i) **
      (forall* (em' : value_for et knd m n k).
        (farr |-> Frac f s **
         (s @! i) |-> em') @==>
          array_fragment_pts_to farr #f (Seq.upd ems i em'))

ghost
fn array_fragment_extract_ro
  (#et:Type0)
  (#knd : fragment_kind)
  (#m #n #k : nat)
  (#l : fragment_layout)
  (farr: array (fragment et knd m n k l))
  (#ems : seq (value_for et knd m n k))
  (#f : perm)
  (i : natlt (Seq.length ems))
  requires
    array_fragment_pts_to farr #f ems
  ensures
    exists* (s : (lseq (fragment et knd m n k l) (Seq.length ems))).
      factored
        (farr |-> Frac f s ** (s @! i) |-> Frac f (ems @! i))
        (array_fragment_pts_to farr #f ems)
