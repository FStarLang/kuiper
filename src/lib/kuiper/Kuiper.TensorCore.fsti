module Kuiper.TensorCore

#lang-pulse

open Kuiper
open Pulse.Lib.Trade
open Kuiper.Tensor
open Kuiper.Array2.Strided
include Kuiper.TensorCore.Base
open Kuiper.Seq.Common { (@!) }

inline_for_extraction noextract
fn mma_loadA
  (#et : Type)
  (#m #n #k : erased nat)
  (fr : fragment et FragA m n k FragLRM)
  (#l : layout2 m k) {| strided_row_major l |}
  (gm : array2 et l)
  (#f : perm)
  (#m0 : chest2 et m k)
  (#f0 : erased (value_for et FragA m n k))
  preserves
    gm |-> Frac f m0
  requires
    fr |-> f0
  ensures
    fr |-> m0

inline_for_extraction noextract
fn mma_loadA_cm
  (#et : Type)
  (#m #n #k : erased nat)
  (fr : fragment et FragA m n k FragLCM)
  (#l : layout2 m k) {| strided_col_major l |}
  (gm : array2 et l)
  (#f : perm)
  (#m0 : chest2 et m k)
  (#f0 : erased (value_for et FragA m n k))
  preserves
    gm |-> Frac f m0
  requires
    fr |-> f0
  ensures
    fr |-> m0

inline_for_extraction noextract
fn mma_loadB
  (#et : Type)
  (#m #n #k : erased nat)
  (fr : fragment et FragB m n k FragLRM)
  (#l : layout2 k n) {| strided_row_major l |}
  (gm : array2 et l)
  (#f : perm)
  (#m0 : chest2 et k n)
  (#f0 : erased (value_for et FragB m n k))
  preserves
    gm |-> Frac f m0
  requires
    fr |-> f0
  ensures
    fr |-> m0

inline_for_extraction noextract
fn mma_loadB_cm
  (#et : Type)
  (#m #n #k : erased nat)
  (fr : fragment et FragB m n k FragLCM)
  (#l : layout2 k n) {| strided_col_major l |}
  (gm : array2 et l)
  (#f : perm)
  (#m0 : chest2 et k n)
  (#f0 : erased (value_for et FragB m n k))
  preserves
    gm |-> Frac f m0
  requires
    fr |-> f0
  ensures
    fr |-> m0

inline_for_extraction noextract
fn mma_store
  (#et : Type)
  (#m #n #k : erased nat)
  (fr : fragment et FragAcc m n k FragLAcc)
  (#l : layout2 m n) {| strided_row_major l |}
  (gm : array2 et l)
  (#f0 : erased (value_for et FragAcc m n k))
  (#m0 : chest2 et m n)
  preserves
    fr |-> f0
  requires
    gm |-> Frac (1.0R /. warp_size) m0
  ensures
    gm |-> Frac (1.0R /. warp_size) f0

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
