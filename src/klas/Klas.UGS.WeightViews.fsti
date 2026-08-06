module Klas.UGS.WeightViews
#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.View
open Kuiper.VArray

module SZ = Kuiper.SizeT

let pair_group_n : pos = 64

let gate_col_of (c : nat) : nat =
  128 * (c / pair_group_n) + (c % pair_group_n)

let up_col_of (c : nat) : nat =
  gate_col_of c + pair_group_n

val packed_cols_bound (n : nat) (c : natlt n)
  : Lemma (requires n % pair_group_n == 0)
          (ensures gate_col_of c < 2 * n /\ up_col_of c < 2 * n)

val gate_col_injective (x y : nat)
  : Lemma (requires gate_col_of x == gate_col_of y)
          (ensures x == y)

val up_col_injective (x y : nat)
  : Lemma (requires up_col_of x == up_col_of y)
          (ensures x == y)

val gate_up_disjoint (x y : nat)
  : Lemma (ensures gate_col_of x =!= up_col_of y)

[@@erasable]
val gate_layout (k n : nat { n % pair_group_n == 0 }) : layout2 k n
[@@erasable]
val up_layout (k n : nat { n % pair_group_n == 0 }) : layout2 k n

val gate_layout_index
  (#k #n : nat { n % pair_group_n == 0 })
  (r : natlt k) (c : natlt n)
  : Lemma (
      (gate_layout k n).imap.f (idx2 r c) ==
      r * (2 * n) + gate_col_of c)

val up_layout_index
  (#k #n : nat { n % pair_group_n == 0 })
  (r : natlt k) (c : natlt n)
  : Lemma (
      (up_layout k n).imap.f (idx2 r c) ==
      r * (2 * n) + up_col_of c)

inline_for_extraction noextract
instance val c_gate_layout
  (k : erased nat { SZ.fits k })
  (n : sz { n % pair_group_n == 0 /\
            SZ.fits (2 * n) /\
            SZ.fits (k * (2 * n)) })
  : ctlayout (gate_layout k n)

inline_for_extraction noextract
instance val c_up_layout
  (k : erased nat { SZ.fits k })
  (n : sz { n % pair_group_n == 0 /\
            SZ.fits (2 * n) /\
            SZ.fits (k * (2 * n)) })
  : ctlayout (up_layout k n)

[@@erasable]
val gate_view (k n : nat { n % pair_group_n == 0 })
  : aview half (chest2 half k n)

[@@erasable]
val up_view (k n : nat { n % pair_group_n == 0 })
  : aview half (chest2 half k n)

val gate_up_no_overlap (k n : nat { n % pair_group_n == 0 })
  : Lemma (
      no_overlap
        (gate_view k n).iview.step.imap.f
        (up_view k n).iview.step.imap.f)

[@@erasable]
val packed_aview (k n : nat { n % pair_group_n == 0 })
  : aview half (chest2 half k n & chest2 half k n)

inline_for_extraction noextract
fn split
  (#k #n : nat { n % pair_group_n == 0 })
  (a : varray (packed_aview k n))
  (#f : perm)
  (#vGate : erased (chest2 half k n))
  (#vUp : erased (chest2 half k n))
  requires
    a |-> Frac f (reveal vGate, reveal vUp)
  returns
    p : array2 half (gate_layout k n) & array2 half (up_layout k n)
  ensures
    fst p |-> Frac f (reveal vGate) **
    snd p |-> Frac f (reveal vUp) **
    pure (Kuiper.Tensor.core (fst p) == Kuiper.VArray.core a) **
    pure (Kuiper.Tensor.core (snd p) == Kuiper.VArray.core a)

inline_for_extraction noextract
fn join
  (#k #n : nat { n % pair_group_n == 0 })
  (gate : array2 half (gate_layout k n))
  (up : array2 half (up_layout k n))
  (#f : perm)
  (#vGate : erased (chest2 half k n))
  (#vUp : erased (chest2 half k n))
  requires
    pure (Kuiper.Tensor.core gate == Kuiper.Tensor.core up) **
    gate |-> Frac f (reveal vGate) **
    up |-> Frac f (reveal vUp)
  returns
    a : varray (packed_aview k n)
  ensures
    a |-> Frac f (reveal vGate, reveal vUp) **
    pure (Kuiper.VArray.core a == Kuiper.Tensor.core gate)
