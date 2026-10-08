module Kuiper.Sparse.Array.PtsTo

#lang-pulse

open Kuiper
open Kuiper.Sparse.Common
open Kuiper.Sparse.Math { divup }
open FStar.Tactics.V2 { exact }
open Kuiper.Array.Vectorized

let slice_live
  (#et : Type0)
  (#l : nat)
  (a : larray et l)
  (#[full_default ()] f : perm)
  (i j : nat)
  : slprop
  = exists* s. pts_to_slice a #f i j s

let array_live_cell
  (#et : Type0)
  (#l : nat)
  (a : larray et l)
  (#[full_default ()] f : perm)
  (i : natlt l)
  : slprop
= exists* v. pts_to_cell a #f i v

unfold
let pts_to_vec
  (#a:Type u#0) {| sized a, has_vec_cpy a |}
  (#sz:nat)
  ([@@@mkey] x:larray a sz)
  (#[full_default ()] f : perm)
  ([@@@mkey] i : nat)
  (v : seq a)
: slprop
= pts_to_slice x #f i (i + chunk a) v

unfold
let pts_to_vec'
  (#a:Type) {| sized a, has_vec_cpy a |}
  (#sz:nat)
  ([@@@mkey] x:larray a sz)
  (#[full_default ()] f : perm)
  ([@@@mkey] i : nat)
  (v : seq a)
  (k : natle (len v - chunk a))
: slprop
= pts_to_vec x #f i (Seq.slice v k (k + chunk a))

let live_vec
  (#a:Type) {| sized a, has_vec_cpy a |}
  (#l : nat)
  (x :larray a l)
  (#[full_default ()] f : perm)
  (i : nat)
: slprop
= exists* v. pts_to_vec x #f i v

(* Thread sharing *)

let thread_slice_pts_to
  (#et : Type0)
  (#n : nat)
  (a : larray et n)
  (i j : natle n { i <= j })
  (#m : nat)
  (s : lseq et m)
  (k : natle (m - (j - i)))
  (nthr : nat) (tid : natlt nthr)
: slprop
=
  forall+ (x : natlt ((j - i - tid) `divup` nthr)).
    pts_to_cell a (i + x * nthr + tid <: nat) (s @! k + x * nthr + tid)

let thread_slice_pts_to_value
  (#et : Type0)
  (#n : nat)
  (a : larray et n)
  (i j : natle n { i <= j })
  (v : et)
  (nthr : nat) (tid : natlt nthr)
: slprop
=
  forall+ (x : natlt ((j - i - tid) `divup` nthr)).
    pts_to_cell a (i + x * nthr + tid) v

let thread_slice_live
  (#et : Type0)
  (#n : nat)
  (a : larray et n)
  (i j : natle n {i <= j})
  (nthr : nat) (tid : natlt nthr)
: slprop
=
  forall+ (k : natlt ((j - i - tid) `divup` nthr)).
    array_live_cell a (i + k * nthr + tid)

(* Vector thread sharing *)

let thread_pts_to_chunks
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n : nat)
  ([@@@mkey] x : larray et n)
  (#m : nat)
  (s : lseq et m)
  (i nthr : nat)
  (tid : natlt nthr)
: Pure slprop
  (requires (nthr * chunk et) /? n /\ i + n <= m)
  (ensures fun _ -> true)
=
  forall+ (k : natlt (n / (nthr * chunk et))).
    pts_to_vec' x ((k * nthr + tid) * chunk et)
      s (i + (k * nthr + tid) * chunk et)

let thread_live_chunks
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n : nat)
  ([@@@mkey] x : larray et n)
  (nthr : nat)
  (tid : natlt nthr)
: Pure slprop
  (requires (nthr * chunk et) /? n)
  (ensures fun _ -> true)
=
  forall+ (k : natlt (n / (nthr * chunk et))).
    live_vec x ((k * nthr + tid) * chunk et)

(* Helpers *)

ghost
fn thread_slice_share
  (#et : Type0)
  (#n : nat)
  (x : larray et n)
  (i j : natle n { i <= j })
  (#m : nat)
  (nthr : pos)
  requires slice_live x i j
  ensures  forall+ (tid : natlt nthr). thread_slice_live x i j nthr tid
  ensures  array_exists x

ghost
fn thread_slice_gather
  (#et : Type0)
  (#n : nat)
  (x : larray et n)
  (i j : natle n { i <= j })
  (#m : nat)
  (s : lseq et m)
  (k : natle (m - (j - i)))
  (nthr : pos)
  requires array_exists x
  requires forall+ (tid : natlt nthr). thread_slice_pts_to x i j s k nthr tid
  ensures  pts_to_slice x i j (Seq.slice s k (k + (j - i)))

ghost
fn thread_slice_gather_value
  (#et : Type0)
  (#n : nat)
  (x : larray et n)
  (i j : natle n { i <= j })
  (v : et)
  (nthr : pos)
  requires array_exists x
  requires forall+ (tid : natlt nthr). thread_slice_pts_to_value x i j v nthr tid
  ensures  pts_to_slice x i j (Seq.create (j - i) v)

ghost
fn thread_share_chunks
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n : nat)
  (x : larray et n)
  (nthr : pos)
  (#_: squash ((nthr * chunk et) /? n))
  requires live x
  ensures forall+ (tid : natlt nthr). thread_live_chunks x nthr tid
  ensures array_exists x