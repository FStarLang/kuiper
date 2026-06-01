module Kuiper.Array.Derived

#lang-pulse

open Pulse.Lib.Vec
open Pulse.Lib.WithPure
open Pulse
open FStar.Tactics.V2
open FStar.Seq
open Kuiper.Base
open Kuiper.Sized
open Kuiper.SizeT
open Kuiper.Seq.Common
open Kuiper.Common
open Kuiper.ForEvery
open Kuiper.Divides { (/?+) }

module SZ = FStar.SizeT

open Kuiper.Array.Core

ghost
fn gpu_pts_to_ref
  (#a:Type u#0)
  (#f : perm)
  (x : array a)
  (#v : seq a)
  preserves
    x |-> Frac f v
  ensures
    pure (Seq.length v == Pulse.Lib.Array.length x /\ SZ.fits (Pulse.Lib.Array.length x))

ghost
fn gpu_pts_to_ref_located
  (#a:Type u#0)
  (#f : perm)
  (x : array a)
  (#v : seq a)
  (#l : loc_id)
  preserves
    on l (x |-> Frac f v)
  ensures
    pure (Seq.length v == Pulse.Lib.Array.length x /\ SZ.fits (Pulse.Lib.Array.length x))

(* Not making this unfold as it appears under forall+. Maybe
pulse should only do weak unfolding. *)
let gpu_pts_to_array1
  (#a:Type0)
  ([@@@mkey]arr : array a)
  (#[exact (`1.0R)] f : perm)
  ([@@@mkey]i:nat)
: slprop =
  exists* s. pts_to_slice arr #f i (i+1) s

ghost
fn gpu_slice_split'
  (#a:Type u#0)
  (#sz:nat)
  (arr : larray a sz)
  (#[exact (`1.0R)] f : perm)
  (#s : erased (seq a))
  (i n : nat)
  (#_ : squash (0 <= n-i /\ n-i < length s))
  (m:nat)
  requires pts_to_slice arr #f i m s ** pure (n <= m)
  ensures  pts_to_slice arr #f i n (seq_take (n-i) s) ** pts_to_slice arr #f n m (seq_drop (n-i) s)

ghost
fn gpu_array_unslice_1'
  (#a:Type u#0)
  (#sz:nat)
  (arr : larray a sz)
  (#f : perm)
  requires forall+ (i: natlt sz). exists* v. pts_to_cell arr #f i v
  ensures  exists* v. pts_to arr #f v
