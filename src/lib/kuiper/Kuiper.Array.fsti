module Kuiper.Array

#lang-pulse

open Pulse.Lib.Vec
open Pulse
open Pulse.Lib.BigStar
open FStar.Tactics.V2
open FStar.Seq
open Kuiper.Base
open Kuiper.Sized

module SZ = FStar.SizeT

val gpu_array (a:Type u#0) (sz:nat) : Type u#0

(* x is the base pointer, this gives permission in [i,j) *)
val gpu_pts_to_slice
  (#a:Type u#0)
  (#sz:nat)
  ([@@@mkey] x:gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  (i : nat)
  (j : nat)
  (v : seq a)
: slprop

unfold
let gpu_pts_to_array
  (#a:Type u#0)
  (#sz:nat)
  ([@@@mkey] x:gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  (v : seq a)
: slprop
=
  gpu_pts_to_slice x #f 0 sz v

unfold
instance has_pts_to_gpu_arr (a:Type) (sz : _) : has_pts_to (gpu_array a sz) (Seq.seq a) = {
  pts_to = gpu_pts_to_array;
}

ghost
fn gpu_pts_to_slice_ref
  (#a:Type u#0)
  (#sz:nat)
  (#f : perm)
  (x:gpu_array a sz)
  (i:nat) (j:nat)
  (#v : seq a)
  preserves gpu_pts_to_slice x #f i j v
  requires emp
  ensures  pure (i <= j /\ j <= sz /\ Seq.length v == (j-i) /\ SZ.fits (Seq.length v))

ghost
fn gpu_pts_to_ref
  (#a:Type u#0)
  (#sz:nat)
  (#f : perm)
  (x:gpu_array a sz)
  (#v : seq a)
  preserves pts_to x #f v
  requires emp
  ensures  pure (Seq.length v == sz /\ SZ.fits sz)

noextract
fn gpu_array_alloc
  (#a : Type u#0)
  {| sized a |}
  (sz : SZ.t)
  preserves cpu
  requires emp
  returns  x : gpu_array a (SZ.v sz)
  ensures
    exists* (s:seq a). (x |-> s) ** pure (Seq.length s == sz)

fn gpu_array_free
  (#a:Type u#0)
  (#sz:erased nat)
  (r : gpu_array a sz)
  (#v : erased (seq a))
  preserves cpu
  requires r |-> v
  ensures  emp

[@@noextract_to "krml"]
atomic
fn gpu_array_read
  (#a : Type u#0)
  (#sz : erased nat)
  (#i  : erased nat)
  (#j  : erased nat{i <= j /\ j <= sz})
  (r:gpu_array a sz)
  (#f:perm)
  (idx : SZ.t {i <= SZ.v idx /\ SZ.v idx < j})
  (#s : erased (seq a))
  preserves gpu ** gpu_pts_to_slice #a #sz r #f i j s
  requires emp
  returns  x:a
  ensures  pure (i <= j /\ j <= sz /\ Seq.length s == (j-i) /\
                  x == Seq.index s (SZ.v idx - i))

[@@noextract_to "krml"]
fn gpu_array_write
  (#a:Type u#0)
  (#sz: erased nat)
  (#i: erased nat)
  (#j: erased nat{i <= j /\ j <= sz})
  (r:gpu_array a sz)
  (idx : SZ.t{i <= SZ.v idx /\ SZ.v idx < j})
  (v : a)
  (#s : erased (seq a))
  preserves gpu
  requires gpu_pts_to_slice #a #sz r #1.0R i j s
  ensures  (exists* (s':seq a). gpu_pts_to_slice #a #sz r #1.0R i j s' **
              pure (i <= j /\ j <= sz /\ Seq.length s == (j-i) /\
                    s' == Seq.upd s (SZ.v idx - i) v))

fn gpu_memcpy_host_to_device
  (#a:Type u#0)
  {| sized a |}
  (#dst_sz : erased nat)
  (dst_garr : gpu_array a dst_sz)
  (dst_off : SZ.t)
  (#src_sz : erased nat)
  (src_arr : vec a)
  (src_off : SZ.t)
  (cnt : SZ.t)
  (#f : perm)
  (#v : erased (seq a))
  (#gv : erased (seq a))
  preserves
    cpu **
    pts_to src_arr #f v
  requires
    (dst_garr |-> gv) **
    pure (dst_off + cnt <= dst_sz /\
          src_off + cnt <= src_sz
    )
  ensures
    (dst_garr |-> v) ** // wrong
    pure (Seq.length v == reveal dst_sz)

fn gpu_memcpy_device_to_host
  (#a:Type u#0)
  {| sized a |}
  (#sz : erased nat)
  (dst_arr : vec a)
  (src_garr : gpu_array a sz)
  (cnt : SZ.t)
  (#f : perm)
  (#v : erased (seq a))
  (#gv : erased (seq a))
  preserves
    cpu **
    pts_to src_garr #f gv
  requires
    (dst_arr |-> v) **
    pure (SZ.v cnt == sz /\ (Pulse.Lib.Vec.length dst_arr == sz \/ Seq.length v == reveal sz))
  ensures
    (dst_arr |-> gv) **
    pure (Seq.length gv == reveal sz)

fn gpu_memcpy_device_to_device
  (#a:Type u#0)
  {| sized a |}
  (#sz : erased nat)
  (dst_arr : gpu_array a sz)
  (src_garr : gpu_array a sz)
  (cnt : SZ.t)
  (#f : perm)
  (#v : erased (seq a))
  (#gv : erased (seq a))
  preserves
    cpu **
    pts_to src_garr #f gv
  requires
    (dst_arr |-> v) **
    pure (SZ.v cnt == sz /\ (Seq.length gv == sz \/ Seq.length v == sz))
  ensures
    (dst_arr |-> gv) **
    pure (Seq.length gv == reveal sz)


(* Not making this unfold as it appears under bigstars. Maybe
pulse should only do weak unfolding. *)
let gpu_pts_to_array1
  (#a:Type0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  (i:nat)
: slprop =
  exists* s. gpu_pts_to_slice arr i (i+1) s

ghost
fn gpu_array_slice_1
  (#[exact (`0)] uid: int) (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#v : erased (seq a) { Seq.length v == sz })
  requires pts_to arr #f v
  ensures  bigstar #uid 0 sz (fun i -> gpu_pts_to_slice arr #f i (i+1) seq![Seq.index v i])

ghost
fn gpu_array_unslice_1
  (#uid: int) (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#v : erased (seq a) { Seq.length v == sz })
  requires bigstar #uid 0 sz (fun i -> gpu_pts_to_slice arr #f i (i+1) seq![Seq.index v i])
  ensures  pts_to arr #f v

ghost
fn gpu_array_slice_1_underspec
  (#[exact (`0)] uid: int) (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#v : erased (seq a))
  requires pts_to arr #f v
  ensures  bigstar #uid 0 sz (gpu_pts_to_array1 arr #f)

ghost
fn gpu_array_unslice_1_underspec
  (#uid: int) (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  requires bigstar #uid 0 sz (gpu_pts_to_array1 arr #f)
  ensures exists* v. pts_to arr #f v

ghost
fn gpu_slice_concat
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  (#s1 #s2: erased (seq a))
  (i n m:nat)
  requires gpu_pts_to_slice arr #f i n s1 ** gpu_pts_to_slice arr #f n m s2
  ensures  gpu_pts_to_slice arr #f i m (Seq.append s1 s2)

ghost
fn gpu_slice_slice_1_underspec
  (#uid: int) (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#v : erased (seq a))
  (i n:nat)
  (m: nat { i <= m /\ m <= n })
  requires gpu_pts_to_slice arr #f i n v
  ensures
    bigstar #uid 0 (m - i) (fun x -> gpu_pts_to_array1 arr #f (x + i)) **
    gpu_pts_to_slice arr #f m n v

ghost
fn gpu_slice_unslice_1_underspec
  (#uid: int) (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (i n:nat)
  (m: nat { i <= m /\ m <= n })
  requires
    bigstar #uid 0 (m - i) (fun x -> gpu_pts_to_array1 arr #f (x + i)) ** (exists* v. gpu_pts_to_slice arr #f m n v)
  ensures
    exists* v.
      gpu_pts_to_slice arr #f i n v

ghost
fn gpu_slice_empty_elim
  (#a:Type u#0) (#sz:nat)
  (arr : gpu_array a sz)
  (i : nat)
  requires gpu_pts_to_slice arr #'f i i 'v
  ensures  emp

ghost
fn gpu_slice_share
  (#uid: int) (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (m n:nat)
  (k: nat { k > 0 })
  requires gpu_pts_to_slice arr #'f m n 'v
  ensures
    bigstar #uid 0 k (fun x -> gpu_pts_to_slice arr #('f /. Real.of_int k) m n 'v)

ghost
fn gpu_slice_gather
  (#uid: int) (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (m n:nat)
  (k: nat { k > 0 })
  (#f : perm) // FIXME: if we use 'f, it gets type 'real' instead of 'perm'
  requires
    bigstar #uid 0 k (fun x -> gpu_pts_to_slice arr #(f /. Real.of_int k) m n 'v)
  ensures gpu_pts_to_slice arr #f m n 'v

ghost
fn gpu_slice_share_underspec
  (#uid : int) (#a : Type u#0)
  (#sz : nat)
  (arr : gpu_array a sz)
  (m n : nat)
  (k : nat { k > 0 })
  requires gpu_pts_to_slice arr #'f m n 'v
  ensures bigstar #uid 0 k (fun x -> exists* v. gpu_pts_to_slice arr #('f /. Real.of_int k) m n v)

ghost
fn gpu_slice_gather_underspec
  (#uid : int) (#a : Type u#0)
  (#sz : nat)
  (arr : gpu_array a sz)
  (#f : perm) // FIXME: if we use 'f, it gets type 'real' instead of 'perm'
  (m n : nat)
  (k : nat { k > 0 })
  requires bigstar #uid 0 k (fun x -> exists* v. gpu_pts_to_slice arr #(f /. Real.of_int k) m n v)
  ensures
    exists* v.
      gpu_pts_to_slice arr #f m n v
