module GPU.Array

#lang-pulse

open Pulse
open Pulse.Lib.BigStar
open FStar.Tactics.V2
open FStar.Seq
open GPU.Base
open GPU.Sized

module A = Pulse.Lib.Array
module SZ = FStar.SizeT

val gpu_array (a:Type u#0) (sz:nat) : Type u#0

val gpu_pts_to_array_slice
  (#a:Type u#0)
  (#sz:nat)
  (x:gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  (i:nat) (j:nat)
  (v : seq a)
: slprop

let gpu_pts_to_array
  (#a:Type u#0)
  (#sz:nat)
  (x:gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  (v : seq a)
: slprop
=
  gpu_pts_to_array_slice x #f 0 sz v

val gpu_pts_to_slice_ref
  (#a:Type u#0)
  (#sz:nat)
  (#[exact (`1.0R)] f : perm)
  (x:gpu_array a sz)
  (i:nat) (j:nat)
  (#v : seq a)
  : stt_ghost unit emp_inames
      (gpu_pts_to_array_slice x #f i j v)
      (fun _ -> gpu_pts_to_array_slice x #f i j v ** pure (i <= j /\ j <= sz /\ Seq.length v == (j-i)))

val gpu_pts_to_ref
  (#a:Type u#0)
  (#sz:nat)
  (#[exact (`1.0R)] f : perm)
  (x:gpu_array a sz)
  (#v : seq a)
  : stt_ghost unit emp_inames
      (gpu_pts_to_array x #f v)
      (fun _ -> gpu_pts_to_array x #f v ** pure (Seq.length v == sz))

noextract
fn gpu_array_alloc
  (#a : Type u#0)
  {| sized a |}
  (sz : SZ.t)
  requires cpu
  returns  x : gpu_array a (SZ.v sz)
  ensures  cpu **
            (exists* (s:seq a). gpu_pts_to_array x #1.0R s)

fn gpu_array_free
  (#a:Type u#0)
  (#sz:erased nat)
  (r : gpu_array a sz)
  (#v : erased (seq a))
  requires cpu ** gpu_pts_to_array r #1.0R v
  ensures  cpu

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
  requires gpu ** gpu_pts_to_array_slice #a #sz r #f i j s
  returns  x:a
  ensures  gpu ** gpu_pts_to_array_slice #a #sz r #f i j s **
            pure (i <= j /\ j <= sz /\ Seq.length s == (j-i) /\
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
  requires gpu ** gpu_pts_to_array_slice #a #sz r #1.0R i j s
  ensures  gpu **
            (exists* (s':seq a). gpu_pts_to_array_slice #a #sz r #1.0R i j s' **
              pure (i <= j /\ j <= sz /\ Seq.length s == (j-i) /\
                    s' == Seq.upd s (SZ.v idx - i) v))

fn gpu_memcpy_host_to_device
  (#a:Type u#0)
  {| sized a |}
  (#sz : erased nat)
  (arr : array a)
  (garr : gpu_array a sz)
  (cnt : SZ.t)
  (#f : perm)
  (#v : erased (seq a))
  (#gv : erased (seq a))
  requires cpu ** A.pts_to arr #f v ** gpu_pts_to_array garr #1.0R gv ** pure (SZ.v cnt == sz)
  ensures  cpu ** A.pts_to arr #f v ** gpu_pts_to_array garr #1.0R v
        ** pure (Seq.length v == reveal sz)

fn gpu_memcpy_device_to_host
  (#a:Type u#0)
  {| sized a |}
  (#sz : erased nat)
  (arr : array a)
  (garr : gpu_array a sz)
  (cnt : SZ.t)
  (#f : perm)
  (#v : erased (seq a))
  (#gv : erased (seq a))
  requires cpu ** A.pts_to arr #f v ** gpu_pts_to_array garr #1.0R gv ** pure (SZ.v cnt == sz)
  ensures  cpu ** A.pts_to arr #f gv ** gpu_pts_to_array garr #1.0R gv
        ** pure (Seq.length gv == reveal sz)

let gpu_pts_to_array1
  (#a:Type0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  (i:nat)
: slprop =
  exists* s. gpu_pts_to_array_slice arr i (i+1) s

val gpu_array_slice_1
  (#[exact (`0)] uid: int) (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#v : erased (seq a) { Seq.length v == sz })
: stt_ghost
    unit
    emp_inames
    (gpu_pts_to_array arr #f v)
    (fun _ -> bigstar #uid 0 sz (fun i -> gpu_pts_to_array_slice arr #f i (i+1) (Seq.cons (Seq.index v i) Seq.empty)))

val gpu_array_unslice_1
  (#uid: int) (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#v : erased (seq a) { Seq.length v == sz })
: stt_ghost
    unit
    emp_inames
    (bigstar #uid 0 sz (fun i -> gpu_pts_to_array_slice arr #f i (i+1) (Seq.cons (Seq.index v i) Seq.empty)))
    (fun _ -> gpu_pts_to_array arr #f v)

val gpu_array_slice_1_underspec
  (#[exact (`0)] uid: int) (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#v : erased (seq a))
: stt_ghost
    unit
    emp_inames
    (gpu_pts_to_array arr #f v)
    (fun _ -> bigstar #uid 0 sz (gpu_pts_to_array1 arr #f))

val gpu_array_unslice_1_underspec
  (#uid: int) (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
: stt_ghost
    unit
    emp_inames
    (bigstar #uid 0 sz (gpu_pts_to_array1 arr #f))
    (fun _ -> exists* v. gpu_pts_to_array arr #f v)

val gpu_slice_concat
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  (#s1 #s2: erased (seq a))
  (i n m:nat)
  : stt_ghost unit emp_inames
      (gpu_pts_to_array_slice arr #f i n s1 ** gpu_pts_to_array_slice arr #f n m s2)
      (fun _ -> gpu_pts_to_array_slice arr #f i m (Seq.append s1 s2))

ghost
fn gpu_slice_slice_1_underspec
  (#uid: int) (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#v : erased (seq a))
  (i n:nat)
  (m: nat { i <= m /\ m <= n })
  requires gpu_pts_to_array_slice arr #f i n v
  ensures bigstar #uid 0 (m - i) (fun x -> gpu_pts_to_array1 arr #f (x + i)) ** gpu_pts_to_array_slice arr #f m n v

ghost
fn gpu_slice_unslice_1_underspec
  (#uid: int) (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (i n:nat)
  (m: nat { i <= m /\ m <= n })
  requires bigstar #uid 0 (m - i) (fun x -> gpu_pts_to_array1 arr #f (x + i)) ** (exists* v. gpu_pts_to_array_slice arr #f m n v)
  ensures exists* v. gpu_pts_to_array_slice arr #f i n v

ghost
fn gpu_slice_empty_elim
  (#a:Type u#0) (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (i : nat)
  requires exists* v. gpu_pts_to_array_slice arr #f i i v
  ensures  emp

ghost
fn gpu_slice_share
  (#uid: int) (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#v : erased (seq a))
  (m n:nat)
  (k: nat { k > 0 })
  requires gpu_pts_to_array_slice arr #f m n v
  ensures bigstar #uid 0 k (fun x -> gpu_pts_to_array_slice arr #(f /. Real.of_int k) m n v)

ghost
fn gpu_slice_gather
  (#uid: int) (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#v : erased (seq a))
  (m n:nat)
  (k: nat { k > 0 })
  requires bigstar #uid 0 k (fun x -> gpu_pts_to_array_slice arr #(f /. Real.of_int k) m n v)
  ensures gpu_pts_to_array_slice arr #f m n v

ghost
fn gpu_slice_share_underspec
  (#uid: int) (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (m n:nat)
  (k: nat { k > 0 })
  requires exists* v. gpu_pts_to_array_slice arr #f m n v
  ensures bigstar #uid 0 k (fun x -> exists* v. gpu_pts_to_array_slice arr #(f /. Real.of_int k) m n v)

ghost
fn gpu_slice_gather_underspec
  (#uid: int) (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (m n:nat)
  (k: nat { k > 0 })
  requires bigstar #uid 0 k (fun x -> exists* v. gpu_pts_to_array_slice arr #(f /. Real.of_int k) m n v)
  ensures exists* v. gpu_pts_to_array_slice arr #f m n v
