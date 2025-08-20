module Kuiper.Array

#lang-pulse

open Pulse.Lib.Vec
open Pulse
open Pulse.Lib.BigStar
open FStar.Tactics.V2
open FStar.Seq
open Kuiper.Base
open Kuiper.Sized
open Kuiper.SizeT
open Kuiper.Seq.Common

module SZ = FStar.SizeT

new
val gpu_array (a : Type u#0) (sz : SZ.t) : Type u#0

(* FIXME: I think having nat here, which forces to use erased nat
   in concrete functions, hurts Pulse inference a lot. Try to make all
   these ints. *)

(* x is the base pointer, this gives permission in [i,j) *)
val gpu_pts_to_slice
  (#a:Type u#0)
  (#sz:SZ.t)
  ([@@@mkey] x:gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  ([@@@mkey] i : SZ.t)
  (j : SZ.t)
  (v : seq a)
: slprop

unfold
let gpu_pts_to_cell
  (#a:Type u#0)
  (#sz:SZ.t)
  ([@@@mkey] x:gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  ([@@@mkey] i : SZ.t{SZ.fits (SZ.v i + 1)})
  (v : a)
: slprop
=
  gpu_pts_to_slice x #f i (i `SZ.add` 1sz) seq![v]

unfold
let gpu_pts_to_array
  (#a:Type u#0)
  (#sz:SZ.t)
  ([@@@mkey] x:gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  (v : seq a)
: slprop
=
  gpu_pts_to_slice x #f 0sz sz v

unfold
instance has_pts_to_gpu_arr (a:Type) (sz : _) :
  has_pts_to (gpu_array a sz) (Seq.seq a) =
{
  pts_to = gpu_pts_to_array;
}

ghost
fn gpu_pts_to_slice_ref
  (#a:Type u#0)
  (#sz:SZ.t)
  (#f : perm)
  (x:gpu_array a sz)
  (i:SZ.t) (j:SZ.t)
  (#v : seq a)
  preserves gpu_pts_to_slice x #f i j v
  requires emp
  ensures  pure (i <= j /\ j <= sz /\ Seq.length v == (j-i) /\ SZ.fits sz)

ghost
fn gpu_pts_to_ref
  (#a:Type u#0)
  (#sz:SZ.t)
  (#f : perm)
  (x:gpu_array a sz)
  (#v : seq a)
  preserves x |-> Frac f v
  ensures  pure (Seq.length v == sz /\ SZ.fits sz)

noextract
fn gpu_array_alloc
  (#a : Type u#0)
  {| sized a |}
  (sz : SZ.t)
  preserves cpu
  returns   x : gpu_array a sz
  ensures
    exists* (s:seq a). (x |-> s) ** pure (Seq.length s == sz)

fn gpu_array_free
  (#a:Type u#0)
  (#sz:erased SZ.t)
  (r : gpu_array a sz)
  (#v : erased (seq a))
  preserves cpu
  requires r |-> v
  ensures  emp

[@@noextract_to "krml"]
atomic
fn gpu_array_read
  (#a : Type u#0)
  (#sz : erased SZ.t)
  (#i  : erased SZ.t)
  (#j  : erased SZ.t)
  (r:gpu_array a sz)
  (#f:perm)
  (idx : SZ.t)
  (#s : erased (seq a))
  preserves gpu
  preserves gpu_pts_to_slice #a #sz r #f i j s
  requires pure (SZ.v i <= SZ.v idx /\ SZ.v idx < SZ.v j)
  returns  x:a
  ensures  pure (SZ.v i <= SZ.v j /\ SZ.v j <= reveal sz /\ Seq.length s == (SZ.v j - SZ.v i) /\
                 SZ.v i <= SZ.v idx /\ SZ.v idx < SZ.v j /\
                 x == Seq.index s (SZ.v idx - SZ.v i))

[@@noextract_to "krml"]
fn gpu_array_write
  (#a:Type u#0)
  (#sz: erased SZ.t)
  (#i: erased SZ.t)
  (#j: erased SZ.t)
  (r:gpu_array a sz)
  (idx : SZ.t)
  (v : a)
  (#s : erased (seq a))
  preserves gpu
  requires pure (SZ.v i <= SZ.v idx /\ SZ.v idx < SZ.v j)
  requires gpu_pts_to_slice #a #sz r #1.0R i j s
  ensures  (exists* (s':seq a). gpu_pts_to_slice #a #sz r #1.0R i j s' **
              pure (SZ.v i <= SZ.v j /\ SZ.v j <= reveal sz /\ Seq.length s == (SZ.v j - SZ.v i) /\
                    SZ.v i <= SZ.v idx /\ SZ.v idx < SZ.v j /\
                    s' == Seq.upd s (SZ.v idx - SZ.v i) v))

fn gpu_memcpy_host_to_device
  (#a:Type u#0)
  {| sized a |}
  (#sz : erased SZ.t)
  (dst_garr : gpu_array a sz)
  (src_arr : vec a)
  (cnt : SZ.t)
  (#f : perm)
  (#v : erased (seq a))
  (#gv : erased (seq a))
  preserves
    cpu **
    (src_arr |-> Frac f v)
  requires
    (dst_garr |-> gv) **
    pure (cnt == sz /\ (Pulse.Lib.Vec.length src_arr == SZ.v sz \/ Seq.length v == reveal sz))
  ensures
    (dst_garr |-> v) **
    pure (Seq.length v == reveal sz)

(* blit *)
// fn gpu_memcpy_host_to_device'
//   (#a:Type u#0)
//   {| sized a |}
//   (#dst_sz : erased nat)
//   (dst_garr : gpu_array a dst_sz)
//   (dst_off : SZ.t)
//   (#src_sz : erased nat)
//   (src_arr : vec a)
//   (src_off : SZ.t)
//   (cnt : SZ.t {
//     dst_off + cnt <= dst_sz /\
//           src_off + cnt <= src_sz

//   })
//   (#f : perm)
//   (#v : erased (seq a){ Seq.length v == src_sz })
//   (#gv : erased (seq a){ Seq.length gv == dst_sz })
//   preserves
//     cpu **
//     pts_to src_arr #f v
//   requires
//     (dst_garr |-> gv)
//   ensures
//     (dst_garr |-> seq_blit gv dst_off v src_off cnt) **
//     pure (Seq.length v == reveal dst_sz)

// fn gpu_memcpy_device_to_host
//   (#a:Type u#0)
//   {| sized a |}
//   (#sz : erased nat)
//   (dst_arr : vec a)
//   (src_garr : gpu_array a sz)
//   (cnt : SZ.t)
//   (#f : perm)
//   (#v : erased (seq a))
//   (#gv : erased (seq a))
//   preserves
//     cpu **
//     (src_garr |-> Frac f gv)
//   requires
//     (dst_arr |-> v) **
//     pure (SZ.v cnt == sz /\ (Pulse.Lib.Vec.length dst_arr == sz \/ Seq.length v == reveal sz))
//   ensures
//     (dst_arr |-> gv) **
//     pure (Seq.length gv == reveal sz)

// (* blit *)
// fn gpu_memcpy_device_to_host'
//   (#a:Type u#0)
//   {| sized a |}
//   (#dst_sz : erased nat)
//   (dst_arr : vec a)
//   (dst_off : SZ.t)
//   (#src_sz : erased nat)
//   (src_garr : gpu_array a src_sz)
//   (src_off : SZ.t)
//   (cnt : SZ.t {
//     dst_off + cnt <= dst_sz /\
//           src_off + cnt <= src_sz
//   })
//   (#f : perm)
//   (#v : erased (seq a){ Seq.length v == src_sz })
//   (#gv : erased (seq a){ Seq.length gv == dst_sz })
//   preserves
//     cpu **
//     pts_to src_garr #f v
//   requires
//     (dst_arr |-> gv)
//   ensures
//     (dst_arr |-> seq_blit gv dst_off v src_off cnt) **
//     pure (Seq.length v == reveal dst_sz)

// fn gpu_memcpy_device_to_device
//   (#a:Type u#0)
//   {| sized a |}
//   (#sz : erased nat)
//   (dst_arr : gpu_array a sz)
//   (src_garr : gpu_array a sz)
//   (cnt : SZ.t)
//   (#f : perm)
//   (#v : erased (seq a))
//   (#gv : erased (seq a))
//   preserves
//     cpu **
//     (src_garr |-> Frac f gv)
//   requires
//     (dst_arr |-> v) **
//     pure (SZ.v cnt == sz /\ (Seq.length gv == sz \/ Seq.length v == sz))
//   ensures
//     (dst_arr |-> gv) **
//     pure (Seq.length gv == reveal sz)


(* Not making this unfold as it appears under bigstars. Maybe
pulse should only do weak unfolding. *)
let gpu_pts_to_array1
  (#a:Type0)
  (#sz:SZ.t)
  ([@@@mkey]arr : gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  ([@@@mkey]i:SZ.t{SZ.fits (SZ.v i + 1)})
: slprop =
  exists* s. gpu_pts_to_slice arr i (i `SZ.add` 1sz) s

ghost
fn gpu_array_slice_1
  (#[exact (`0)] uid: int) (#a:Type u#0)
  (#sz:SZ.t)
  (arr : gpu_array a sz)
  (#f : perm)
  (#v : erased (seq a) { Seq.length v == sz })
  requires pts_to arr #f v
  ensures  bigstar #uid 0 sz (fun i -> gpu_pts_to_slice arr #f (SZ.uint_to_t i) (SZ.uint_to_t (i+1)) seq![v @! i])

ghost
fn gpu_array_unslice_1
  (#uid: int) (#a:Type u#0)
  (#sz:SZ.t)
  (arr : gpu_array a sz)
  (#f : perm)
  (#v : erased (seq a) { Seq.length v == sz })
  requires bigstar #uid 0 sz (fun i -> gpu_pts_to_slice arr #f (SZ.uint_to_t i) (SZ.uint_to_t (i+1)) seq![v @! i])
  ensures  pts_to arr #f v

ghost
fn gpu_array_slice_1_underspec
  (#[exact (`0)] uid: int) (#a:Type u#0)
  (#sz:SZ.t)
  (arr : gpu_array a sz)
  (#f : perm)
  (#v : erased (seq a))
  requires arr |-> Frac f v
  ensures  bigstar #uid 0 sz (fun i -> gpu_pts_to_array1 arr #f (SZ.uint_to_t i))

ghost
fn gpu_array_unslice_1_underspec
  (#uid: int) (#a:Type u#0)
  (#sz:SZ.t)
  (arr : gpu_array a sz)
  (#f : perm)
  requires bigstar #uid 0 sz (fun i -> gpu_pts_to_array1 arr #f (SZ.uint_to_t i))
  ensures exists* (v : seq a). arr |-> Frac f v

ghost
fn gpu_slice_concat
  (#a:Type u#0)
  (#sz:SZ.t)
  (arr : gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  (#s1 #s2: erased (seq a))
  (i n m:SZ.t)
  requires gpu_pts_to_slice arr #f i n s1 ** gpu_pts_to_slice arr #f n m s2
  ensures  gpu_pts_to_slice arr #f i m (Seq.append s1 s2)

ghost
fn gpu_slice_split
  (#a:Type u#0)
  (#sz:SZ.t)
  (arr : gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  (#s1 #s2: erased (seq a))
  (i n m:SZ.t)
  requires gpu_pts_to_slice arr #f i m (Seq.append s1 s2)
  ensures  gpu_pts_to_slice arr #f i n s1 ** gpu_pts_to_slice arr #f n m s2

ghost
fn gpu_slice_empty_elim
  (#a:Type u#0) (#sz:SZ.t)
  (arr : gpu_array a sz)
  (i : SZ.t)
  requires gpu_pts_to_slice arr #'f i i 'v
  ensures  emp

ghost
fn gpu_slice_share
  (#uid: int) (#a:Type u#0)
  (#sz:SZ.t)
  (arr : gpu_array a sz)
  (m n:SZ.t)
  (k: SZ.t { k > 0 })
  (#f : perm)
  requires gpu_pts_to_slice arr #f m n 'v
  ensures
    bigstar #uid 0 k (fun _ -> gpu_pts_to_slice arr #(f /. Real.of_int k) m n 'v)

ghost
fn gpu_slice_gather
  (#uid: int) (#a:Type u#0)
  (#sz:SZ.t)
  (arr : gpu_array a sz)
  (m n:SZ.t)
  (k: SZ.t { k > 0 })
  (#f : perm) // FIXME: if we use 'f, it gets type 'real' instead of 'perm'
  requires
    bigstar #uid 0 k (fun _ -> gpu_pts_to_slice arr #(f /. Real.of_int k) m n 'v)
  ensures gpu_pts_to_slice arr #f m n 'v

ghost
fn gpu_slice_share_underspec
  (#uid : int) (#a : Type u#0)
  (#sz : SZ.t)
  (arr : gpu_array a sz)
  (m n : SZ.t)
  (k : SZ.t { k > 0 })
  requires gpu_pts_to_slice arr #'f m n 'v
  ensures bigstar #uid 0 k (fun x -> exists* v. gpu_pts_to_slice arr #('f /. Real.of_int k) m n v)

ghost
fn gpu_slice_gather_underspec
  (#uid : int) (#a : Type u#0)
  (#sz : SZ.t)
  (arr : gpu_array a sz)
  (#f : perm) // FIXME: if we use 'f, it gets type 'real' instead of 'perm'
  (m n : SZ.t)
  (k : SZ.t { k > 0 })
  requires bigstar #uid 0 k (fun x -> exists* v. gpu_pts_to_slice arr #(f /. Real.of_int k) m n v)
  ensures
    exists* v.
      gpu_pts_to_slice arr #f m n v

val adjacent
  (#a : Type u#0)
  (#s1 #s2 : SZ.t)
  (arr1 : gpu_array a s1)
  (arr2 : gpu_array a s2)
  : slprop

ghost
fn gpu_array_cut
  (#a : Type u#0)
  (#sz : SZ.t)
  (arr : gpu_array a sz)
  (#f : perm) // FIXME: if we use 'f, it gets type 'real' instead of 'perm'
  (k : SZ.t{ k <= sz })
  (#s : lseq a sz)
  requires
    (arr |-> Frac f s)
  returns
    p : (gpu_array a k & gpu_array a (sz `SZ.sub` k))
  ensures
    (p._1 |-> Frac f (seq_take k s)) **
    (p._2 |-> Frac f (seq_drop k s)) **
    adjacent p._1 p._2

ghost
fn gpu_array_paste
  (#a : Type u#0)
  (#sz1 #sz2 : SZ.t {SZ.fits (SZ.v sz1 + SZ.v sz2)})
  (arr1 : gpu_array a sz1)
  (arr2 : gpu_array a sz2)
  (#f : perm)
  requires
    (arr1 |-> Frac f 's1) **
    (arr2 |-> Frac f 's2) **
    adjacent arr1 arr2
  returns
    arr : gpu_array a (sz1 `SZ.add` sz2)
  ensures
    arr |-> Frac f (Seq.append 's1 's2)
