module Kuiper.Array.Core

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
open Kuiper.ArrayCoreAssumptions
module T = FStar.Tactics
module SZ = Kuiper.SizeT

val gpu_array (a : Type u#0) (sz : nat) : Type u#0
val loc_id_of_array (#a:Type u#0) (#sz:nat) (x:gpu_array a sz) : loc_id
val visibility_of #a #sz (x:gpu_array a sz) : visibility

let visible_at #a #sz (x:gpu_array a sz) (l:loc_id) =
  visibility_of x (loc_id_of_array x) ==
  visibility_of x l

let visible_on_gpu #a #sz (x:gpu_array a sz) (gpu_id:int) =
  gpu_of (loc_id_of_array x) == gpu_id_loc gpu_id

let is_global_array #a #sz (#[T.exact (`0)]gpu_id:int) (x:gpu_array a sz) : prop =
  visible_on_gpu x gpu_id /\
  visibility_of x == gpu_of

let is_shmem_array #a #sz (#[T.exact (`0)]gpu_id:int) (bid:int) (x:gpu_array a sz) : prop =
  visibility_of x == block_of /\
  block_of (loc_id_of_array x) == block_id_loc #gpu_id bid

let gpu_global_array (#[T.exact (`0)]gpu_id:int) a sz =
  x:gpu_array a sz { is_global_array #a #sz #gpu_id x }

let gpu_shmem_array (#[T.exact (`0)]gpu_id:int) (bid:int) a sz =
  x:gpu_array a sz { is_shmem_array #a #sz #gpu_id bid x }

(* Base address of the GPU array, used to model alignment. This number
is in units of *bytes*, not array elements. *)
val base_address (#a : Type u#0) (#sz : nat) (x : gpu_array a sz) : GTot nat

let aligned (n:pos) (#a:Type u#0) (#sz:nat) (x:gpu_array a sz) : prop =
  n /?+ base_address x

(* An offset within the array is aligned. *)
let aligned' (n:pos)
  (#a:Type u#0) {| sized a |}
  (#sz:nat) (x:gpu_array a sz)
  (off : nat) : prop =
  n /?+ (base_address x + off * size #a)

(* FIXME: I think having nat here, which forces to use erased nat
   in concrete functions, hurts Pulse inference a lot. Try to make all
   these ints. *)

(* x is the base pointer, this gives permission in [i,j) *)
val gpu_pts_to_slice
  (#a:Type u#0)
  (#sz:nat)
  ([@@@mkey] x:gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  ([@@@mkey] i : nat)
  (j : nat)
  (v : seq a)
: slprop


instance val is_send_gpu_pts_to_slice
  (#a:Type u#0)
  (#sz:nat)
  ([@@@mkey] x:gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  ([@@@mkey] i : nat)
  (j : nat)
  (v : seq a)
: is_send_across
     (visibility_of x)
     (gpu_pts_to_slice #a #sz x #f i j v)

instance is_send_across_global_array
  (#et:Type0) (#sz:_)
  (a:gpu_array et sz { is_global_array a })
  (#i #j #f #s:_)
: is_send_across gpu_of (gpu_pts_to_slice a #f i j s)
= let i : is_send_across (visibility_of a) (gpu_pts_to_slice a #f i j s) = solve in
  i

unfold
let gpu_pts_to_cell
  (#a:Type u#0)
  (#sz:nat)
  ([@@@mkey] x:gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  ([@@@mkey] i : nat)
  (v : a)
: slprop
=
  gpu_pts_to_slice x #f i (i + 1) seq![v]

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
instance has_pts_to_gpu_arr (a:Type) (sz : _) :
  has_pts_to (gpu_array a sz) (Seq.seq a) =
{
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
  ensures  pure (i <= j /\ j <= sz /\ Seq.length v == (j-i) /\ SZ.fits sz)

//allocates in gpu 0
noextract
fn gpu_array_alloc
  (#a : Type u#0)
  {| sized a |}
  (sz : SZ.t { sz > 0 })
  preserves cpu
  returns   x : gpu_array a (SZ.v sz)
  ensures
    exists* (s:seq a).
      on gpu_loc (x |-> s) **
      pure (
        Seq.length s == sz /\
        aligned 128 x /\
        is_global_array x
      )

fn gpu_array_free
  (#a:Type u#0)
  (#sz:erased nat)
  (r : gpu_array a sz)
  (#v : erased (seq a))
  preserves cpu
  requires on gpu_loc (r |-> v)
  ensures  emp

[@@noextract_to "krml"]
fn gpu_array_read
  (#a : Type u#0)
  (#sz : erased nat)
  (#i  : erased nat)
  (#j  : erased nat)
  (r : gpu_array a sz)
  (#f : perm)
  (idx : SZ.t)
  (#s : erased (seq a))
  preserves gpu_pts_to_slice #a #sz r #f i j s
  requires pure (i <= SZ.v idx /\ SZ.v idx < j)
  returns  x:a
  ensures  pure (i <= j /\ j <= sz /\ Seq.length s == (j-i) /\
                 i <= SZ.v idx /\ SZ.v idx < j /\
                 x == Seq.index s (SZ.v idx - i))

[@@noextract_to "krml"]
fn gpu_array_write
  (#a:Type u#0)
  (#sz: erased nat)
  (#i: erased nat)
  (#j: erased nat)
  (r:gpu_array a sz)
  (idx : SZ.t)
  (v : a)
  (#s : erased (seq a))
  requires pure (i <= SZ.v idx /\ SZ.v idx < j)
  requires gpu_pts_to_slice #a #sz r #1.0R i j s
  ensures
    with_pure
      (i <= j /\ j <= sz /\ Seq.length s == (j-i) /\ i <= SZ.v idx /\ SZ.v idx < j)
      (fun _ -> exists* s'. gpu_pts_to_slice #a #sz r #1.0R i j s' ** pure (s' == Seq.upd s (SZ.v idx - i) v))

(* blit *)
fn gpu_memcpy_host_to_device'
  (#a:Type u#0)
  {| sized a |}
  (#dst_sz : erased nat)
  (dst_garr : gpu_array a dst_sz)
  (dst_off : SZ.t)
  (#src_sz : erased nat)
  (src_arr : vec a)
  (src_off : SZ.t)
  (cnt : SZ.t {
    dst_off + cnt <= dst_sz /\
          src_off + cnt <= src_sz

  })
  (#f : perm)
  (#v : erased (seq a){ Seq.length v == src_sz })
  (#gv : erased (seq a){ Seq.length gv == dst_sz })
  preserves
    cpu **
    pts_to src_arr #f v
  requires
    on gpu_loc (dst_garr |-> gv)
  ensures
    exists* s'. on gpu_loc (dst_garr |-> s')
               ** pure (s' == seq_blit gv dst_off v src_off cnt /\ Seq.length s' == reveal dst_sz)

fn gpu_memcpy_host_to_device
  (#a:Type u#0)
  {| sized a |}
  (#sz : erased nat)
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
    on gpu_loc (dst_garr |-> gv)
  requires
    pure (SZ.v cnt == sz /\
          (Pulse.Lib.Vec.length src_arr == sz \/ Seq.length v == reveal sz))
  ensures
    on gpu_loc (dst_garr |-> v) **
    pure (Seq.length v == reveal sz)

(* blit *)
fn gpu_memcpy_device_to_host'
  (#a:Type u#0)
  {| sized a |}
  (#dst_sz : erased nat)
  (dst_arr : vec a)
  (dst_off : SZ.t)
  (#src_sz : erased nat)
  (src_garr : gpu_array a src_sz)
  (src_off : SZ.t)
  (cnt : SZ.t {
    dst_off + cnt <= dst_sz /\
          src_off + cnt <= src_sz
  })
  (#f : perm)
  (#v : erased (seq a){ Seq.length v == src_sz })
  (#gv : erased (seq a){ Seq.length gv == dst_sz })
  preserves
    cpu **
    on gpu_loc (src_garr |-> Frac f (v<:seq _))
  requires
    (dst_arr |-> gv)
  ensures
    exists* s'. dst_arr |-> s' **
    pure (s'==seq_blit gv dst_off v src_off cnt /\ Seq.length s' == reveal dst_sz)

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
    on gpu_loc (src_garr |-> Frac f gv)
  requires
    (dst_arr |-> v) **
    pure (
      SZ.v cnt == sz /\
      (Pulse.Lib.Vec.length dst_arr == sz \/ Seq.length v == reveal sz))
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
    on gpu_loc (src_garr |-> Frac f gv)
  requires
    on gpu_loc (dst_arr |-> v) **
    pure (
      SZ.v cnt == sz /\
      (Seq.length gv == sz \/ Seq.length v == sz))
  ensures
    on gpu_loc (dst_arr |-> gv) **
    pure (Seq.length gv == reveal sz)


ghost
fn gpu_array_slice_1
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#v : erased (seq a) { Seq.length v == sz })
  requires pts_to arr #f v
  ensures  forall+ (i: natlt sz). gpu_pts_to_cell arr #f i (v @! i)

ghost
fn gpu_array_unslice_1
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#v : erased (seq a) { Seq.length v == sz })
  requires forall+ (i: natlt sz). gpu_pts_to_cell arr #f i (v @! i)
  ensures  pts_to arr #f v

ghost
fn gpu_slice_concat
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  (#s1 #s2: erased (seq a))
  (i n m:nat)
  requires gpu_pts_to_slice arr #f i n s1 ** gpu_pts_to_slice arr #f n m s2
  ensures  gpu_pts_to_slice arr #f i m (s1 @+ s2)

ghost
fn gpu_slice_split
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  (#s1 #s2: erased (seq a))
  (i n m:nat)
  requires gpu_pts_to_slice arr #f i m (s1 @+ s2) ** pure (i <= n /\ n <= m /\ (i + Seq.length s1 == n \/ n + Seq.length s2 == m))
  ensures  gpu_pts_to_slice arr #f i n s1 ** gpu_pts_to_slice arr #f n m s2

ghost
fn gpu_slice_share
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (m n:nat)
  (k: nat { k > 0 })
  (#f : perm)
  requires gpu_pts_to_slice arr #f m n 'v
  ensures
    forall+ (_:natlt k). gpu_pts_to_slice arr #(f /. Real.of_int k) m n 'v

ghost
fn gpu_slice_share_underspec
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (m n:nat)
  (k: nat { k > 0 })
  (#f : perm)
  requires gpu_pts_to_slice arr #f m n 'v
  ensures
    forall+ (_:natlt k). exists* v. gpu_pts_to_slice arr #(f /. Real.of_int k) m n v

ghost
fn gpu_slice_gather
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (m n:nat)
  (k: nat { k > 0 })
  (#f : perm) // FIXME: if we use 'f, it gets type 'real' instead of 'perm'
  requires
    forall+ (_:natlt k). gpu_pts_to_slice arr #(f /. Real.of_int k) m n 'v
  ensures gpu_pts_to_slice arr #f m n 'v

ghost
fn gpu_slice_gather_underspec
  (#a : Type u#0)
  (#sz : nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (m n : nat)
  (k : nat { k > 0 })
  requires
    forall+ (_ : natlt k).
      exists* v. gpu_pts_to_slice arr #(f /. Real.of_int k) m n v
  ensures
    exists* v.
      gpu_pts_to_slice arr #f m n v

ghost
fn gpu_slice_pts_to_eq
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (m n:nat)
  (#f1 f2 : perm)
  (#v1 #v2 : seq a)
  requires
    gpu_pts_to_slice arr #f1 m n v1 **
    gpu_pts_to_slice arr #f2 m n v2
  ensures
    gpu_pts_to_slice arr #f1 m n v2 **
    gpu_pts_to_slice arr #f2 m n v2

// val adjacent
//   (#a : Type u#0)
//   (#s1 #s2 : nat)
//   (arr1 : gpu_global_array a s1)
//   (arr2 : gpu_global_array a s2)
//   : slprop

// ghost
// fn gpu_array_cut
//   (#a : Type u#0) {| sized a |}
//   (#sz : nat)
//   (arr : gpu_global_array a sz)
//   (k : SZ.t{ k <= sz })
//   (#s : lseq a sz)
//   requires
//     (arr |-> Frac 'f s)
//   returns
//     p : (gpu_global_array a k & gpu_global_array a (sz - k))
//   ensures
//     (p._1 |-> Frac 'f (seq_take k s)) **
//     (p._2 |-> Frac 'f (seq_drop k s)) **
//     adjacent p._1 p._2
//   ensures
//     // Should this below just be the definition of adjacent?
//     pure (base_address p._1 == base_address arr) **
//     pure (base_address p._2 == base_address arr + SZ.v k * size #a)

// ghost
// fn gpu_array_paste
//   (#a : Type u#0)
//   (#sz1 #sz2 : nat)
//   (arr1 : gpu_global_array a sz1)
//   (arr2 : gpu_global_array a sz2)
//   requires
//     (arr1 |-> Frac 'f 's1) **
//     (arr2 |-> Frac 'f 's2) **
//     adjacent arr1 arr2
//   returns
//     arr : gpu_global_array a (sz1 + sz2)
//   ensures
//     arr |-> Frac 'f ('s1 @+ 's2)
