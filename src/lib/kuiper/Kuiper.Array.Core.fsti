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
open Kuiper.PtsTo
open Kuiper.ForEvery
open Kuiper.Divides { (/?+) }
open Kuiper.ArrayCoreAssumptions
module T = FStar.Tactics
module SZ = Kuiper.SizeT
module A = Pulse.Lib.Array

val is_full_slice
  (#et:Type)
  ([@@@mkey]a : array et)
  (n : nat) : slprop

// val gpu_array (a : Type u#0) (sz : nat) : Type u#0
// val loc_id_of_array (#a:Type u#0) (#sz:nat) (x:gpu_array a sz) : loc_id
val visibility_of #a (x:array a) : visibility

// let visible_at #a #sz (x:gpu_array a sz) (l:loc_id) =
//   visibility_of x (loc_id_of_array x) ==
//   visibility_of x l

let visible_on_gpu #a (x:array a) (gpu_id:int) =
  gpu_of (loc_id_of_array x) == gpu_id_loc gpu_id

let is_global_array #a (#[T.exact (`0)]gpu_id:int) (x:array a) : prop =
  visible_on_gpu x gpu_id /\
  visibility_of x == gpu_of

// let is_shmem_array #a #sz (#[T.exact (`0)]gpu_id:int) (bid:int) (x:gpu_array a sz) : prop =
//   visibility_of x == block_of /\
//   block_of (loc_id_of_array x) == block_id_loc #gpu_id bid

(* Base address of the GPU array, used to model alignment. This number
is in units of *bytes*, not array elements. *)
val base_address (#a : Type u#0) (x : array a) : GTot nat

let aligned (n:pos) (#a:Type u#0) (x:array a) : prop =
  n /?+ base_address x

(* An offset within the array is aligned. *)
let aligned' (n:pos)
  (#a:Type u#0) {| sized a |}
  (x : array a)
  (off : nat) : prop =
  n /?+ (base_address x + off * size #a)

(* FIXME: I think having nat here, which forces to use erased nat
   in concrete functions, hurts Pulse inference a lot. Try to make all
   these ints. *)

(* x is the base pointer, this gives permission in [i,j) *)
val pts_to_slice
  (#a:Type u#0)
  ([@@@mkey] x : array a)
  (#[exact (`1.0R)] f : perm)
  ([@@@mkey] i : nat)
  (j : nat)
  (v : seq a)
  : slprop

[@@pulse_intro]
ghost
fn array_to_slice
  (#et:Type)
  (a : array et)
  (#f : perm)
  (#s : seq et)
  requires
    a |-> Frac f s
  ensures
    pts_to_slice a #f 0 (Seq.length s) s **
    is_full_slice a (Seq.length s) **
    pure (Seq.length s == Pulse.Lib.Array.length a)

// FIXME: This should probably use two different [n] parameters and a proof of
// their equality. Or replace is_full_slice by `n == Array.length a`.
[@@pulse_intro]
ghost
fn slice_to_array
  (#et:Type)
  (a : array et)
  (#f : perm)
  (#s : seq et)
  (#n : nat)
  requires
    pts_to_slice a #f 0 n s **
    is_full_slice a n
  ensures
    (* Cannot use typeclasses here or lemma will not kick in. *)
    Pulse.Lib.Array.pts_to a #f s

unfold
let pts_to_cell
  (#a:Type u#0)
  ([@@@mkey] x : array a)
  (#[exact (`1.0R)] f : perm)
  ([@@@mkey] i : nat)
  (v : a)
: slprop
=
  pts_to_slice x #f i (i + 1) seq![v]

// unfold
// let pts_to
//   (#a:Type u#0)
//   (#sz:nat)
//   ([@@@mkey] x : larray a sz)
//   (#[exact (`1.0R)] f : perm)
//   (v : seq a)
// : slprop
// =
//   pts_to_slice x #f 0 sz v

instance val is_send_pts_to
  (#a:Type u#0)
  (x : array a)
  (#[exact (`1.0R)] f : perm)
  (v : seq a)
  : is_send_across
      (visibility_of x)
      (pts_to x #f v)

instance val is_send_pts_to_slice
  (#a:Type u#0)
  (x : array a)
  (#[exact (`1.0R)] f : perm)
  (i j : nat)
  (v : seq a)
  : is_send_across
      (visibility_of x)
      (pts_to_slice #a x #f i j v)

(* Single generic weakening instance per base resource: sendable across any v
   that refines the array's home visibility. Subsumes the old gpu-specific
   is_send_across_global_array and the gpu->block lift (via the block_refines_gpu
   SMTPat), and serves block arrays too (v == visibility_of x reflexively). *)
instance is_send_pts_to_weaken
  (#et:Type0)
  (x : array et)
  (v : visibility { vis_refines v (visibility_of x) })
  (#[exact (`1.0R)] f : perm)
  (s : seq et)
: is_send_across v (pts_to x #f s)
= weaken (is_send_pts_to x #f s) ()

instance is_send_pts_to_slice_weaken
  (#et:Type0)
  (x : array et)
  (v : visibility { vis_refines v (visibility_of x) })
  (#[exact (`1.0R)] f : perm)
  (i j : nat) (s : seq et)
: is_send_across v (pts_to_slice x #f i j s)
= weaken (is_send_pts_to_slice x #f i j s) ()

(* Sendability stated on the raw [A.pts_to] (a named val), independent of the
   pointer's static type and of the [pts_to] class wrappers. The [pts_to] class
   method, the [frac] wrapper ([x |-> Frac f v]) and the [lseq] wrapper are all
   [pulse_unfold] and reduce to [A.pts_to], so this single instance serves
   array, larray, frac- and lseq-keyed goals alike (e.g. [live] of a raw larray,
   or a sparse matrix's backing larrays which use [|-> Frac]). *)
instance is_send_pts_to_raw
  (#et:Type0)
  (x : array et)
  (v : visibility { vis_refines v (visibility_of x) })
  (#[exact (`1.0R)] f : perm)
  (s : seq et)
: is_send_across v (A.pts_to x #f s)
= is_send_pts_to_weaken x v s


[@@pulse_unfold; FStar.Tactics.Typeclasses.noinst]
instance cell_pts_to (#a : Type)
  : has_pts_to (cell (array a) nat) a
= {
  pts_to = (fun (Cell ar i) #f v -> pts_to_cell ar #f i v);
}

ghost
fn pts_to_slice_ref
  (#a:Type u#0)
  (#f : perm)
  (x:array a)
  (i:nat) (j:nat)
  (#v : seq a)
  preserves pts_to_slice x #f i j v
  requires emp
  ensures  pure (i <= j /\ j <= Pulse.Lib.Array.length x /\ Seq.length v == (j-i) /\ SZ.fits (Pulse.Lib.Array.length x))

// Allocates in gpu 0
noextract
fn gpu_array_alloc
  (#a : Type u#0)
  {| sized a |}
  (sz : SZ.t { sz > 0 })
  preserves cpu
  returns   x : larray a (SZ.v sz)
  ensures
    exists* (s:seq a).
      on gpu_loc (x |-> s) **
      pure (
        Seq.length s == sz /\
        aligned 128 x /\
        is_global_array x /\
        A.is_full_array x
      )

fn gpu_array_free
  (#a:Type u#0)
  (r : array a)
  (#v : erased (seq a))
  preserves cpu
  requires pure (A.is_full_array r)
  requires on gpu_loc (r |-> v)
  ensures  emp

fn slice_read
  (#a : Type u#0)
  (#i  : erased nat)
  (#j  : erased nat)
  (r : array a)
  (#f : perm)
  (idx : SZ.t)
  (#s : erased (seq a))
  preserves
    pts_to_slice #a r #f i j s
  requires pure (i <= idx /\ idx < j)
  returns  x:a
  ensures
    pure (i <= j /\ Seq.length s == (j-i) /\
          i <= SZ.v idx /\ SZ.v idx < j /\
          x == Seq.index s (SZ.v idx - i))

fn slice_write
  (#a:Type u#0)
  (#i: erased nat)
  (#j: erased nat)
  (r : array a)
  (idx : SZ.t)
  (v : a)
  (#s : erased (seq a))
  requires pure (i <= idx /\ idx < j)
  requires
    pts_to_slice r #1.0R i j s
  ensures
    with_pure (Seq.length s == (j-i) /\ i <= idx /\ idx < j) (fun _ ->
      exists* s'.
        pts_to_slice r #1.0R i j s' **
        pure (s' == Seq.upd s (SZ.v idx - i) v))

(* blit *)
fn gpu_memcpy_host_to_device'
  (#a:Type u#0)
  {| sized a |}
  (#dst_sz : erased nat)
  (dst_garr : larray a dst_sz)
  (dst_off : SZ.t)
  (#src_sz : erased nat)
  (src_arr : vec a)
  (src_off : SZ.t)
  (cnt : SZ.t { dst_off + cnt <= dst_sz /\ src_off + cnt <= src_sz })
  (#f : perm)
  (#v : erased (seq a){ Seq.length v == src_sz })
  (#gv : erased (seq a){ Seq.length gv == dst_sz })
  preserves
    cpu **
    src_arr |-> Frac f (v <: seq a)
  requires
    on gpu_loc (dst_garr |-> gv)
  ensures
    exists* s'.
      on gpu_loc (dst_garr |-> s') **
      pure (s' == seq_blit gv dst_off v src_off cnt /\ Seq.length s' == reveal dst_sz)

fn gpu_memcpy_host_to_device
  (#a:Type u#0)
  {| sized a |}
  (#sz : erased nat)
  (dst_garr : larray a sz)
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
  (src_garr : larray a src_sz)
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
    dst_arr |-> gv
  ensures
    exists* s'. dst_arr |-> s' **
    pure (s'==seq_blit gv dst_off v src_off cnt /\ Seq.length s' == reveal dst_sz)

fn gpu_memcpy_device_to_host
  (#a:Type u#0)
  {| sized a |}
  (#sz : erased nat)
  (dst_arr : vec a)
  (src_garr : larray a sz)
  (cnt : SZ.t)
  (#f : perm)
  (#v : erased (seq a))
  (#gv : erased (seq a))
  preserves
    cpu **
    on gpu_loc (src_garr |-> Frac f gv)
  requires
    dst_arr |-> v **
    pure (
      SZ.v cnt == sz /\
      (Pulse.Lib.Vec.length dst_arr == sz \/ Seq.length v == reveal sz))
  ensures
    dst_arr |-> gv **
    pure (Seq.length gv == reveal sz)

fn gpu_memcpy_device_to_device
  (#a:Type u#0)
  {| sized a |}
  (#sz : erased nat)
  (dst_garr : larray a sz)
  (src_garr : larray a sz)
  (cnt : SZ.t)
  (#f : perm)
  (#v : erased (seq a))
  (#gv : erased (seq a))
  preserves
    cpu **
    on gpu_loc (src_garr |-> Frac f gv)
  requires
    on gpu_loc (dst_garr |-> v) **
    pure (
      SZ.v cnt == sz /\
      (Seq.length gv == sz \/ Seq.length v == sz))
  ensures
    on gpu_loc (dst_garr |-> gv) **
    pure (Seq.length gv == reveal sz)


ghost
fn array_slice_1
  (#a:Type u#0)
  (#sz:nat)
  (arr : larray a sz)
  (#f : perm)
  (#v : erased (seq a) { Seq.length v == sz })
  requires
    pts_to arr #f v
  ensures
    forall+ (i: natlt sz).
      pts_to_cell arr #f i (v @! i)

ghost
fn array_unslice_1
  (#a:Type u#0)
  (#sz:nat)
  (arr : larray a sz)
  (#f : perm)
  (#v : erased (seq a) { Seq.length v == sz })
  requires
    forall+ (i: natlt sz).
      pts_to_cell arr #f i (v @! i)
  ensures
    pts_to arr #f v

ghost
fn slice_concat
  (#a:Type u#0)
  (arr : array a)
  (#[exact (`1.0R)] f : perm)
  (#s1 #s2: erased (seq a))
  (i n m:nat)
  requires pts_to_slice arr #f i n s1 ** pts_to_slice arr #f n m s2
  ensures  pts_to_slice arr #f i m (s1 @+ s2)

ghost
fn slice_split
  (#a:Type u#0)
  (arr : array a)
  (#[exact (`1.0R)] f : perm)
  (#s1 #s2: erased (seq a))
  (i n m:nat)
  requires pts_to_slice arr #f i m (s1 @+ s2) ** pure (i <= n /\ n <= m /\ (i + Seq.length s1 == n \/ n + Seq.length s2 == m))
  ensures  pts_to_slice arr #f i n s1 ** pts_to_slice arr #f n m s2

ghost
fn slice_share
  (#a:Type u#0)
  (arr : array a)
  (m n:nat)
  (k: nat { k > 0 })
  (#f : perm)
  requires
    pts_to_slice arr #f m n 'v
  ensures
    forall+ (_:natlt k).
      pts_to_slice arr #(f /. Real.of_int k) m n 'v

ghost
fn slice_gather
  (#a:Type u#0)
  (arr : array a)
  (m n:nat)
  (k: nat { k > 0 })
  (#f : perm) // FIXME: if we use 'f, it gets type 'real' instead of 'perm'
  requires
    forall+ (_:natlt k).
      pts_to_slice arr #(f /. Real.of_int k) m n 'v
  ensures
    pts_to_slice arr #f m n 'v

ghost
fn slice_gather_underspec
  (#a : Type u#0)
  (arr : array a)
  (#f : perm)
  (m n : nat)
  (k : nat { k > 0 })
  requires
    forall+ (_ : natlt k).
      exists* v. pts_to_slice arr #(f /. Real.of_int k) m n v
  ensures
    exists* v.
      pts_to_slice arr #f m n v

ghost
fn array_gather_underspec
  (#a : Type u#0)
  (arr : array a)
  (#f : perm)
  (k : nat { k > 0 })
  requires
    forall+ (_ : natlt k).
      exists* (v : seq a). arr |-> Frac (f /. Real.of_int k) v
  ensures
    exists* (v : seq a).
      arr |-> Frac f v

ghost
fn slice_pts_to_eq
  (#a:Type u#0)
  (arr : array a)
  (m n:nat)
  (#f1 f2 : perm)
  (#v1 #v2 : seq a)
  preserves
    pts_to_slice arr #f1 m n v1 **
    pts_to_slice arr #f2 m n v2
  ensures
    pure (v1 == v2)

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

val ref_of_array_cell
  (#et : Type0)
  (a : array et)
  (i : nat)
  : ref et

fn get_ref_of_array_cell
  (#et : Type0)
  (a : array et)
  (i : sz)
  returns
    r : ref et
  ensures
    pure (r == ref_of_array_cell a i)

ghost
fn array_cell_to_ref
  (#et : Type0)
  (a : array et)
  (i : nat)
  (#f : perm)
  (#v : erased et)
  requires
    Cell a i |-> Frac f v
  ensures
    ref_of_array_cell a i |-> Frac f v

ghost
fn array_cell_from_ref
  (#et : Type0)
  (a : array et)
  (i : nat)
  (#f : perm)
  (#v : erased et)
  requires
    ref_of_array_cell a i |-> Frac f v
  ensures
    Cell a i |-> Frac f v
