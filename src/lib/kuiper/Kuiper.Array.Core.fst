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
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module SendSync = Pulse.Lib.SendSync

assume
val sized_types_inhabited (#a:Type) {| sized a |} : a

instance 
is_send_across_pts_to_mask_instance (#a: Type u#a) (x:A.array a) (f:perm) (s:seq a) (mask:nat -> prop)
: is_send_across (visibility_of_array x) (pts_to_mask x #f s mask)
= is_send_across_pts_to_mask x f s mask


//a trusted primitive to model gpu alloc, memcpy, shared memory allocation etc.
//these are commands that can be run from the CPU but has effect at the gpu location
fn __primitive__exec_on_gpu_loc
  (#a:Type0) (#pre:slprop) (#post:a -> slprop)
  {| placeless pre |} 
  (placeless_post: (x:a -> placeless (post x)))
  (l:loc_id)
  (f: unit -> stt a (loc l ** pre) (fun x -> loc l ** post x))
preserves cpu 
requires pre
returns x:a
ensures post x
{ admit() }

let gpu_array_core (a : Type u#0) (sz : nat) : Type u#0 = A.larray a sz
let loc_id_of_array #a #sz x = loc_id_of_array x
let visibility_of #a #sz x = visibility_of_array x

(* Base address of the GPU array, used to model alignment. This number
is in units of *bytes*, not array elements. *)
let base_address (#a : Type u#0) (#sz : nat) (x : gpu_array_core a sz)
: GTot nat
= core_base_address x

let mask_of (i j:nat) (n:nat) : prop = i <= n /\ n < j

(* x is the base pointer, this gives permission in [i,j) *)
let gpu_pts_to_slice_core 
  (#a:Type u#0)
  (#sz:nat)
  ([@@@mkey] x:gpu_array_core a sz)
  (#[exact (`1.0R)] f : perm)
  ([@@@mkey] i : nat)
  (j : nat)
  (v : seq a)
= exists* (s:erased (Seq.seq a)). 
    A.pts_to_mask x #f s (mask_of i j) **
    pure (i <= j /\
          j <= Seq.length s /\
          Seq.slice s i j `Seq.equal` v /\
          SZ.fits sz /\
          Seq.length s == sz /\
          A.is_full_array x)

instance is_send_gpu_pts_to_slice_core 
  (#a:Type u#0)
  (#sz:nat)
  ([@@@mkey] x:gpu_array_core a sz)
  (#[exact (`1.0R)] f : perm)
  ([@@@mkey] i : nat)
  (j : nat)
  (v : seq a)
: is_send_across
     (visibility_of_array x)
     (gpu_pts_to_slice_core #a #sz x #f i j v)
= solve

let gpu_pts_to_slice
  (#a:Type u#0)
  (#sz:nat)
  ([@@@mkey] x:gpu_array_core a sz)
  (#[exact (`1.0R)] f : perm)
  ([@@@mkey] i : nat)
  (j : nat)
  (v : seq a)
: slprop
= on (loc_id_of_array x) (gpu_pts_to_slice_core x #f i j v)

let placeless_gpu_pts_to_slice
  (#a:Type u#0)
  (#sz:nat)
  (x:gpu_array_core a sz)
  (#[exact (`1.0R)] f : perm)
  (i : nat)
  (j : nat)
  (v : seq a)
 : placeless (gpu_pts_to_slice x #f i j v)
 = solve

ghost
fn gpu_pts_to_slice_ref
  (#a:Type u#0)
  (#sz:nat)
  (#f : perm)
  (x:gpu_array_core a sz)
  (i:nat) (j:nat)
  (#v : seq a)
  preserves gpu_pts_to_slice x #f i j v
  requires emp
  ensures  pure (i <= j /\ j <= sz /\ Seq.length v == (j-i) /\ SZ.fits sz)
{
  ghost
  fn aux ()
  requires (loc (loc_id_of_array x) ** gpu_pts_to_slice x #f i j v)
  ensures  (loc (loc_id_of_array x) ** 
    (gpu_pts_to_slice x #f i j v **
     pure (i <= j /\ j <= sz /\ Seq.length v == (j-i) /\ SZ.fits sz)))
  { 
    unfold gpu_pts_to_slice;
    on_elim _;
    unfold gpu_pts_to_slice_core;
    fold (gpu_pts_to_slice_core x #f i j v);
    on_intro #_ (gpu_pts_to_slice_core x #f i j v);
    fold gpu_pts_to_slice;
  };
  ghost_impersonate _ _ _ aux
}

let dummy = emp 


noextract
fn gpu_array_alloc_vis
  (#a : Type u#0)
  {| sized a |}
  (sz : SZ.t)
  (l:loc_id)
  (vis:visibility)
  preserves cpu
  returns   x : gpu_array_core a (SZ.v sz)
  ensures
    exists* (s:seq a). x |-> s ** pure (Seq.length s == sz)
  ensures
    pure (
      aligned 128 x /\
      visibility_of_array x == vis /\
      loc_id_of_array x == l
    )
{
  fn aux ()
  requires loc l ** dummy
  returns  x : gpu_array_core a (SZ.v sz)
  ensures loc l
  ensures exists* (s:seq a). 
    gpu_pts_to_array x s **
    pure (Seq.length s == sz /\
          visibility_of_array x == vis /\
          loc_id_of_array x == l)
  {
    unfold dummy;
    let default_value : a = sized_types_inhabited;
    let x = alloc_array_with_vis default_value sz l vis;
    A.mask_mext x (mask_of 0 (SZ.v sz));
    fold (gpu_pts_to_slice_core #a #(SZ.v sz) x 0 (SZ.v sz) (Seq.create (SZ.v sz) default_value));
    on_intro #l (gpu_pts_to_slice_core _ _ _ _);
    fold (gpu_pts_to_slice _ _ _ _);
    fold (gpu_pts_to_array _ _);
    x
  };
  fold dummy;
  __primitive__exec_on_gpu_loc (fun _ -> solve) l aux;
}


noextract
fn gpu_array_alloc
  (#a : Type u#0)
  {| d: sized a |}
  (sz : SZ.t)
  preserves cpu
  returns   x : gpu_array a (SZ.v sz)
  ensures
    exists* (s:seq a). x |-> s ** pure (Seq.length s == sz)
  ensures
    pure (aligned 128 x)
{
  let x = gpu_array_alloc_vis #a sz (gpu_of (gpu_id_loc 0)) (gpu_of);
  gpu_of_idem (gpu_id_loc 0);
  x
}

fn gpu_array_free
  (#a:Type u#0)
  (#sz:erased nat)
  (r : gpu_array_core a sz)
  (#v : erased (seq a))
  preserves cpu
  requires r |-> v
  ensures  emp
{
  fn aux ()
  requires loc (loc_id_of_array r)
  requires r |-> v
  ensures  loc (loc_id_of_array r)
  ensures dummy
  {
    unfold gpu_pts_to_array;
    unfold gpu_pts_to_slice;
    on_elim _;
    unfold gpu_pts_to_slice_core;
    A.mask_mext r (fun _ -> True);
    A.mask_free r;
    fold dummy;
  };
  __primitive__exec_on_gpu_loc (fun _ -> solve) (loc_id_of_array r) aux;
  unfold dummy;
}

[@@noextract_to "krml"]
fn gpu_array_read_local
  (#a : Type u#0)
  (#sz : erased nat)
  (#i  : erased nat)
  (#j  : erased nat)
  (r : gpu_array_core a sz)
  (#f : perm)
  (idx : SZ.t)
  (#s : erased (seq a))
  preserves gpu_pts_to_slice_core #a #sz r #f i j s
  requires pure (i <= SZ.v idx /\ SZ.v idx < j)
  returns  x:a
  ensures  pure (i <= j /\ j <= sz /\ Seq.length s == (j-i) /\
                 i <= SZ.v idx /\ SZ.v idx < j /\
                 x == Seq.index s (SZ.v idx - i))
{
  unfold gpu_pts_to_slice_core;
  let v = A.mask_read r idx;
  fold gpu_pts_to_slice_core #a #sz r #f i j s;
  v
}

[@@noextract_to "krml"]
fn gpu_array_write_local
  (#a:Type u#0)
  (#sz: erased nat)
  (#i: erased nat)
  (#j: erased nat)
  (r:gpu_array_core a sz)
  (idx : SZ.t)
  (v : a)
  (#s : erased (seq a))
  requires pure (i <= SZ.v idx /\ SZ.v idx < j)
  requires gpu_pts_to_slice_core #a #sz r #1.0R i j s
  ensures
    with_pure
      (i <= j /\ j <= sz /\ Seq.length s == (j-i) /\ i <= SZ.v idx /\ SZ.v idx < j)
      (fun _ -> gpu_pts_to_slice_core #a #sz r #1.0R i j (Seq.upd s (SZ.v idx - i) v))
{
  unfold gpu_pts_to_slice_core;
  A.mask_write r idx v;
  fold (gpu_pts_to_slice_core #a #sz r #1.0R i j (Seq.upd s (SZ.v idx - i) v));
}

[@@noextract_to "krml"]
fn gpu_array_read_loc
  (#a : Type u#0)
  (#sz : erased nat)
  (#i  : erased nat)
  (#j  : erased nat)
  (r : gpu_array_core a sz)
  (#f : perm)
  (idx : SZ.t)
  (#s : erased (seq a))
  (#l : loc_id)
  preserves loc l
  preserves gpu_pts_to_slice #a #sz r #f i j s
  requires pure (i <= SZ.v idx /\ SZ.v idx < j /\ r `visible_at` l)
  returns  x:a
  ensures  pure (i <= j /\ j <= sz /\ Seq.length s == (j-i) /\
                 i <= SZ.v idx /\ SZ.v idx < j /\
                 x == Seq.index s (SZ.v idx - i))
{
  unfold gpu_pts_to_slice;
  SendSync.is_send_across_elim (visibility_of_array r) (gpu_pts_to_slice_core r #f i j s) l;
  on_elim _;
  let v = gpu_array_read_local r idx;
  on_intro #l (gpu_pts_to_slice_core #a #sz r #f i j s);
  SendSync.is_send_across_elim (visibility_of_array r) (gpu_pts_to_slice_core r #f i j s) (loc_id_of_array r);
  fold gpu_pts_to_slice;
  v
}

[@@noextract_to "krml"]
fn gpu_array_write_loc
  (#a:Type u#0)
  (#sz: erased nat)
  (#i: erased nat)
  (#j: erased nat)
  (r:gpu_array_core a sz)
  (idx : SZ.t)
  (v : a)
  (#s : erased (seq a))
  (#l : loc_id)
  preserves loc l
  requires pure (i <= SZ.v idx /\ SZ.v idx < j /\ r `visible_at` l)
  requires gpu_pts_to_slice #a #sz r #1.0R i j s
  ensures
    with_pure
      (i <= j /\ j <= sz /\ Seq.length s == (j-i) /\ i <= SZ.v idx /\ SZ.v idx < j)
      (fun _ -> gpu_pts_to_slice #a #sz r #1.0R i j (Seq.upd s (SZ.v idx - i) v))
{
  unfold gpu_pts_to_slice;
  SendSync.is_send_across_elim (visibility_of_array r) (gpu_pts_to_slice_core r i j s) l;
  on_elim _;
  gpu_array_write_local r idx v;
  on_intro #l (gpu_pts_to_slice_core #a #sz r i j _);
  SendSync.is_send_across_elim (visibility_of_array r) (gpu_pts_to_slice_core r i j (Seq.upd s (SZ.v idx - i) v)) (loc_id_of_array r);
  fold gpu_pts_to_slice;
}


[@@noextract_to "krml"]
fn gpu_array_read
  (#a : Type u#0)
  (#sz : erased nat)
  (#i  : erased nat)
  (#j  : erased nat)
  (#gpu_id : erased int)
  (r : gpu_array #gpu_id a sz)
  (#f : perm)
  (idx : SZ.t)
  (#s : erased (seq a))
  preserves gpu #gpu_id
  preserves gpu_pts_to_slice #a #sz r #f i j s
  requires pure (i <= SZ.v idx /\ SZ.v idx < j)
  returns  x:a
  ensures  pure (i <= j /\ j <= sz /\ Seq.length s == (j-i) /\
                 i <= SZ.v idx /\ SZ.v idx < j /\
                 x == Seq.index s (SZ.v idx - i))
{
  unfold (gpu #gpu_id); 
  let x = gpu_array_read_loc r idx;
  fold (gpu #gpu_id);
  x
}

[@@noextract_to "krml"]
fn gpu_array_write
  (#a:Type u#0)
  (#sz: erased nat)
  (#i: erased nat)
  (#j: erased nat)
  (#gpu_id : erased int)
  (r:gpu_array #gpu_id a sz)
  (idx : SZ.t)
  (v : a)
  (#s : erased (seq a))
  preserves gpu #gpu_id
  requires pure (i <= SZ.v idx /\ SZ.v idx < j)
  requires gpu_pts_to_slice #a #sz r #1.0R i j s
  ensures
    with_pure
      (i <= j /\ j <= sz /\ Seq.length s == (j-i) /\ i <= SZ.v idx /\ SZ.v idx < j)
      (fun _ -> gpu_pts_to_slice #a #sz r #1.0R i j (Seq.upd s (SZ.v idx - i) v))
{
  unfold (gpu #gpu_id);
  gpu_array_write_loc r idx v;
  fold (gpu #gpu_id);
}


[@@noextract_to "krml"]
fn gpu_shmem_array_read
  (#a : Type u#0)
  (#sz : erased nat)
  (#i  : erased nat)
  (#j  : erased nat)
  (#gpu_id #nblk #bid:erased int)
  (r:gpu_shmem_array #gpu_id bid a sz)
  (#f : perm)
  (idx : SZ.t)
  (#s : erased (seq a))
  preserves block_id #gpu_id nblk bid
  preserves gpu_pts_to_slice #a #sz r #f i j s
  requires pure (i <= SZ.v idx /\ SZ.v idx < j)
  returns  x:a
  ensures  pure (i <= j /\ j <= sz /\ Seq.length s == (j-i) /\
                 i <= SZ.v idx /\ SZ.v idx < j /\
                 x == Seq.index s (SZ.v idx - i))
{
  unfold block_id #gpu_id nblk bid;
  let v = gpu_array_read_loc r idx;
  fold block_id #gpu_id nblk bid;
  v
}


[@@noextract_to "krml"]
fn gpu_shmem_array_write
  (#a:Type u#0)
  (#sz: erased nat)
  (#i: erased nat)
  (#j: erased nat)
  (#gpu_id #nblk #bid:erased int)
  (r:gpu_shmem_array #gpu_id bid a sz)
  (idx : SZ.t)
  (v : a)
  (#s : erased (seq a))
  preserves block_id #gpu_id nblk bid
  requires pure (i <= SZ.v idx /\ SZ.v idx < j)
  requires gpu_pts_to_slice #a #sz r #1.0R i j s
  ensures
    with_pure
      (i <= j /\ j <= sz /\ Seq.length s == (j-i) /\ i <= SZ.v idx /\ SZ.v idx < j)
      (fun _ -> gpu_pts_to_slice #a #sz r #1.0R i j (Seq.upd s (SZ.v idx - i) v))
{
  unfold block_id #gpu_id nblk bid;
  gpu_array_write_loc r idx v;
  fold block_id #gpu_id nblk bid;
}

fn gpu_memcpy_host_to_device'
  (#a:Type u#0)
  {| sized a |}
  (#dst_sz : erased nat)
  (#gpu_id: erased int)
  (dst_garr : gpu_array #gpu_id a dst_sz)
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
    (dst_garr |-> gv)
  ensures
    (dst_garr |-> seq_blit gv dst_off v src_off cnt) **
    pure (Seq.length v == reveal dst_sz)
{ admit () }

fn gpu_memcpy_host_to_device
  (#a:Type u#0)
  {| sized a |}
  (#sz : erased nat)
  (#gpu_id: erased int)
  (dst_garr : gpu_array #gpu_id a sz)
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
    pure (SZ.v cnt == sz /\ (Pulse.Lib.Vec.length src_arr == sz \/ Seq.length v == reveal sz))
  ensures
    (dst_garr |-> v) **
    pure (Seq.length v == reveal sz)
{
  Pulse.Lib.Vec.pts_to_len src_arr;
  gpu_pts_to_slice_ref dst_garr _ _;
  gpu_memcpy_host_to_device' dst_garr 0sz #sz src_arr 0sz cnt;
  assert pure (Seq.equal v (seq_blit gv 0sz v 0sz cnt));
}

(* blit *)
fn gpu_memcpy_device_to_host'
  (#a:Type u#0)
  {| sized a |}
  (#dst_sz : erased nat)
  (dst_arr : vec a)
  (dst_off : SZ.t)
  (#src_sz : erased nat)
  (#gpu_id : erased int)
  (src_garr : gpu_array #gpu_id a src_sz)
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
    pts_to src_garr #f v
  requires
    (dst_arr |-> gv)
  ensures
    (dst_arr |-> seq_blit gv dst_off v src_off cnt) **
    pure (Seq.length v == reveal dst_sz)
{
  admit()
}

fn gpu_memcpy_device_to_host
  (#a:Type u#0)
  {| sized a |}
  (#sz : erased nat)
  (#gpu_id : erased int)
  (dst_arr : vec a)
  (src_garr : gpu_array #gpu_id a sz)
  (cnt : SZ.t)
  (#f : perm)
  (#v : erased (seq a))
  (#gv : erased (seq a))
  preserves
    cpu **
    (src_garr |-> Frac f gv)
  requires
    (dst_arr |-> v) **
    pure (SZ.v cnt == sz /\ (Pulse.Lib.Vec.length dst_arr == sz \/ Seq.length v == reveal sz))
  ensures
    (dst_arr |-> gv) **
    pure (Seq.length gv == reveal sz)
{
  Pulse.Lib.Vec.pts_to_len dst_arr;
  gpu_pts_to_slice_ref src_garr _ _;
  gpu_memcpy_device_to_host' #_ #_ #sz dst_arr 0sz #sz src_garr 0sz cnt;
  assert pure (Seq.equal gv (seq_blit v 0sz gv 0sz cnt));
}

fn gpu_memcpy_device_to_device
  (#a:Type u#0)
  {| sized a |}
  (#sz : erased nat)
  (#gpu1 #gpu2:erased int)
  (dst_arr : gpu_array #gpu1 a sz)
  (src_garr : gpu_array #gpu2 a sz)
  (cnt : SZ.t)
  (#f : perm)
  (#v : erased (seq a))
  (#gv : erased (seq a))
  preserves
    cpu **
    (src_garr |-> Frac f gv)
  requires
    (dst_arr |-> v) **
    pure (SZ.v cnt == sz /\ (Seq.length gv == sz \/ Seq.length v == sz))
  ensures
    (dst_arr |-> gv) **
    pure (Seq.length gv == reveal sz)
{
  admit()
}


ghost
fn gpu_array_slice_1
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array_core a sz)
  (#f : perm)
  (#v : erased (seq a) { Seq.length v == sz })
  requires pts_to arr #f v
  ensures  forall+ (i: natlt sz). gpu_pts_to_cell arr #f i (v @! i)
{
  admit()
}

ghost
fn gpu_array_unslice_1
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array_core a sz)
  (#f : perm)
  (#v : erased (seq a) { Seq.length v == sz })
  requires forall+ (i: natlt sz). gpu_pts_to_cell arr #f i (v @! i)
  ensures  pts_to arr #f v
{
  admit()
}


ghost
fn gpu_slice_concat
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array_core a sz)
  (#[exact (`1.0R)] f : perm)
  (#s1 #s2: erased (seq a))
  (i n m:nat)
  requires gpu_pts_to_slice arr #f i n s1 ** gpu_pts_to_slice arr #f n m s2
  ensures  gpu_pts_to_slice arr #f i m (s1 @+ s2)
{
  admit()
}

ghost
fn gpu_slice_split
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array_core a sz)
  (#[exact (`1.0R)] f : perm)
  (#s1 #s2: erased (seq a))
  (i n m:nat)
  requires gpu_pts_to_slice arr #f i m (s1 @+ s2)
  ensures  gpu_pts_to_slice arr #f i n s1 ** gpu_pts_to_slice arr #f n m s2
{
  admit()
}

ghost
fn gpu_slice_empty_elim
  (#a:Type u#0) (#sz:nat)
  (arr : gpu_array_core a sz)
  (i : nat)
  requires gpu_pts_to_slice arr #'f i i 'v
  ensures  emp
{
  admit()
}

ghost
fn gpu_slice_empty_intro
  (#a:Type u#0) (#sz:nat)
  (arr : gpu_array_core a sz)
  (i : nat)
  requires emp
  ensures  gpu_pts_to_slice arr #'f i i seq![]
{
  admit()
}

ghost
fn gpu_slice_share
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array_core a sz)
  (m n:nat)
  (k: nat { k > 0 })
  (#f : perm)
  requires gpu_pts_to_slice arr #f m n 'v
  ensures
    forall+ (_:natlt k). gpu_pts_to_slice arr #(f /. Real.of_int k) m n 'v
{
  admit()
}

ghost
fn gpu_slice_gather
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array_core a sz)
  (m n:nat)
  (k: nat { k > 0 })
  (#f : perm) // FIXME: if we use 'f, it gets type 'real' instead of 'perm'
  requires
    forall+ (_:natlt k). gpu_pts_to_slice arr #(f /. Real.of_int k) m n 'v
  ensures gpu_pts_to_slice arr #f m n 'v
{
  admit()
}

ghost
fn gpu_slice_pts_to_eq
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array_core a sz)
  (m n:nat)
  (#f1 f2 : perm)
  (#v1 #v2 : seq a)
  requires
    gpu_pts_to_slice arr #f1 m n v1 **
    gpu_pts_to_slice arr #f2 m n v2
  ensures
    gpu_pts_to_slice arr #f1 m n v2 **
    gpu_pts_to_slice arr #f2 m n v2
{
  admit()
}

// val adjacent
//   (#a : Type u#0)
//   (#s1 #s2 : nat)
//   (arr1 : gpu_array_core a s1)
//   (arr2 : gpu_array_core a s2)
//   : slprop

// ghost
// fn gpu_array_cut
//   (#a : Type u#0) {| sized a |}
//   (#sz : nat)
//   (arr : gpu_array_core a sz)
//   (k : SZ.t{ k <= sz })
//   (#s : lseq a sz)
//   requires
//     (arr |-> Frac 'f s)
//   returns
//     p : (gpu_array_core a k & gpu_array_core a (sz - k))
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
//   (arr1 : gpu_array_core a sz1)
//   (arr2 : gpu_array_core a sz2)
//   requires
//     (arr1 |-> Frac 'f 's1) **
//     (arr2 |-> Frac 'f 's2) **
//     adjacent arr1 arr2
//   returns
//     arr : gpu_array_core a (sz1 + sz2)
//   ensures
//     arr |-> Frac 'f ('s1 @+ 's2)
