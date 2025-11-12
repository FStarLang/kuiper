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

let gpu_array (a : Type u#0) (sz : nat) : Type u#0 = A.larray a sz
let loc_id_of_array #a #sz x = loc_id_of_array x
let visibility_of #a #sz x = visibility_of_array x

(* Base address of the GPU array, used to model alignment. This number
is in units of *bytes*, not array elements. *)
let base_address (#a : Type u#0) (#sz : nat) (x : gpu_array a sz)
: GTot nat
= core_base_address x

let mask_of (i j:nat) (n:nat) : prop = i <= n /\ n < j

(* x is the base pointer, this gives permission in [i,j) *)
let gpu_pts_to_slice
  (#a:Type u#0)
  (#sz:nat)
  ([@@@mkey] x:gpu_array a sz)
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

instance is_send_gpu_pts_to_slice
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
= solve

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
{
  unfold gpu_pts_to_slice;
  fold (gpu_pts_to_slice x #f i j v);
}


ghost
fn gpu_pts_to_slice_ref_anywhere
  (#a:Type u#0)
  (#sz:nat)
  (#f : perm)
  (x:gpu_array a sz)
  (i:nat) (j:nat)
  (#v : seq a)
  (#l:loc_id)
  preserves on l (gpu_pts_to_slice x #f i j v)
  requires emp
  ensures  pure (i <= j /\ j <= sz /\ Seq.length v == (j-i) /\ SZ.fits sz)
{
  ghost_impersonate l
     (on l (gpu_pts_to_slice x #f i j v))
     (on l (gpu_pts_to_slice x #f i j v) **
      pure (i <= j /\ j <= sz /\ Seq.length v == (j-i) /\ SZ.fits sz))
    fn () {
      on_elim _;
      gpu_pts_to_slice_ref x i j;
      on_intro (gpu_pts_to_slice x #f i j v);
    }
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
  returns  x : gpu_array a (SZ.v sz)
  ensures
    exists* (s:seq a). 
      on l (x |-> s) **
      pure (Seq.length s == sz)
  ensures
    pure (
      aligned 128 x /\
      visibility_of_array x == vis /\
      loc_id_of_array x == l
    )
{
  fn aux ()
  requires loc l ** dummy
  returns  x : gpu_array a (SZ.v sz)
  ensures loc l
  ensures exists* (s:seq a). 
    on l (gpu_pts_to_array x s) **
    pure (Seq.length s == sz /\
          visibility_of_array x == vis /\
          loc_id_of_array x == l)
  {
    unfold dummy;
    let default_value : a = sized_types_inhabited;
    let x = mask_alloc_with_vis default_value sz vis;
    A.mask_mext x (mask_of 0 (SZ.v sz));
    fold (gpu_pts_to_slice #a #(SZ.v sz) x 0 (SZ.v sz) (Seq.create (SZ.v sz) default_value));
    fold (gpu_pts_to_array _ _);
    on_intro #l (gpu_pts_to_array _ _);
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
    exists* (s:seq a). 
      on gpu_loc (x |-> s) **
      pure (
        Seq.length s == sz /\
        aligned 128 x /\
        is_global_array x
      )
{
  let x = gpu_array_alloc_vis #a sz gpu_loc (gpu_of);
  gpu_of_idem (gpu_id_loc 0);
  x
}

fn gpu_array_free
  (#a:Type u#0)
  (#sz:erased nat)
  (r : gpu_array a sz)
  (#v : erased (seq a))
  preserves cpu
  requires on gpu_loc (r |-> v)
  ensures  emp
{
  fn aux ()
  requires loc gpu_loc
  requires on gpu_loc (r |-> v)
  ensures  loc gpu_loc
  ensures dummy
  {
    on_elim _;
    unfold gpu_pts_to_array;
    unfold gpu_pts_to_slice;
    A.mask_mext r (fun _ -> True);
    A.mask_free r;
    fold dummy;
  };
  __primitive__exec_on_gpu_loc (fun _ -> solve) gpu_loc aux;
  unfold dummy;
}

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
{
  unfold gpu_pts_to_slice;
  let v = A.mask_read r idx;
  fold gpu_pts_to_slice #a #sz r #f i j s;
  v
}

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
      (fun _ -> 
        exists* s'. gpu_pts_to_slice #a #sz r #1.0R i j s' **
            (pure (s' == Seq.upd s (SZ.v idx - i) v)))
{
  unfold gpu_pts_to_slice;
  A.mask_write r idx v;
  fold (gpu_pts_to_slice #a #sz r #1.0R i j (Seq.upd s (SZ.v idx - i) v));
}

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
    exists* s'. 
      on gpu_loc (dst_garr |-> s') **
      pure (s' == seq_blit gv dst_off v src_off cnt)
  ensures
    pure (Seq.length v == reveal dst_sz)
{ admit () } //this is a CUDA primitive


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
{
  Pulse.Lib.Vec.pts_to_len src_arr;
  gpu_pts_to_slice_ref_anywhere dst_garr _ _;
  gpu_memcpy_host_to_device' dst_garr 0sz #sz src_arr 0sz cnt;
  assert pure (Seq.equal v (seq_blit gv 0sz v 0sz cnt));
  with ss. 
    rewrite (on gpu_loc (dst_garr |-> ss))
    as      (on gpu_loc (dst_garr |-> v));
}

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
    pure (s'==seq_blit gv dst_off v src_off cnt /\ Seq.length v == reveal dst_sz)
{ admit() }  //this is a CUDA primitive

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
{
  Pulse.Lib.Vec.pts_to_len dst_arr;
  gpu_pts_to_slice_ref_anywhere src_garr _ _;
  gpu_memcpy_device_to_host' #_ #_ #sz dst_arr 0sz #sz src_garr 0sz cnt;
  assert pure (Seq.equal gv (seq_blit v 0sz gv 0sz cnt));
}


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
{ admit() } //this is a CUDA primitive

ghost
fn gpu_array_slice_1
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
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
  (arr : gpu_array a sz)
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
  (arr : gpu_array a sz)
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
  (arr : gpu_array a sz)
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
  (arr : gpu_array a sz)
  (i : nat)
  requires gpu_pts_to_slice arr #'f i i 'v
  ensures  emp
{
  admit()
}

ghost
fn gpu_slice_empty_intro
  (#a:Type u#0) (#sz:nat)
  (arr : gpu_array a sz)
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
  (arr : gpu_array a sz)
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
{ 
  gpu_slice_share arr m n k; 
  forevery_map #(natlt k)
    (fun _ -> gpu_pts_to_slice arr #(f /. Real.of_int k) m n 'v)
    (fun _ -> exists* v. gpu_pts_to_slice arr #(f /. Real.of_int k) m n v)
    fn _ { () }
}

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
{
  admit()
}

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
{
  admit()
}

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
{
  forevery_natlt_pop k _;
  with vv. assert gpu_pts_to_slice arr #(f /. Real.of_int k) m n vv;
  ghost
  fn aux (_ : natlt (k-1))
    norewrite
    requires
      gpu_pts_to_slice arr #(f /. Real.of_int k) m n vv ** (exists* v. gpu_pts_to_slice arr #(f /. Real.of_int k) m n v)
    ensures
      gpu_pts_to_slice arr #(f /. Real.of_int k) m n vv ** gpu_pts_to_slice arr #(f /. Real.of_int k) m n vv
  {
    gpu_slice_pts_to_eq arr m n (f /. Real.of_int k) #_ #vv;
  };
  forevery_map_extra #(natlt (k-1)) (gpu_pts_to_slice arr #(f /. Real.of_int k) m n vv)
    (fun (_ : natlt (k-1)) -> exists* v. gpu_pts_to_slice arr #(f /. Real.of_int k) m n v)
    (fun (_ : natlt (k-1)) -> gpu_pts_to_slice arr #(f /. Real.of_int k) m n vv)
    aux;
  forevery_natlt_push k _;
  gpu_slice_gather arr m n k;
}



// val adjacent
//   (#a : Type u#0)
//   (#s1 #s2 : nat)
//   (arr1 : gpu_array a s1)
//   (arr2 : gpu_array a s2)
//   : slprop

// ghost
// fn gpu_array_cut
//   (#a : Type u#0) {| sized a |}
//   (#sz : nat)
//   (arr : gpu_array a sz)
//   (k : SZ.t{ k <= sz })
//   (#s : lseq a sz)
//   requires
//     (arr |-> Frac 'f s)
//   returns
//     p : (gpu_array a k & gpu_array a (sz - k))
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
//   (arr1 : gpu_array a sz1)
//   (arr2 : gpu_array a sz2)
//   requires
//     (arr1 |-> Frac 'f 's1) **
//     (arr2 |-> Frac 'f 's2) **
//     adjacent arr1 arr2
//   returns
//     arr : gpu_array a (sz1 + sz2)
//   ensures
//     arr |-> Frac 'f ('s1 @+ 's2)
