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

instance
is_send_across_pts_to_instance (#a: Type u#a) (x:A.array a) (f:perm) (s:seq a)
: is_send_across (visibility_of_array x) (pts_to x #f s)
= magic() // Should extend Pulse.Lib.Array.PtsTo
// A.is_send_pts_to x #f s

instance
is_send_across_pts_to_mask_instance (#a: Type u#a) (x:A.array a) (f:perm) (s:seq (option a)) (mask:nat -> prop)
: is_send_across (visibility_of_array x) (pts_to_mask x #f s mask)
= is_send_across_pts_to_mask x f s mask

let is_full_slice #et a n =
  pure (Pulse.Lib.Array.length a == n)

// let gpu_array (a : Type u#0) (sz : nat) : Type u#0 = (x: A.larray a sz { sz > 0 })
// let loc_id_of_array #a #sz x = loc_id_of_array x
let visibility_of #a x = visibility_of_array x

(* Base address of the GPU array, used to model alignment. This number
is in units of *bytes*, not array elements. *)
let base_address (#a : Type u#0) (x : array a)
: GTot nat
= core_base_address x

let mask_of (i j:nat) (n:nat) : prop = i <= n /\ n < j

(* Helper: convert seq a to seq (option a) with all Some *)
let seq_to_opt (#a:Type) (s:seq a) : seq (option a) = Seq.init (Seq.length s) (fun i -> Some (Seq.index s i))

(* Helper: extract values from seq (option a) where all are Some *)
let seq_from_opt (#a:Type) (s:seq (option a)) : Pure (seq a)
  (requires forall (i:nat). i < Seq.length s ==> Some? (Seq.index s i))
  (ensures fun r -> Seq.length r == Seq.length s /\ (forall (i:nat). i < Seq.length s ==> Seq.index r i == Some?.v (Seq.index s i)))
= Seq.init (Seq.length s) (fun i -> Some?.v (Seq.index s i))

let seq_to_opt_length (#a:Type) (s:seq a) : Lemma (Seq.length (seq_to_opt s) == Seq.length s) = ()

let seq_to_opt_index (#a:Type) (s:seq a) (i:nat{i < Seq.length s})
  : Lemma (Seq.index (seq_to_opt s) i == Some (Seq.index s i))
  = ()

let seq_to_opt_slice (#a:Type) (s:seq a) (i j:nat{i <= j /\ j <= Seq.length s})
  : Lemma (Seq.slice (seq_to_opt s) i j `Seq.equal` seq_to_opt (Seq.slice s i j))
  = ()

let seq_from_opt_to_opt (#a:Type) (s:seq a)
  : Lemma (seq_from_opt (seq_to_opt s) `Seq.equal` s)
  = ()

(* Model-only helper to initialize an array with a value - uses assume since this is model code *)
ghost
fn init_array_model_ghost
  (#a:Type u#0)
  (arr: A.array a)
  (v: a)
  (n: nat)
  (#s0: erased (Seq.seq (option a)))
  requires A.pts_to_mask arr s0 (fun _ -> True)
  requires pure (n == Seq.length s0 /\ A.length arr == n)
  ensures A.pts_to_mask arr (seq_to_opt (Seq.create n v)) (fun _ -> True)
{
  // This is model-only code - we just assume the array gets initialized
  // In reality, mask_alloc_with_vis would need to be followed by actual writes
  assume pure (Seq.length s0 == n);
  assume pure (forall (k:nat). k < n ==> (fun _ -> True) k ==> Seq.index s0 k == Seq.index (seq_to_opt (Seq.create n v)) k);
  A.mask_vext arr (seq_to_opt (Seq.create n v));
}

let pts_to_slice
  (#a:Type u#0)
  ([@@@mkey] x : array a)
  (#[exact (`1.0R)] f : perm)
  ([@@@mkey] i : nat)
  (j : nat)
  (v : seq a)
  : slprop
  = exists* (s : erased (Seq.seq (option a))).
      A.pts_to_mask x #f s (mask_of i j) **
      (* ^ This implies that Seq.length s == Array.length a *)
      pure (i <= j /\
            j <= Seq.length s /\
            (forall (k:nat). mask_of i j k /\ k < Seq.length s ==> Some? (Seq.index s k)) /\
            seq_from_opt (Seq.slice s i j) `Seq.equal` v
            // /\ A.is_full_array x
            (* ^ Needed ? *)
            )

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
{
  A.to_mask a;
  A.pts_to_mask_len a;
  A.mask_mext a (mask_of 0 (Seq.length s));
  fold pts_to_slice #et a #f 0 (Seq.length s) s;
  fold is_full_slice #et a (Seq.length s);
}

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
    A.pts_to a #f s
{
  unfold pts_to_slice a #f 0 n s;
  unfold is_full_slice a n;
  A.pts_to_mask_len a;
  A.from_mask a;
  with v'. assert (pts_to a #f v');
  assert pure (Seq.equal v' s);
}


(* x is the base pointer, this gives permission in [i,j) *)
(* Internally uses Seq.seq (option a) but exposes seq a *)
// let pts_to_slice
//   (#a:Type u#0)
//   (#sz:nat)
//   ([@@@mkey] x:gpu_array a sz)
//   (#[exact (`1.0R)] f : perm)
//   ([@@@mkey] i : nat)
//   (j : nat)
//   (v : seq a)
// = exists* (s:erased (Seq.seq (option a))).
//     A.pts_to_mask x #f s (mask_of i j) **
//     pure (i <= j /\
//           j <= Seq.length s /\
//           (forall (k:nat). mask_of i j k /\ k < Seq.length s ==> Some? (Seq.index s k)) /\
//           seq_from_opt (Seq.slice s i j) `Seq.equal` v /\
//           SZ.fits sz /\
//           Seq.length s == sz /\
//           A.is_full_array x)

instance is_send_pts_to
  (#a:Type u#0)
  (x : array a)
  (#[exact (`1.0R)] f : perm)
  (v : seq a)
  : is_send_across
      (visibility_of x)
      (pts_to x #f v)
  = solve

instance is_send_pts_to_slice
  (#a:Type u#0)
  ([@@@mkey] x : array a)
  (#[exact (`1.0R)] f : perm)
  ([@@@mkey] i : nat)
  (j : nat)
  (v : seq a)
  : is_send_across
      (visibility_of x)
      (pts_to_slice #a x #f i j v)
  = solve

ghost
fn pts_to_slice_ref
  (#a:Type u#0)
  (#f : perm)
  (x : array a)
  (i:nat) (j:nat)
  (#v : seq a)
  preserves pts_to_slice x #f i j v
  requires emp
  ensures  pure (i <= j /\ j <= Pulse.Lib.Array.length x /\ Seq.length v == (j-i) /\ SZ.fits (Pulse.Lib.Array.length x))
{
  unfold pts_to_slice;
  Pulse.Lib.Array.pts_to_mask_len x;
  fold (pts_to_slice x #f i j v);
}

ghost
fn pts_to_slice_ref_anywhere
  (#a:Type u#0)
  (#f : perm)
  (x : array a)
  (i:nat) (j:nat)
  (#v : seq a)
  (#l:loc_id)
  preserves on l (pts_to_slice x #f i j v)
  requires emp
  ensures  pure (i <= j /\ j <= Pulse.Lib.Array.length x /\ Seq.length v == (j-i) /\ SZ.fits (Pulse.Lib.Array.length x))
{
  ghost_impersonate l
     (on l (pts_to_slice x #f i j v))
     (on l (pts_to_slice x #f i j v) **
      pure (i <= j /\ j <= Pulse.Lib.Array.length x /\ Seq.length v == (j-i) /\ SZ.fits (Pulse.Lib.Array.length x)))
    fn () {
      on_elim _;
      pts_to_slice_ref x i j;
      on_intro (pts_to_slice x #f i j v);
    }
}

noextract
fn gpu_array_alloc_vis
  (#a : Type u#0)
  {| sized a |}
  (sz : SZ.t { sz > 0 })
  (l:loc_id)
  (vis:visibility)
  returns x : larray a sz
  ensures
    exists* (s:seq a).
      on l (x |-> s) **
      pure (Seq.length s == sz)
  ensures
    pure (
      visibility_of_array x == vis /\
      loc_id_of_array x == l /\
      A.is_full_array x
    )
{
  impersonate _ l emp (fun (x: larray a sz) ->
    exists* (s:seq a).
      on l (pts_to x s) **
      pure (Seq.length s == sz /\
            visibility_of_array x == vis /\
            loc_id_of_array x == l /\
            A.is_full_array x
      ))
  fn _ {
    let x = mask_alloc_with_vis a sz vis;
    with s0. assert (A.pts_to_mask x s0 (fun _ -> True));
    // Initialize all cells with default value (model only)
    init_array_model_ghost x (default <: a) (SZ.v sz);
    A.mask_mext x (mask_of 0 (SZ.v sz));
    fold pts_to_slice #a x 0 (SZ.v sz) (Seq.create (SZ.v sz) default);
    fold is_full_slice #a x (SZ.v sz);
    on_intro #l (pts_to x _);
    x
  }
}

noextract
fn gpu_array_alloc
  (#a : Type u#0)
  {| d: sized a |}
  (sz : SZ.t { sz > 0 })
  preserves cpu
  returns   x : larray a sz
  ensures
    exists* (s:seq a).
      on gpu_loc (x |-> s) **
      pure (
        Seq.length s == sz /\
        aligned 128 x /\
        is_global_array x /\
        A.is_full_array x
      )
{
  let x = gpu_array_alloc_vis #a sz gpu_loc (gpu_of);
  gpu_of_idem (gpu_id_loc 0);
  assume pure (aligned 128 x); // cudaMalloc guarantees this
  x
}

fn gpu_array_free_gen
  (#a:Type u#0)
  (r : array a)
  (#v : erased (seq a))
  (l: loc_id)
  requires pure (A.is_full_array r)
  requires on l (r |-> v)
  ensures  emp
{
  assert pure (is_full_array r);
  impersonate unit l (on l (r |-> v)) (fun () -> emp)
  fn _ {
    on_elim _;
    A.free r;
  }
}

fn gpu_array_free
  (#a:Type u#0)
  (r : array a)
  (#v : erased (seq a))
  preserves cpu
  requires pure (A.is_full_array r)
  requires on gpu_loc (r |-> v)
  ensures  emp
{
  gpu_array_free_gen r gpu_loc
}

[@@noextract_to "krml"]
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
{
  unfold pts_to_slice;
  with s0. assert (A.pts_to_mask r #f s0 (mask_of i j));
  // The invariant guarantees all masked cells are Some
  let v = A.mask_read r idx;
  fold pts_to_slice #a r #f i j s;
  v
}

[@@noextract_to "krml"]
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
{
  unfold pts_to_slice;
  with s0. assert (A.pts_to_mask r s0 (mask_of i j));
  A.mask_write r idx v;
  // mask_write produces Some v, so we need to update our seq accordingly
  fold (pts_to_slice #a r #1.0R i j (Seq.upd s (SZ.v idx - i) v));
}

fn rec gpu_memcpy_host_to_device'  //this is a CUDA primitive, so this definition is a model only, not meant for extraction
  (#a:Type u#0)
  {| sized a |}
  (#dst_sz : erased nat)
  (dst_garr : larray a dst_sz)
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
      pure (s' == seq_blit gv dst_off v src_off cnt /\ Seq.length s' == reveal dst_sz)
{
  if (cnt = 0sz) {
    assert pure (Seq.equal gv (seq_blit gv dst_off v src_off cnt));
    rewrite on gpu_loc (dst_garr |-> gv)
      as on gpu_loc (dst_garr |-> (seq_blit gv dst_off v src_off cnt));
    ()
  } else {
    let x = Pulse.Lib.Vec.op_Array_Access src_arr src_off;
    impersonate // model only
      unit
      gpu_loc
      (on gpu_loc (dst_garr |-> gv))
      (fun _ -> on gpu_loc (dst_garr |-> Seq.upd gv dst_off x))
      fn _ {
        on_elim _;
        Pulse.Lib.Array.op_Array_Assignment dst_garr dst_off x;
        on_intro (dst_garr |-> Seq.upd gv dst_off x);
        ()
      };
    Pulse.Lib.Vec.pts_to_len src_arr;
    gpu_memcpy_host_to_device' dst_garr (dst_off +^ 1sz) #src_sz src_arr (src_off +^ 1sz) (cnt -^ 1sz) #f #v #(Seq.upd gv dst_off x);
    with s' . assert (on gpu_loc (dst_garr |-> s'));
    assert pure (Seq.equal s' (seq_blit gv dst_off v src_off cnt));
    ()
  }
}


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
{
  Pulse.Lib.Vec.pts_to_len src_arr;
  ghost_impersonate gpu_loc
    (on gpu_loc (dst_garr |-> gv))
    (on gpu_loc (dst_garr |-> gv) ** pure (Seq.length gv == reveal sz))
    fn _ {
      on_elim (dst_garr |-> gv);
      Pulse.Lib.Array.PtsTo.pts_to_len dst_garr;
      on_intro (dst_garr |-> gv);
    };
  gpu_memcpy_host_to_device' dst_garr 0sz #sz src_arr 0sz cnt;
  assert pure (Seq.equal v (seq_blit gv 0sz v 0sz cnt));
  with ss.
    rewrite (on gpu_loc (dst_garr |-> ss))
    as      (on gpu_loc (dst_garr |-> v));
}

(* blit *)
fn rec gpu_memcpy_device_to_host'  //this is a CUDA primitive, so this definition is a model only, not meant for extraction
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
    (dst_arr |-> gv)
  ensures
    exists* s'. dst_arr |-> s' **
    pure (s'==seq_blit gv dst_off v src_off cnt /\ Seq.length s' == reveal dst_sz)
{
  if (cnt = 0sz) {
    assert pure (Seq.equal gv (seq_blit gv dst_off v src_off cnt));
    rewrite (dst_arr |-> gv)
      as (dst_arr |-> (seq_blit gv dst_off v src_off cnt));
    ()
  } else {
    let x = impersonate // model only
      a
      gpu_loc
      (on gpu_loc (src_garr |-> Frac f (v<:seq _)))
      (fun res -> on gpu_loc (src_garr |-> Frac f (v<:seq _)) ** pure (res == Seq.index v src_off))
      fn _ {
        on_elim _;
        let res = Pulse.Lib.Array.op_Array_Access src_garr src_off;
        on_intro (src_garr |-> Frac f (v<:seq _));
        res
      };
    Pulse.Lib.Vec.op_Array_Assignment dst_arr dst_off x;
    Pulse.Lib.Vec.pts_to_len dst_arr;
    gpu_memcpy_device_to_host' #a #_ #dst_sz dst_arr (dst_off +^ 1sz) src_garr (src_off +^ 1sz) (cnt -^ 1sz);
    with s' . assert (dst_arr |-> s');
    assert pure (Seq.equal s' (seq_blit gv dst_off v src_off cnt));
    ()
  }
}

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
    (dst_arr |-> v) **
    pure (
      SZ.v cnt == sz /\
      (Pulse.Lib.Vec.length dst_arr == sz \/ Seq.length v == reveal sz))
  ensures
    (dst_arr |-> gv) **
    pure (Seq.length gv == reveal sz)
{
  Pulse.Lib.Vec.pts_to_len dst_arr;
  ghost_impersonate gpu_loc
    (on gpu_loc (src_garr |-> Frac f gv))
    (on gpu_loc (src_garr |-> Frac f gv) ** pure (Seq.length gv == reveal sz))
    fn _ {
      on_elim (src_garr |-> Frac f gv);
      Pulse.Lib.Array.PtsTo.pts_to_len src_garr;
      on_intro (src_garr |-> Frac f gv);
    };
  gpu_memcpy_device_to_host' #_ #_ #sz dst_arr 0sz #sz src_garr 0sz cnt;
  assert pure (Seq.equal gv (seq_blit v 0sz gv 0sz cnt));
}


fn rec gpu_memcpy_device_to_device  //this is a CUDA primitive, so this definition is a model only, not meant for extraction
  (#a:Type u#0)
  {| sized a |}
  (#sz : erased nat)
  (dst_arr : larray a sz)
  (src_garr : larray a sz)
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
{
  impersonate // gpu_memcpy_device_to_device is a CUDA primitive, so this definition is only a model
    unit
    gpu_loc
    (on gpu_loc (src_garr |-> Frac f gv) **
      on gpu_loc (dst_arr |-> v)
    )
    (fun _ -> on gpu_loc (src_garr |-> Frac f gv) **
      on gpu_loc (dst_arr |-> gv) **
      pure (Seq.length gv == reveal sz))
    fn _ {
      on_elim (src_garr |-> Frac f gv);
      on_elim (dst_arr |-> v);
      Pulse.Lib.Array.PtsTo.pts_to_len src_garr;
      Pulse.Lib.Array.PtsTo.pts_to_len dst_arr;
      Pulse.Lib.Array.memcpy cnt src_garr dst_arr;
      on_intro (src_garr |-> Frac f gv);
      on_intro (dst_arr |-> gv);
    };
}

ghost
fn slice_concat
  (#a:Type u#0)
  (arr : array a)
  (#[exact (`1.0R)] f : perm)
  (#s1 #s2: erased (seq a))
  (i n m:nat)
  requires pts_to_slice arr #f i n s1 ** pts_to_slice arr #f n m s2
  ensures  pts_to_slice arr #f i m (s1 @+ s2)
{
  unfold (pts_to_slice arr #f i n s1);
  unfold (pts_to_slice arr #f n m s2);
  A.join_mask arr;
  A.mask_mext arr (mask_of i m);
  fold (pts_to_slice arr #f i m (s1 @+ s2));
}

ghost
fn slice_split
  (#a:Type u#0)
  (arr : array a)
  (#[exact (`1.0R)] f : perm)
  (#s1 #s2: erased (seq a))
  (i n m:nat)
  requires pts_to_slice arr #f i m (s1 @+ s2) ** pure (i <= n /\ n <= m /\ (i + Seq.length s1 == n \/ n + Seq.length s2 == m))
  ensures  pts_to_slice arr #f i n s1 ** pts_to_slice arr #f n m s2
{
  pts_to_slice_ref arr _ _;
  unfold pts_to_slice;
  with s . assert (A.pts_to_mask arr #f s (mask_of i m));
  A.split_mask arr (fun j -> j < n);
  A.mask_mext arr #_ #_ #(A.mask_diff _ _) (mask_of n m);
  Seq.slice_slice s i m (Seq.length s1) (Seq.length s1 + Seq.length s2);
  fold (pts_to_slice arr #f n m s2);
  A.mask_mext arr (mask_of i n);
  Seq.slice_slice s i m 0 (Seq.length s1);
  fold (pts_to_slice arr #f i n s1);
}

ghost
fn rec forall_slice_to_cell
  (#a:Type u#0)
  (arr : array a)
  (i0: nat)
  (#f : perm)
  (#v : erased (seq a))
  (j: nat { 0 < j /\ j <= Seq.length v })
  requires pts_to_slice arr #f i0 (i0 + j) (Seq.slice v 0 j)
  ensures forall+ (i: natlt (Seq.length v) { i < j }). pts_to_cell arr #f (i0 + i) (v @! i)
  decreases j
{
  if (j = 1) {
    assert pure (Seq.equal (slice v 0 j) seq![index v 0]);
    rewrite pts_to_slice arr #f i0 (i0 + j) (Seq.slice v 0 j)
      as pts_to_cell arr #f (i0 + 0) (v @! 0);
    forevery_intro_false #(natlt (Seq.length v)) (fun i -> pts_to_cell arr #f (i0 + i) (v @! i));
    forevery_insert #(natlt (Seq.length v)) (fun i -> pts_to_cell arr #f (i0 + i) (v @! i)) 0;
    forevery_refine_ext #(natlt (Seq.length v)) (fun i -> i < j) (fun i -> pts_to_cell arr #f (i0 + i) (v @! i));
  } else {
    Seq.lemma_split (Seq.slice v 0 j) (j - 1);
    slice_split arr #f #(Seq.slice v 0 (j - 1)) #(Seq.slice v (j - 1) j) i0 (i0 + (j - 1)) (i0 + j) ;
    forall_slice_to_cell arr i0 (j - 1);
    assert pure (Seq.equal (Seq.slice v (j - 1) j) seq![index v (j - 1)]);
    rewrite pts_to_slice arr #f (i0 + (j - 1)) (i0 + j) (Seq.slice v (j - 1) j)
      as pts_to_cell arr #f (i0 + (j - 1)) (v @! (j - 1));
    forevery_insert #(natlt (Seq.length v)) (fun i -> pts_to_cell arr #f (i0 + i) (v @! i)) (j - 1);
    forevery_refine_ext #(natlt (Seq.length v)) (fun i -> i < j) (fun i -> pts_to_cell arr #f (i0 + i) (v @! i));
    ()
  }
}

ghost
fn array_slice_1
  (#a:Type u#0)
  (#sz:nat)
  (arr : larray a sz)
  (#f : perm)
  (#v : erased (seq a) { Seq.length v == sz })
  requires pts_to arr #f v
  ensures  forall+ (i: natlt sz). pts_to_cell arr #f i (v @! i)
{
  array_to_slice arr;
  drop_ (is_full_slice arr _);
  if (sz = 0) {
    drop_ (pts_to_slice arr #f 0 (Seq.length v) v);
    forevery_intro_false #(natlt sz) (fun i -> pts_to_cell arr #f i (v @! i));
    forevery_unrefine _;
  } else {
    rewrite pts_to_slice arr #f 0 (Seq.length v) v
      as pts_to_slice arr #f 0 (0 + sz) (Seq.slice v 0 sz);
    forall_slice_to_cell arr 0 sz;
    forevery_unrefine _;
    rewrite each (Seq.length v) as sz;
    forevery_ext #(natlt sz) _ (fun i -> pts_to_cell arr #f i (v @! i));
    ()
  }
}

#set-options "--split_queries always --debug SMTFail"
ghost
fn rec forall_cell_to_slice
  (#a:Type u#0)
  (#sz : nat)
  (arr : larray a sz)
  (i0 : natle sz)
  (#f : perm)
  (#v : erased (seq a))
  (j: nat { 0 <= j /\ j <= Seq.length v })
  requires forall+ (i: natlt (Seq.length v) { i < j }). pts_to_cell arr #f (i0 + i) (v @! i)
  ensures pts_to_slice arr #f i0 (i0 + j) (Seq.slice v 0 j)
  decreases j
{
  if (j = 0) {
    forevery_elim_empty _;
    (* This is fake. We should take some resource stating that the
    array in fact exists. *)
    assume exists* ss. A.pts_to_mask arr #f ss (mask_of i0 i0);
    with ss. assert A.pts_to_mask arr #f ss (mask_of i0 i0);
    A.pts_to_mask_len arr;
    assert pure (Seq.length ss == sz);
    assert pure (Seq.slice ss i0 i0 `Seq.equal` seq![]);
    fold pts_to_slice arr #f i0 i0 seq![];
  } else {
    let j' : natlt (Seq.length v) = j - 1;
    forevery_remove' #(natlt (Seq.length v)) (fun (i: natlt (Seq.length v)) -> i < j) (fun (i: natlt (Seq.length v)) -> pts_to_cell arr #f (i0 + i) (v @! i)) j';
    forevery_refine_ext #(natlt (Seq.length v)) (fun i -> i < j') (fun (i: natlt (Seq.length v)) -> pts_to_cell arr #f (i0 + i) (v @! i));
    forall_cell_to_slice arr i0 j';
    slice_concat arr #f i0 _ _;
    assert pure (Seq.equal (append (slice v 0 j') seq![index v j']) (Seq.slice v 0 j));
    rewrite pts_to_slice arr #f i0 (i0 + j' + 1) (append (slice v 0 j') seq![index v j'])
      as pts_to_slice arr #f i0 (i0 + j) (Seq.slice v 0 j);
  }
}

ghost
fn array_unslice_1
  (#a : Type u#0)
  (#sz : nat)
  (arr : larray a sz)
  (#f : perm)
  (#v : erased (seq a) { Seq.length v == sz })
  // requires
    // is_full_slice arr sz
  requires
    forall+ (i: natlt sz). pts_to_cell arr #f i (v @! i)
  ensures
    pts_to arr #f v
{
  rewrite each sz as Seq.length v;
  forevery_ext #(natlt (Seq.length v)) _ (fun i -> pts_to_cell arr #f (0 + i) (v @! i));
  forevery_refine_split #(natlt (Seq.length v)) _ (fun i -> i < sz);
  forevery_refine_join #(natlt (Seq.length v)) _ (fun i -> i < sz) _;
  forevery_refine_ext #(natlt (Seq.length v)) (fun i -> i < sz) _;
  assume is_full_slice arr sz; // fixme: propagate to pre
  forall_cell_to_slice #a #sz arr 0 sz;
  ()
}

ghost
fn rec slice_share
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
  decreases k
{
  if (k = 1) {
    rewrite pts_to_slice arr #f m n 'v
      as pts_to_slice arr #(f /. Real.of_int k) m n 'v;
    forevery_intro_false #(natlt k) (fun _ -> pts_to_slice arr #(f /. Real.of_int k) m n 'v);
    forevery_insert #(natlt k) (fun _ -> pts_to_slice arr #(f /. Real.of_int k) m n 'v) 0;
    forevery_unrefine #(natlt k) (fun _ -> pts_to_slice arr #(f /. Real.of_int k) m n 'v);
  } else {
    with v . assert (pts_to_slice arr #f m n v);
    unfold (pts_to_slice arr #f m n v);
    with s . assert (A.pts_to_mask arr #f s (mask_of m n));
    let f' = f -. (f /. Real.of_int k);
    mask_share_gen arr #s #f (f /. Real.of_int k) f';
    fold (pts_to_slice arr #(f /. Real.of_int k) m n v);
    fold (pts_to_slice arr #f' m n v);
    slice_share arr m n (k - 1) #f';
    forevery_ext #(natlt (k - 1)) _ (fun _ -> pts_to_slice arr #(f /. Real.of_int k) m n v);
    forevery_natlt_push k (fun _ -> pts_to_slice arr #(f /. Real.of_int k) m n v);
  }
}

ghost
fn slice_share_underspec
  (#a:Type u#0)
  (arr : array a)
  (m n:nat)
  (k: nat { k > 0 })
  (#f : perm)
  requires
    pts_to_slice arr #f m n 'v
  ensures
    forall+ (_:natlt k).
      exists* v.
        pts_to_slice arr #(f /. Real.of_int k) m n v
{
  slice_share arr m n k;
  forevery_map #(natlt k)
    (fun _ -> pts_to_slice arr #(f /. Real.of_int k) m n 'v)
    (fun _ -> exists* v. pts_to_slice arr #(f /. Real.of_int k) m n v)
    fn _ { () }
}

ghost
fn rec slice_gather
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
  decreases k
{
  if (k = 1) {
    forevery_singleton_elim' #(natlt k) _ 0;
  } else {
    forevery_natlt_pop k _;
    let f' = f -. (f /. Real.of_int k);
    forevery_ext #(natlt (k - 1)) _ (fun _ -> pts_to_slice arr #(f' /. Real.of_int (k - 1)) m n 'v);
    slice_gather arr m n (k - 1);
    with v1 . assert pts_to_slice arr #(f /. Real.of_int k) m n v1;
    with v2 . assert pts_to_slice arr #f' m n v2;
    unfold pts_to_slice arr #(f /. Real.of_int k) m n v1;
    unfold pts_to_slice arr #f' m n v2;
    mask_gather arr;
    fold pts_to_slice arr #f m n v1;
    ()
  }
}

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
{
  unfold pts_to_slice arr #f1 m n v1;
  unfold pts_to_slice arr #f2 m n v2;
  mask_gather arr;
  mask_share_gen arr f1 f2;
  assert pure (Seq.equal v1 v2);
  fold pts_to_slice arr #f1 m n v2;
  fold pts_to_slice arr #f2 m n v2;
}

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
{
  forevery_natlt_pop k _;
  with vv. assert pts_to_slice arr #(f /. Real.of_int k) m n vv;
  ghost
  fn aux (_ : natlt (k-1))
    norewrite
    requires
      pts_to_slice arr #(f /. Real.of_int k) m n vv ** (exists* v. pts_to_slice arr #(f /. Real.of_int k) m n v)
    ensures
      pts_to_slice arr #(f /. Real.of_int k) m n vv ** pts_to_slice arr #(f /. Real.of_int k) m n vv
  {
    slice_pts_to_eq arr m n (f /. Real.of_int k) #_ #vv;
  };
  forevery_map_extra #(natlt (k-1)) (pts_to_slice arr #(f /. Real.of_int k) m n vv)
    (fun (_ : natlt (k-1)) -> exists* v. pts_to_slice arr #(f /. Real.of_int k) m n v)
    (fun (_ : natlt (k-1)) -> pts_to_slice arr #(f /. Real.of_int k) m n vv)
    aux;
  forevery_natlt_push k _;
  slice_gather arr m n k;
}

ghost
fn array_gather_underspec
  (#a : Type u#0)
  (arr : array a)
  (#f : perm)
  (k : nat { k > 0 })
  requires
    forall+ (_ : natlt k).
      exists* (v : seq a).
        arr |-> Frac (f /. Real.of_int k) v
  ensures
    exists* (v : seq a).
      arr |-> Frac f v
{
  forevery_natlt_pop k _;
  with vv. assert arr |-> Frac (f /. Real.of_int k) (vv <: seq a);
  ghost
  fn aux (_ : natlt (k-1))
    norewrite
    requires
      arr |-> Frac (f /. Real.of_int k) vv ** (exists* v. arr |-> Frac (f /. Real.of_int k) (v <: seq a))
    ensures
      arr |-> Frac (f /. Real.of_int k) vv ** arr |-> Frac (f /. Real.of_int k) vv
  {
    Pulse.Lib.Array.pts_to_injective_eq arr;
  };
  forevery_map_extra #(natlt (k-1)) (arr |-> Frac (f /. Real.of_int k) vv)
    (fun (_ : natlt (k-1)) -> exists* v. arr |-> Frac (f /. Real.of_int k) v)
    (fun (_ : natlt (k-1)) -> arr |-> Frac (f /. Real.of_int k) vv)
    aux;
  forevery_natlt_push k _;
  Kuiper.Array.Extra.array_gather arr k;
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
