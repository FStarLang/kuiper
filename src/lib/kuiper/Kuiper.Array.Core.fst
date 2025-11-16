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
is_send_across_pts_to_mask_instance (#a: Type u#a) (x:A.array a) (f:perm) (s:seq a) (mask:nat -> prop)
: is_send_across (visibility_of_array x) (pts_to_mask x #f s mask)
= is_send_across_pts_to_mask x f s mask


let gpu_array (a : Type u#0) (sz : nat) : Type u#0 = (x: A.larray a sz { sz > 0 })
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

noextract
fn gpu_array_alloc_vis
  (#a : Type u#0)
  {| sized a |}
  (sz : SZ.t { sz > 0 })
  (l:loc_id)
  (vis:visibility)
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
  impersonate _ l emp (fun (x: gpu_array a (SZ.v sz)) ->
    exists* (s:seq a).
      on l (gpu_pts_to_array x s) **
      pure (Seq.length s == sz /\
            visibility_of_array x == vis /\
            loc_id_of_array x == l))
  fn _ {
    let x = mask_alloc_with_vis (default <: a) sz vis;
    A.mask_mext x (mask_of 0 (SZ.v sz));
    fold (gpu_pts_to_slice #a #(SZ.v sz) x 0 (SZ.v sz) (Seq.create (SZ.v sz) default));
    fold (gpu_pts_to_array _ _);
    on_intro #l (gpu_pts_to_array _ _);
    x
  }
}


noextract
fn gpu_array_alloc
  (#a : Type u#0)
  {| d: sized a |}
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
{
  let x = gpu_array_alloc_vis #a sz gpu_loc (gpu_of);
  gpu_of_idem (gpu_id_loc 0);
  x
}

fn gpu_array_free_gen // needed for Kuiper.Kernel.Sync.free_c_shmems (to model launch_kernel_full_sync), to model allocation and liberation of per-block shared memory by the GPU runtime.
  (#a:Type u#0)
  (#sz:erased nat)
  (r : gpu_array a sz)
  (#v : erased (seq a))
  (l: loc_id)
  requires on l (r |-> v)
  ensures  emp
{
  impersonate unit l (on l (r |-> v)) (fun _ -> emp) fn _ {
    on_elim _;
    unfold gpu_pts_to_array;
    unfold gpu_pts_to_slice;
    A.mask_mext r (fun _ -> True);
    A.mask_free r;
  };
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
  gpu_array_free_gen r gpu_loc
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

fn rec gpu_memcpy_host_to_device'  //this is a CUDA primitive, so this definition is a model only, not meant for extraction
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
        gpu_array_write dst_garr dst_off x;
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
fn rec gpu_memcpy_device_to_host'  //this is a CUDA primitive, so this definition is a model only, not meant for extraction
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
        let res = gpu_array_read src_garr src_off;
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


fn rec gpu_memcpy_device_to_device  //this is a CUDA primitive, so this definition is a model only, not meant for extraction
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
      unfold (gpu_pts_to_slice src_garr #f 0 sz gv);
      unfold (gpu_pts_to_slice dst_arr 0 sz v);
      Pulse.Lib.Array.PtsTo.from_mask src_garr;
      Pulse.Lib.Array.PtsTo.from_mask dst_arr;
      Pulse.Lib.Array.pts_to_len src_garr;
      Pulse.Lib.Array.pts_to_len dst_arr;
      Pulse.Lib.Array.memcpy cnt src_garr dst_arr;
      Pulse.Lib.Array.pts_to_len dst_arr;
      Pulse.Lib.Array.PtsTo.to_mask src_garr;
      Pulse.Lib.Array.mask_mext src_garr (mask_of 0 sz);
      fold (gpu_pts_to_slice src_garr #f 0 sz gv);
      on_intro (src_garr |-> Frac f gv);
      Pulse.Lib.Array.PtsTo.to_mask dst_arr;
      Pulse.Lib.Array.mask_mext dst_arr (mask_of 0 sz);
      fold (gpu_pts_to_slice dst_arr 0 sz gv);
      on_intro (dst_arr |-> gv);
    };
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
  unfold (gpu_pts_to_slice arr #f i n s1);
  unfold (gpu_pts_to_slice arr #f n m s2);
  A.join_mask arr;
  A.mask_mext arr (mask_of i m);
  fold (gpu_pts_to_slice arr #f i m (s1 @+ s2));
}

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
{
  gpu_pts_to_slice_ref arr _ _;
  unfold gpu_pts_to_slice;
  with s . assert (A.pts_to_mask arr #f s (mask_of i m));
  A.split_mask arr (fun j -> j < n);
  A.mask_mext arr #_ #_ #(A.mask_diff _ _) (mask_of n m);
  Seq.slice_slice s i m (Seq.length s1) (Seq.length s1 + Seq.length s2);
  fold (gpu_pts_to_slice arr #f n m s2);
  A.mask_mext arr (mask_of i n);
  Seq.slice_slice s i m 0 (Seq.length s1);
  fold (gpu_pts_to_slice arr #f i n s1);
}

ghost
fn rec gpu_forall_slice_to_cell
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (i0: nat)
  (#f : perm)
  (#v : erased (seq a))
  (j: nat { 0 < j /\ j <= Seq.length v })
  requires gpu_pts_to_slice arr #f i0 (i0 + j) (Seq.slice v 0 j)
  ensures forall+ (i: natlt (Seq.length v) { i < j }). gpu_pts_to_cell arr #f (i0 + i) (v @! i)
  decreases j
{
  if (j = 1) {
    assert pure (Seq.equal (slice v 0 j) seq![index v 0]);
    rewrite gpu_pts_to_slice arr #f i0 (i0 + j) (Seq.slice v 0 j)
      as gpu_pts_to_cell arr #f (i0 + 0) (v @! 0);
    forevery_intro_false #(natlt (Seq.length v)) (fun i -> gpu_pts_to_cell arr #f (i0 + i) (v @! i));
    forevery_insert #(natlt (Seq.length v)) (fun i -> gpu_pts_to_cell arr #f (i0 + i) (v @! i)) 0;
    forevery_refine_ext #(natlt (Seq.length v)) (fun i -> i < j) (fun i -> gpu_pts_to_cell arr #f (i0 + i) (v @! i));
  } else {
    Seq.lemma_split (Seq.slice v 0 j) (j - 1);
    gpu_slice_split arr #f #(Seq.slice v 0 (j - 1)) #(Seq.slice v (j - 1) j) i0 (i0 + (j - 1)) (i0 + j) ;
    gpu_forall_slice_to_cell arr i0 (j - 1);
    assert pure (Seq.equal (Seq.slice v (j - 1) j) seq![index v (j - 1)]);
    rewrite gpu_pts_to_slice arr #f (i0 + (j - 1)) (i0 + j) (Seq.slice v (j - 1) j)
      as gpu_pts_to_cell arr #f (i0 + (j - 1)) (v @! (j - 1));
    forevery_insert #(natlt (Seq.length v)) (fun i -> gpu_pts_to_cell arr #f (i0 + i) (v @! i)) (j - 1);
    forevery_refine_ext #(natlt (Seq.length v)) (fun i -> i < j) (fun i -> gpu_pts_to_cell arr #f (i0 + i) (v @! i));
    ()
  }
}

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
  rewrite pts_to arr #f v
    as gpu_pts_to_slice arr #f 0 (0 + sz) (Seq.slice v 0 sz);
  gpu_forall_slice_to_cell arr 0 sz;
  forevery_unrefine _;
  rewrite each (Seq.length v) as sz;
  forevery_ext #(natlt sz) _ (fun i -> gpu_pts_to_cell arr #f i (v @! i));
  ()
}

ghost
fn rec gpu_forall_cell_to_slice
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (i0: nat)
  (#f : perm)
  (#v : erased (seq a))
  (j: nat { 0 < j /\ j <= Seq.length v })
  requires forall+ (i: natlt (Seq.length v) { i < j }). gpu_pts_to_cell arr #f (i0 + i) (v @! i)
  ensures gpu_pts_to_slice arr #f i0 (i0 + j) (Seq.slice v 0 j)
  decreases j
{
  let j' : natlt (Seq.length v) = j - 1;
  if (j' = 0) {
    forevery_singleton_elim' #(i: natlt (Seq.length v) { i < j }) _ j';
    assert pure (Seq.equal seq![index v j'] (Seq.slice v 0 j));
    rewrite gpu_pts_to_slice arr #f (i0 + j') (i0 + j' + 1) seq![index v j']
      as gpu_pts_to_slice arr #f i0 (i0 + j) (Seq.slice v 0 j);
    ()
  } else {
    forevery_remove' #(natlt (Seq.length v)) (fun (i: natlt (Seq.length v)) -> i < j) (fun (i: natlt (Seq.length v)) -> gpu_pts_to_cell arr #f (i0 + i) (v @! i)) j';
    forevery_refine_ext #(natlt (Seq.length v)) (fun i -> i < j') (fun (i: natlt (Seq.length v)) -> gpu_pts_to_cell arr #f (i0 + i) (v @! i));
    gpu_forall_cell_to_slice arr i0 j';
    gpu_slice_concat arr #f i0 _ _;
    assert pure (Seq.equal (append (slice v 0 j') seq![index v j']) (Seq.slice v 0 j));
    rewrite gpu_pts_to_slice arr #f i0 (i0 + j' + 1) (append (slice v 0 j') seq![index v j'])
      as gpu_pts_to_slice arr #f i0 (i0 + j) (Seq.slice v 0 j);
  }
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
  rewrite each (sz) as Seq.length v;
  forevery_ext #(natlt (Seq.length v)) _ (fun i -> gpu_pts_to_cell arr #f (0 + i) (v @! i));
  forevery_refine_split #(natlt (Seq.length v)) _ (fun i -> i < sz);
  forevery_refine_join #(natlt (Seq.length v)) _ (fun i -> i < sz) _;
  forevery_refine_ext #(natlt (Seq.length v)) (fun i -> i < sz) _;
  gpu_forall_cell_to_slice arr 0 sz;
  ()
}

ghost
fn rec gpu_slice_share
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (m n:nat)
  (k: nat { k > 0 })
  (#f : perm)
  requires gpu_pts_to_slice arr #f m n 'v
  ensures
    forall+ (_:natlt k). gpu_pts_to_slice arr #(f /. Real.of_int k) m n 'v
  decreases k
{
  if (k = 1) {
    rewrite gpu_pts_to_slice arr #f m n 'v
      as gpu_pts_to_slice arr #(f /. Real.of_int k) m n 'v;
    forevery_intro_false #(natlt k) (fun _ -> gpu_pts_to_slice arr #(f /. Real.of_int k) m n 'v);
    forevery_insert #(natlt k) (fun _ -> gpu_pts_to_slice arr #(f /. Real.of_int k) m n 'v) 0;
    forevery_unrefine #(natlt k) (fun _ -> gpu_pts_to_slice arr #(f /. Real.of_int k) m n 'v);
  } else {
    with v . assert (gpu_pts_to_slice arr #f m n v);
    unfold (gpu_pts_to_slice arr #f m n v);
    with s . assert (A.pts_to_mask arr #f s (mask_of m n));
    let f' = f -. (f /. Real.of_int k);
    mask_share_gen arr #s #f (f /. Real.of_int k) f';
    fold (gpu_pts_to_slice arr #(f /. Real.of_int k) m n v);
    fold (gpu_pts_to_slice arr #f' m n v);
    gpu_slice_share arr m n (k - 1) #f';
    forevery_ext #(natlt (k - 1)) _ (fun _ -> gpu_pts_to_slice arr #(f /. Real.of_int k) m n v);
    forevery_natlt_push k (fun _ -> gpu_pts_to_slice arr #(f /. Real.of_int k) m n v);
  }
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
fn rec gpu_slice_gather
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (m n:nat)
  (k: nat { k > 0 })
  (#f : perm) // FIXME: if we use 'f, it gets type 'real' instead of 'perm'
  requires
    forall+ (_:natlt k). gpu_pts_to_slice arr #(f /. Real.of_int k) m n 'v
  ensures gpu_pts_to_slice arr #f m n 'v
  decreases k
{
  if (k = 1) {
    forevery_singleton_elim' #(natlt k) _ 0;
  } else {
    forevery_natlt_pop k _;
    let f' = f -. (f /. Real.of_int k);
    forevery_ext #(natlt (k - 1)) _ (fun _ -> gpu_pts_to_slice arr #(f' /. Real.of_int (k - 1)) m n 'v);
    gpu_slice_gather arr m n (k - 1);
    with v1 . assert gpu_pts_to_slice arr #(f /. Real.of_int k) m n v1;
    with v2 . assert gpu_pts_to_slice arr #f' m n v2;
    unfold gpu_pts_to_slice arr #(f /. Real.of_int k) m n v1;
    unfold gpu_pts_to_slice arr #f' m n v2;
    mask_gather arr;
    fold gpu_pts_to_slice arr #f m n v1;
    ()
  }
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
  unfold gpu_pts_to_slice arr #f1 m n v1;
  unfold gpu_pts_to_slice arr #f2 m n v2;
  mask_gather arr;
  mask_share_gen arr f1 f2;
  fold gpu_pts_to_slice arr #f1 m n v2;
  fold gpu_pts_to_slice arr #f2 m n v2;
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
