module Kuiper.Poly.DotProduct

#lang-pulse

(* DotProduct over HReduce: first do a pointwise mul of a1,a2 into a1
   then reduce a1, then return the first element of a1. This trashes a1.
   Can we uniformly support writing the temporary products to a temporary
   array r, that could be instantiated to ga1 too?
   This means something like
     requires
       a1 |-> __ **
       a2 |-> __ **
       (swap == a1 \/ swap == a2 \/
        swap |-> __)
*)

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.IsReduction

module V = Pulse.Lib.Vec
module SZ = FStar.SizeT

module HR = Kuiper.Poly.HReduce
friend Kuiper.Poly.HReduce (* use gpu_pts_to_slice_sum, refactor ! *)

(* - Mutable permission over single cell in ga1
   - Read permission over same cell in ga2
   - Barrier for operating over ga1. *)
let kpre
  (#et:Type0) {| scalar et |}
  (lena : nat)
  (ga1 ga2 : gpu_array et lena)
  (s1 s2 : erased (seq et))
  (#_: squash ( len s1 == lena /\ len s2 == lena ))
  (tid : natlt lena)
  : slprop
  = gpu_pts_to_slice ga1                 tid (tid + 1) seq![s1 @! tid] **
    gpu_pts_to_slice ga2 #(1.0R /. lena) tid (tid + 1) seq![s2 @! tid] **
    mbarrier_tok lena (HR.barrier_matrix lena ga1 (pmul s1 s2)) 0 tid

(* ^ Note how the instantiation of the barrier states (pmul s1 s2) regarless
     of whatever is in sr. This is since the first thing the kernel will do is
     write to r without any synchronization happening before that. Or, in other words,
    we are going to call hreduce after writing into r with the components of (pmul s1 s2),
    so that is the "contract" that the barrier must use. *)

let kpost
  (#et:Type0) {| scalar et |}
  (lena : nat)
  (ga1 ga2 : gpu_array et lena)
  (s1 s2 : erased (seq et))
  (#_: squash ( len s1 == lena /\ len s2 == lena ))
  (tid : natlt lena)
  : slprop
  = gpu_pts_to_slice ga2 #(1.0R /. lena) tid (tid + 1) seq![s2 @! tid] **
    (exists* it.
      mbarrier_tok lena (HR.barrier_matrix lena ga1 (pmul s1 s2)) it tid) **
    (if_ (tid = 0) (gpu_pts_to_slice_sum ga1 0 lena (pmul s1 s2)))

// noextract inline_for_extraction
// fn fixup
//   (nth: erased nat { 0 < nth /\ nth <= 1024 })
//   (ar: gpu_array et nth)
//   (r: gpu_array et nth)
//   (s1 s2: erased (seq et))
//   (#_: squash (len s1 = nth /\ len s2 = nth))
//   (tid: SZ.t { SZ.v tid < nth })
//   requires gpu **
//     if_ (SZ.v tid = 0) (exists* sr. gpu_pts_to_array r sr) **
//     HR.kpost ar (pmul s1 s2) tid
//   ensures  gpu **
//     if_ (SZ.v tid = 0) (exists* sr. gpu_pts_to_array r sr) **
//     HR.kpost ar (pmul s1 s2) tid
// {
//   let dot_v = hide (pmul s1 s2);
//   if (tid = 0sz) {
//     rewrite each (SZ.v tid = 0) as true;
//     if_elim_true (exists* sr. gpu_pts_to_array r sr);

//     // Duplicate
//     if_elim_true (HR.gpu_pts_to_slice_sum ar 0 nth dot_v);

//     unfold (HR.gpu_pts_to_slice_sum ar 0 nth dot_v);
//     if_elim_true (exists* v. HR.gpu_pts_to_slice_sum_inner ar 0 nth dot_v v);

//     let vv = gpu_array_read #et #nth #0 #nth ar 0sz; (* CONCRETE STEP *)
//     with cv. assert (gpu_pts_to_array r cv);
//     gpu_pts_to_ref r;
//     gpu_array_write #et #nth #0 #nth r 0sz vv; (* CONCRETE STEP *)

//     with v1. assert (gpu_pts_to_slice ar 0 nth v1);
//     // assert (pure (Seq.index v1 0 == HR.sum dot_v));
//     fold HR.gpu_pts_to_slice_sum_inner #_ #_ #nth ar 0 nth dot_v v1;
//     if_intro_true (exists* v. HR.gpu_pts_to_slice_sum_inner #_ #_ #nth ar 0 nth dot_v v);
//     fold HR.gpu_pts_to_slice_sum ar 0 nth dot_v;

//     with v2. assert (gpu_pts_to_slice r 0 nth v2);
//     fold HR.gpu_pts_to_slice_sum_inner #_ #_ #nth r 0 nth dot_v v2;
//     if_intro_true (exists* v. HR.gpu_pts_to_slice_sum_inner #_ #_ #nth r 0 nth dot_v v);
//     fold HR.gpu_pts_to_slice_sum r 0 nth dot_v;

//     if_intro_true' (SZ.v tid = 0) (HR.gpu_pts_to_slice_sum r 0  (1 * nth) (pmul s1 s2));
//     if_intro_true' (SZ.v tid = 0) (HR.gpu_pts_to_slice_sum ar 0 (1 * nth) (pmul s1 s2));
//   } else {
//     rewrite each (SZ.v tid = 0) as false;
//     if_elim_false (exists* sr. gpu_pts_to_array r sr);
//     if_intro_false (HR.gpu_pts_to_slice_sum r 0 nth dot_v);
//     rewrite (if_ false (HR.gpu_pts_to_slice_sum r 0 nth dot_v))
//          as (if_ (SZ.v tid = 0) (HR.gpu_pts_to_slice_sum r 0 (1 * nth) (pmul s1 s2)));
//   }
// }

inline_for_extraction noextract
fn kf
  (#et:Type0) {| scalar et |}
  (lena : szp{lena <= max_threads})
  (ga1 ga2 : gpu_array et lena)
  (#s1 #s2 : erased (seq et))
  (#_: squash ( len s1 == lena /\ len s2 == lena ))
  (tid : szlt lena)
  ()
  requires
    gpu **
    kpre lena ga1 ga2 s1 s2 tid **
    thread_id lena tid
  ensures
    gpu **
    kpost lena ga1 ga2 s1 s2 tid **
    thread_id lena tid
{
  (**)unfold (kpre lena ga1 ga2 s1 s2 tid);

  let v1 = gpu_array_read #et #(SZ.v lena) #tid #(tid + 1) ga1 tid #_;
  let v2 = gpu_array_read #et #(SZ.v lena) #tid #(tid + 1) ga2 tid #_;

  let vm = mul v1 v2;

  gpu_array_write #et #(SZ.v lena) #(SZ.v tid) #(hide (SZ.v tid+1)) ga1 tid vm;

  (* Convince the SMT solver that these sequences are equal *)
  with s'.
    assert (gpu_pts_to_slice ga1 tid (tid+1) s');
  assert (pure (Seq.equal s' seq![pmul s1 s2 @! tid]));

  (* Reduction *)
  HR.d_reduce lena ga1 #(pmul s1 s2) #() tid ();

  fold (kpost lena ga1 ga2 s1 s2 tid);
  ()
}

ghost
fn setup
  (#et:Type0) {| scalar et |}
  (lena : szp{SZ.v lena <= max_threads})
  (ga1 ga2 : gpu_array et lena)
  (#s1 #s2 : erased (seq et))
  (_: squash ( len s1 == SZ.v lena /\ len s2 == SZ.v lena ))
  requires
    block_setup_tok lena **
    ((ga2 |-> s2) ** (ga1 |-> s1))
  ensures
    block_setup_tok lena **
    (forall+ (bid : natlt lena). kpre lena ga1 ga2 s1 s2 bid) **
    emp (* frame *)
{
  // mk_mbarrier nthr (HR.barrier_matrix nthr ar (pmul s1 s2));
  mk_mbarrier lena (HR.barrier_matrix lena ga1 (pmul s1 s2));
  gpu_array_slice_1 ga1;
  bigstar_zip 0 lena _ _;
  gpu_array_slice_1 ga2;
  bigstar_zip 0 lena _ _;
  // slice
  // bigstar_zip 0 nthr (gpu_pts_to_array1 ar) (mbarrier_tok nthr (HR.barrier_matrix nthr ar (pmul s1 s2)) 0);
  // rewrite each nthr as Enumerable.cardinal (natlt nthr) #_;
  // forevery_fromstar #(natlt nthr) (fun i ->
  //   gpu_pts_to_array1 ar i **
  //   mbarrier_tok nthr (HR.barrier_matrix nthr ar (pmul s1 s2)) 0 i);
  admit()
}

inline_for_extraction noextract
let dp_kernel
  (#et:Type0) {| scalar et |}
  (lena : szp{SZ.v lena <= max_threads})
  (ga1 ga2 : gpu_array et lena)
  (#s1 #s2 : erased (seq et))
  (#_: squash ( len s1 == SZ.v lena /\ len s2 == SZ.v lena ))
  : kernel_desc
      ((ga2 |-> s2) ** (ga1 |-> s1))
      ((ga2 |-> s2) ** (exists* s1'. gpu_pts_to_array ga1 #1.0R s1' **
                                     pure (len s1' == len s1 /\ squash (is_reduction zero add (pmul s1 s2) (Seq.head s1')))))
  = {
    nthr = lena;
    f = kf lena ga1 ga2 #s1 #s2;

    block_setup    = setup lena ga1 ga2 #s1 #s2;
    block_teardown = magic ();

    kpre  = kpre lena ga1 ga2 s1 s2;
    kpost = kpost lena ga1 ga2 s1 s2;

    frame = emp;
  } <: kernel_desc_1_n _ _

inline_for_extraction noextract
fn dotprod
  (#et:Type0) {| scalar et |}
  (lena : szp{lena <= max_threads})
  (a1 a2: vec et)
  (v1 v2: erased (seq et))
  (#_: squash (len v1 == lena /\ len v2 == lena))
  preserves
    cpu **
    (a1 |-> v1) **
    (a2 |-> v2)
  requires
    pure (is_comm_semigroup #et zero add)
  returns
    dp: et
  ensures
    pure (dp == sum (pmul v1 v2))
{
  Pulse.Lib.Vec.pts_to_len a1;
  Pulse.Lib.Vec.pts_to_len a2;

  (* swap space *)
  let ar = V.alloc #et zero 1sz;

  let ga1 = gpu_array_alloc #et lena;
  let ga2 = gpu_array_alloc #et lena;

  Kuiper.Array.gpu_memcpy_host_to_device ga1 a1 lena;
  Kuiper.Array.gpu_memcpy_host_to_device ga2 a2 lena;

  (* Why are the implicits needed? *)
  launch_sync (dp_kernel lena ga1 ga2 #v1 #v2);

  gpu_pts_to_ref ga1;
  gpu_pts_to_ref ga2;

  assert (pure (Seq.length v1 == lena));
  assert (pure (Seq.length v2 == lena));
  assert (pure (0sz + 1sz <= lena));
  (* inference sucks here, what's going on? *)
  Kuiper.Array.gpu_memcpy_device_to_host' #_ #_ #1 ar 0sz #_ ga1 0sz 1sz;

  gpu_array_free ga1;
  gpu_array_free ga2;

  let dp = ar.(0sz);

  (* Finally, ensure that the reduction must be sum *)
  Kuiper.IsReduction.ac_eq_foldl zero add (pmul v1 v2) dp;

  V.free ar;
  dp
}
