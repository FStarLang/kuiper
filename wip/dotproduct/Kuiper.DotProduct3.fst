module Kuiper.DotProduct3

#lang-pulse

(* DotProduct over HReduce: first do a pointwise mul of a1,a2 into r,
   then reduce r, then return the first element of r. We put r in
   shared memory. *)

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math

module V = Pulse.Lib.Vec
module SZ = FStar.SizeT
module U64 = FStar.UInt64

module HR = Kuiper.HReduce
friend Kuiper.HReduce (* use gpu_pts_to_slice, refactor ! *)

#set-options "--z3rlimit 20"

let kpre
  (lena : nat)
  (ga1 ga2 r : gpu_array u64 lena)
  (s1 s2 sr: erased (seq u64))
  (#_: squash ( len s1 == lena /\ len s2 == lena /\ len sr == lena ))
  (tid : natlt lena)
  : slprop
  = gpu_pts_to_array #u64 ga1 #(1.0R /. lena) s1 **
    gpu_pts_to_array #u64 ga2 #(1.0R /. lena) s2 **
    gpu_pts_to_slice #u64 r tid (tid+1) seq![sr @! tid] **
    mbarrier_tok lena (HR.barrier_matrix lena r sr) 0 tid **
    shmem_tok r

let kpost
  (lena : nat)
  (ga1 ga2 r : gpu_array u64 lena)
  (s1 s2 : erased (seq u64))
  (#_: squash ( len s1 == lena /\ len s2 == lena ))
  (tid : natlt lena)
  : slprop
  = gpu_pts_to_array #u64 #lena ga1 #(1.0R /. lena) s1 **
    gpu_pts_to_array #u64 #lena ga2 #(1.0R /. lena) s2 **
    shmem_tok r **
    (exists* it.
      mbarrier_tok lena (HR.barrier_matrix lena r (pmul s1 s2)) it tid) **
    (if_ (tid = 0) (HR.gpu_pts_to_slice_sum r 0 lena (pmul s1 s2)))

// noextract inline_for_extraction
// fn fixup
//   (nth: erased nat { 0 < nth /\ nth <= 1024 })
//   (ar: gpu_array u64 nth)
//   (r: gpu_array u64 nth)
//   (s1 s2: erased (seq u64))
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

//     let vv = gpu_array_read #u64 #nth #0 #nth ar 0sz; (* CONCRETE STEP *)
//     with cv. assert (gpu_pts_to_array r cv);
//     gpu_pts_to_ref r;
//     gpu_array_write #u64 #nth #0 #nth r 0sz vv; (* CONCRETE STEP *)

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

[@@CPrologue "__device__"; "KrmlPrivate"]
fn kernel
  (lena : sz { 0 < SZ.v lena /\ SZ.v lena <= 1024 })
  (ga1 ga2 : gpu_array u64 lena)
  (#s1 #s2 #sr : erased (seq u64))
  (#_: squash ( len s1 == lena /\ len s2 == lena /\ len sr == lena ))
  (ear: erased (gpu_array u64 lena))
  (etid : enatlt lena)
  requires
    gpu **
    kpre lena ga1 ga2 ear s1 s2 sr etid **
    thread_id lena etid
  ensures
    gpu **
    thread_id lena etid **
    kpost lena ga1 ga2 ear s1 s2 etid
{
  let tid = get_tid (); rewrite each etid as tid;
  (**)unfold (kpre lena ga1 ga2 ear s1 s2 sr tid);

  let v1 = gpu_array_read #u64 #(SZ.v lena) #0 #(SZ.v lena) ga1 tid #s1;
  let v2 = gpu_array_read #u64 #(SZ.v lena) #0 #(SZ.v lena) ga2 tid #s2;

  let vm = U64.mul_mod v1 v2;
  (**)let dot_v = hide (pmul s1 s2);

  let ar = obtain_shmem ear; rewrite each ear as ar;
  gpu_array_write #u64 #(SZ.v lena) #(SZ.v tid) #(hide (SZ.v tid+1)) ar tid vm;

  (* sigh... this is terrible. It's a one element sequence. *)
  with s'. assert (gpu_pts_to_slice ar tid (tid+1) s');
  assert (pure (vm == Seq.index dot_v tid));
  assert (pure (Seq.index s' 0 == vm));
  assert (pure (len s' == 1));
  Kuiper.Seq.Common.lem_one_elem s' vm; (* oof *)
  assert (pure (s' == seq![vm <: u64])); (* the freaking refinement made this very difficult. *)
  rewrite each s' as seq![vm <: u64];

  (* Reduction *)
  HR.d_reduce lena ar #dot_v #() etid ();

  fold (kpost lena ga1 ga2 ear s1 s2 tid);
  ()
}

unfold
let shared_array (#nth : nat { nth <> 0 }) (ga : gpu_array u64 nth) (#v: seq u64 { len v == nth }) (_: nat): slprop =
  gpu_pts_to_array ga #(1.0R /. nth) v

ghost
fn share_array
  (#nth : nat { nth <> 0 })
  (ga : gpu_array u64 nth)
  (#v: erased (seq u64) { reveal (len v) == nth })
  requires gpu_pts_to_array ga #1.0R v
  ensures  bigstar 0 nth (shared_array #nth ga #v)
{
  rewrite gpu_pts_to_array ga #1.0R v
    as gpu_pts_to_array ga #(1.0R /. of_int 1) v;
  admit();
}

ghost
fn gather_array
  (#uid : int)
  (#nth : nat { nth <> 0 })
  (ga : gpu_array u64 nth)
  (#v: erased (seq u64) { reveal (len v) == nth })
  requires bigstar #uid 0 nth (shared_array #nth ga #v)
  ensures  gpu_pts_to_array ga #1.0R v
{
  admit();
}

ghost
fn setup
  (nthr: nat { 0 < nthr /\ nthr <= 1024 })
  (ear: gpu_array u64 nthr)
  (bid: nat)
  (gr: gpu_array u64 nthr)
  (s1 s2: erased (seq u64))
  (#_: squash ( len s1 == nthr /\ len s2 == nthr ))
  requires block_setup_tok nthr ** (exists* v. gpu_pts_to_array #u64 #nthr ear #1.0R v)
  ensures  block_setup_tok nthr ** (forall+ (tid : natlt nthr). shared_pre nthr ear gr s1 s2 0 0 tid)
{
  mk_mbarrier nthr (HR.barrier_matrix nthr ear (pmul s1 s2));
  gpu_array_slice_1_underspec ear;
  bigstar_zip 0 nthr (gpu_pts_to_array1 ear) (mbarrier_tok nthr (HR.barrier_matrix nthr ear (pmul s1 s2)) 0);
  rewrite each nthr as Enumerable.cardinal (natlt nthr) #_;
  forevery_fromstar #(natlt nthr) (fun i ->
    gpu_pts_to_array1 ear i **
    mbarrier_tok nthr (HR.barrier_matrix nthr ear (pmul s1 s2)) 0 i);
  ()
}

let u64_comm_semigroup ()
: squash (is_comm_semigroup #u64 zero add)
= admit()

fn main
  (a1 a2: vec u64)
  (v1 v2: erased (seq u64))
  (#_: squash (len v1 = dp2_size /\ len v2 = dp2_size))
  preserves cpu ** (a1 |-> v1) ** (a2 |-> v2)
  requires emp
  returns  dp: u64
  ensures  pure (dp == sum (pmul v1 v2))
{
  (* RESTORE *)
  admit();
  // let ar = V.alloc #u64 0UL dp2_size;

  // let ga1 = gpu_array_alloc #u64 dp2_size;
  // let ga2 = gpu_array_alloc #u64 dp2_size;

  // Kuiper.Array.gpu_memcpy_host_to_device ga1 a1 dp2_size;
  // Kuiper.Array.gpu_memcpy_host_to_device ga2 a2 dp2_size;

  // let gr = gpu_array_alloc #u64 dp2_size;

  // // Slicing the arrays
  // (**)share_array ga1;
  // (**)share_array ga2;
  // (**)gpu_array_slice_1_underspec gr;

  // // Boring combination of resources
  // (**)bigstar_zip 0 dp2_size (shared_array ga1) (shared_array ga2);
  // (**)bigstar_zip 0 dp2_size _ (gpu_pts_to_array1 gr);

  // forevery_fromstar #(natlt dp2_size) (fun i ->
  //   (shared_array ga1 i **
  //    shared_array ga2 i **
  //    gpu_pts_to_array1 gr i));

  // forevery_factor dp2_size 1 dp2_size _;
  // rewrite
  //   forall+ (bid:natlt 1) (tid:natlt dp2_size).
  //     ((shared_array ga1 #v1 (bid * dp2_size + tid) **
  //       shared_array ga2 #v2 (bid * dp2_size + tid)) **
  //      gpu_pts_to_array1 gr (bid * dp2_size + tid))
  // as
  //   forall+ (bid:natlt 1) (tid:natlt dp2_size).
  //     (kpre dp2_size ga1 ga2 gr v1 v2 bid tid)
  // by tadmit (); // needs intro exists within body, TODO

  // launch_kernel_n_m_shmem 1sz dp2_size
  //   #(kpre dp2_size ga1 ga2 gr v1 v2 #())
  //   #(kpost dp2_size ga1 ga2 gr v1 v2 #())
  //   u64
  //   dp2_size
  //   #(fun ear bid tid -> shared_pre dp2_size ear gr v1 v2 0 bid tid)
  //   #(fun ear bid tid -> shared_post dp2_size ear gr v1 v2 0 bid tid)
  //   (fun ear bid -> setup dp2_size ear bid gr v1 v2)
  //   (fun ear _ebid etid -> kernel dp2_size ga1 ga2 gr #v1 #v2 ear etid);

  // rewrite
  //   forall+ (bid:natlt 1) (tid:natlt dp2_size).
  //     (kpost dp2_size ga1 ga2 gr v1 v2 bid tid)
  // as
  //   forall+ (bid:natlt 1) (tid:natlt dp2_size).
  //     ((shared_array ga1 #v1 (bid * dp2_size + tid) **
  //       shared_array ga2 #v2 (bid * dp2_size + tid)) **
  //      if_ (bid = 0 && tid = 0) (HR.gpu_pts_to_slice_sum gr 0 dp2_size (pmul v1 v2)))
  // by tadmit (); // needs intro exists within body, TODO

  // forevery_unfactor' dp2_size 1 dp2_size _;
  // forevery_tostar #(natlt dp2_size) _;

  // (**)bigstar_unzip #1 #2 #0 0 dp2_size _ _;
  // (**)bigstar_unzip #3 #4 #1 0 dp2_size _ _;

  // (**)bigstar_uneta () #3 #0 #dp2_size #(shared_array #dp2_size ga1 #v1);
  // gather_array ga1;
  // (**)bigstar_uneta () #4 #0 #dp2_size #(shared_array #dp2_size ga2 #v2);
  // gather_array ga2;

  // bigstar_if_elim #_ #0 #dp2_size 0 (fun _ -> HR.gpu_pts_to_slice_sum #_ #_ #dp2_size gr 0 dp2_size (pmul v1 v2));

  // unfold HR.gpu_pts_to_slice_sum #_ #_ #dp2_size gr 0 dp2_size (pmul v1 v2);
  // if_elim_true _;
  // with res. assert (gpu_pts_to_slice gr 0 dp2_size res);

  // // TODO: don't copy whole array
  // Kuiper.Array.gpu_memcpy_device_to_host ar gr dp2_size;

  // gpu_array_free ga1;
  // gpu_array_free ga2;
  // gpu_array_free gr;

  // let dp = ar.(0sz);

  // (* Finally, ensure that the reduction must be sum *)
  // u64_comm_semigroup ();
  // IsReduction.ac_eq_foldl zero add (pmul v1 v2) dp;

  // V.free ar;
  // dp
}
