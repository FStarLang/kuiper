module Kuiper.DotProduct3

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math

module A = Pulse.Lib.Array
module SZ = FStar.SizeT
module U32 = FStar.UInt32
module U64 = FStar.UInt64

module HR = Kuiper.HReduceU64Plus

#set-options "--z3rlimit 20"

let kpre (nth: nat) (ga1 ga2 r : gpu_array u64 nth) (s1 s2: erased (seq u64))
  (#_: squash ( len s1 == nth /\ len s2 == nth )) (tid:nat{tid < nth})
  : slprop =
    (gpu_pts_to_array #u64 #nth ga1 #(1.0R /. Real.of_int nth) s1 **
    gpu_pts_to_array #u64 #nth ga2 #(1.0R /. Real.of_int nth) s2) **
    if_ (tid = 0) (exists* sr. gpu_pts_to_array #u64 #nth r sr)

let kpost (nth: nat) (ga1 ga2 r : gpu_array u64 nth) (s1 s2: erased (seq u64))
  (#_: squash ( len s1 == nth /\ len s2 == nth )) (tid:nat{tid < nth})
  : slprop =
    ((gpu_pts_to_array #u64 #nth ga1 #(1.0R /. Real.of_int nth) s1 **
    gpu_pts_to_array #u64 #nth ga2 #(1.0R /. Real.of_int nth) s2) **
    if_ (tid = 0) (HR.gpu_pts_to_slice_sum r 0 nth (pmul s1 s2)))

[@@pulse_unfold]
let shared_pre (nth: nat) (sr gr : gpu_array u64 nth) (s1 s2: erased (seq u64))
  (#_: squash ( len s1 == nth /\ len s2 == nth )) (it: nat) (tid:nat{tid < nth})
  : slprop =
    gpu_pts_to_array1 sr tid **
    mbarrier_tok nth (HR.barrier_matrix nth sr (pmul s1 s2)) it tid

[@@pulse_unfold]
let shared_post (nth: nat) (sr gr : gpu_array u64 nth) (s1 s2: erased (seq u64))
  (#_: squash ( len s1 == nth /\ len s2 == nth )) (tid:nat{tid < nth})
  : slprop =
    if_ (tid = 0) (HR.gpu_pts_to_slice_sum sr 0 nth (pmul s1 s2)) **
    (exists* it. mbarrier_tok nth (HR.barrier_matrix nth sr (pmul s1 s2)) it tid)

// #set-options "--ext pulse:env_on_err=1"

[@@ CPrologue "__device__"; "KrmlPrivate"]
noextract inline_for_extraction
fn fixup
  (nth: erased nat { 0 < nth /\ nth <= 1024 })
  (ar: gpu_array u64 nth)
  (r: gpu_array u64 nth)
  (s1 s2: erased (seq u64))
  (#_: squash (len s1 = nth /\ len s2 = nth))
  (tid: SZ.t { SZ.v tid < nth })
  requires gpu **
    if_ (SZ.v tid = 0) (exists* sr. gpu_pts_to_array r sr) **
    HR.kpost nth ar (pmul s1 s2) tid
  ensures  gpu **
    HR.kpost nth r (pmul s1 s2) tid **
    HR.kpost nth ar (pmul s1 s2) tid
{
  let dot_v = hide (pmul s1 s2);
  if (tid = 0sz) {
    if_elim_true (exists* sr. gpu_pts_to_array r sr);

    // Duplicate
    if_elim_true (HR.gpu_pts_to_slice_sum ar 0 nth dot_v);

    unfold HR.gpu_pts_to_slice_sum;
    if_elim_true (exists* v. HR.gpu_pts_to_slice_sum_inner ar 0 nth dot_v v);
    unfold HR.gpu_pts_to_slice_sum_inner;

    let vv = gpu_array_read #u64 #nth #0 #nth ar 0sz;
    with cv. assert (gpu_pts_to_array r cv);
    gpu_pts_to_ref r;
    unfold gpu_pts_to_array r cv;
    gpu_array_write #u64 #nth #0 #nth r 0sz vv;
    
    with v1. assert (gpu_pts_to_array_slice ar 0 nth v1);
    // assert (pure (Seq.index v1 0 == HR.sum dot_v));
    fold HR.gpu_pts_to_slice_sum_inner #nth ar 0 nth dot_v v1;
    if_intro_true (exists* v. HR.gpu_pts_to_slice_sum_inner #nth ar 0 nth dot_v v);
    fold HR.gpu_pts_to_slice_sum ar 0 nth dot_v;

    with v2. assert (gpu_pts_to_array_slice r 0 nth v2);
    fold HR.gpu_pts_to_slice_sum_inner #nth r 0 nth dot_v v2;
    if_intro_true (exists* v. HR.gpu_pts_to_slice_sum_inner #nth r 0 nth dot_v v);
    fold HR.gpu_pts_to_slice_sum r 0 nth dot_v;

    if_intro_true (HR.gpu_pts_to_slice_sum r 0 nth (pmul s1 s2));
    if_intro_true (HR.gpu_pts_to_slice_sum ar 0 nth (pmul s1 s2));
  } else {
    rewrite each (SZ.v tid = 0) as false;
    if_elim_false (exists* sr. gpu_pts_to_array r sr);
    if_intro_false (HR.gpu_pts_to_slice_sum r 0 nth dot_v);
    rewrite (if_ false (HR.gpu_pts_to_slice_sum r 0 nth dot_v))
         as (if_ (SZ.v tid = 0) (HR.gpu_pts_to_slice_sum r 0 nth (pmul s1 s2)));
  }
}

[@@ CPrologue "__global__"]
fn kernel
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
  (ga1 ga2 : gpu_array u64 nth)
  (r : gpu_array u64 nth)
  (#s1 #s2: erased (seq u64))
  (#_: squash ( len s1 == nth /\ len s2 == nth ))
  (ear: erased (gpu_array u64 nth))
  (etid : erased tid_t { (gdim_x etid <: nat) == 1ul /\ (bdim_x etid <: nat) == SZ.sizet_to_uint32 nth })
  requires
    gpu **
    thread_id etid **
    shmem_tok ear **
    shared_pre nth ear r s1 s2 0 (thread_index etid) **
    kpre nth ga1 ga2 r s1 s2 (thread_index etid)
  ensures
    gpu **
    thread_id etid **
    shared_post nth ear r s1 s2 (thread_index etid) **
    kpost nth ga1 ga2 r s1 s2 (thread_index etid)
{
  let tid = thread_idx_x ();
  let tid : sz = SZ.uint32_to_sizet tid;
  (**)unfold (kpre nth ga1 ga2 r s1 s2 tid);

  (**)unfold (gpu_pts_to_array #u64 #(SZ.v nth) ga1 #(1.0R /. Real.of_int nth) s1);
  let v1 = gpu_array_read #u64 #(SZ.v nth) #0 #(SZ.v nth) ga1 tid #s1;
  (**)fold (gpu_pts_to_array #u64 #nth ga1 #(1.0R /. Real.of_int nth) s1);

  (**)unfold (gpu_pts_to_array #u64 #nth ga2 #(1.0R /. Real.of_int nth) s2);
  let v2 = gpu_array_read #u64 #(SZ.v nth) #0 #(SZ.v nth) ga2 tid #s2;
  (**)fold (gpu_pts_to_array #u64 #nth ga2 #(1.0R /. Real.of_int nth) s2);
  
  let vm = U64.mul_mod v1 v2;
  (**)let dot_v = hide (pmul s1 s2);
  
  let ar = obtain_shmem ear;
  unfold gpu_pts_to_array1 ar tid;
  gpu_array_write #u64 #(SZ.v nth) #(SZ.v tid) #(hide (SZ.v tid+1)) ar tid vm;
  
  (* sigh... this is terrible. It's a one element sequence. *)
  with s'. assert (gpu_pts_to_array_slice ar tid (tid+1) s');
  assert (pure (vm == Seq.index dot_v tid));
  assert (pure (Seq.index s' 0 == vm));
  assert (pure (len s' == 1));
  Kuiper.Seq.Common.lem_one_elem s' vm; (* oof *)
  assert (pure (s' == seq![vm <: u64])); (* the freaking refinement made this very difficult. *)
  rewrite each s' as seq![vm <: u64];
  
  (* Reduction *)
  HR.reduce nth ar #dot_v #() etid;
  
  fixup nth ar r s1 s2 tid;
  fold (kpost nth ga1 ga2 r s1 s2 tid);
  fold (shared_post nth ear r s1 s2 tid);
  ()
}

let shared_array (#nth : nat { nth <> 0 }) (ga : gpu_array u64 nth) (#v: seq u64 { len v == nth }) (_: nat): slprop =
  gpu_pts_to_array ga #(1.0R /. Real.of_int nth) v

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
  (#nth : nat { nth <> 0 })
  (ga : gpu_array u64 nth)
  (#v: erased (seq u64) { reveal (len v) == nth })
  requires bigstar 0 nth (shared_array #nth ga #v)
  ensures  gpu_pts_to_array ga #1.0R v
{
  admit();
}

ghost fn setup
  (nthr: SZ.t { 0 < SZ.v nthr /\ SZ.v nthr <= 1024 })
  (ear: gpu_array u64 nthr)
  (bid: SZ.t)
  (gr: gpu_array u64 nthr)
  (s1 s2: erased (seq u64))
  (#_: squash ( len s1 == nthr /\ len s2 == nthr ))
  requires block_setup nthr ** (exists* v. gpu_pts_to_array #u64 #nthr ear #1.0R v)
  ensures  block_setup nthr ** bigstar 0 nthr (fun tid -> shared_pre nthr ear gr s1 s2 0 tid)
{
  mk_mbarrier nthr (HR.barrier_matrix nthr ear (pmul s1 s2));
  gpu_array_slice_1_underspec ear;
  bigstar_zip 0 nthr (gpu_pts_to_array1 ear) (mbarrier_tok nthr (HR.barrier_matrix nthr ear (pmul s1 s2)) 0);
  ()
}

let u64_comm_semigroup ()
: squash (is_comm_semigroup HR.neu HR.op)
= ()

fn main
  (a1 a2: array u64)
  (v1 v2: erased (seq u64))
  (#_: squash (len v1 = dp2_size /\ len v2 = dp2_size))
  requires cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2
  returns  dp: u64
  ensures  cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2 ** pure (dp == sum (pmul v1 v2))
{
  let ar = A.alloc #u64 0UL dp2_size;

  let ga1 = gpu_array_alloc #u64 dp2_size;
  let ga2 = gpu_array_alloc #u64 dp2_size;

  Kuiper.Array.gpu_memcpy_host_to_device ga1 a1 dp2_size;
  Kuiper.Array.gpu_memcpy_host_to_device ga2 a2 dp2_size;
  
  let gr = gpu_array_alloc #u64 dp2_size;

  // Slicing the arrays
  (**)share_array ga1;
  (**)share_array ga2;
  (**)gpu_array_slice_1_underspec gr;

  // Boring combination of resources
  (**)bigstar_zip 0 dp2_size (shared_array ga1) (shared_array ga2);
  (**)bigstar_zip 0 dp2_size _ (gpu_pts_to_array1 gr);
  (**)rewrite
    (bigstar 0 dp2_size
      (fun i -> ((shared_array #dp2_size ga1 #v1 i **
                 shared_array #dp2_size ga2 #v2 i) **
                 gpu_pts_to_array1 gr i)))
  as
    (bigstar 0 dp2_size (fun i -> kpre dp2_size ga1 ga2 gr v1 v2 i))
    by tadmit ();
  // (**)bigstar_uneta ();

  rewrite
    bigstar 0 dp2_size
      (kpre dp2_size ga1 ga2 gr v1 v2)
  as
    bigstar 0 (1 * SZ.v dp2_size)
      (kpre dp2_size ga1 ga2 gr v1 v2);

  launch_kernel_n_m_shmem #0 1sz dp2_size
    #(kpre dp2_size ga1 ga2 gr v1 v2)
    #(kpost dp2_size ga1 ga2 gr v1 v2)
    u64
    dp2_size
    #(fun ear bid tid -> shared_pre dp2_size ear gr v1 v2 0 tid)
    #(fun ear bid tid -> shared_post dp2_size ear gr v1 v2 tid)
    (fun ear bid -> setup dp2_size ear bid gr v1 v2)
    (fun ear etid -> kernel dp2_size ga1 ga2 gr #v1 #v2 ear etid);

  // (**)bigstar_eta ();
  // TODO:
  (**)drop_
        (bigstar 0 (1 * SZ.v dp2_size) (fun i -> kpost dp2_size ga1 ga2 gr v1 v2 i));
  (**)assume
        (bigstar 0 dp2_size
          (fun i -> ((gpu_pts_to_array #u64 #dp2_size ga1 #(1.0R /. Real.of_int dp2_size) v1 **
                    gpu_pts_to_array #u64 #dp2_size ga2 #(1.0R /. Real.of_int dp2_size) v2) **
                    if_ (i = 0) (HR.gpu_pts_to_slice_sum gr 0 dp2_size (pmul v1 v2)))
        ));
  
  (**)bigstar_unzip 0 dp2_size _ _;
  (**)bigstar_unzip 0 dp2_size _ _;
  
  (**)bigstar_uneta () #0 #0 #dp2_size #(shared_array #dp2_size ga1 #v1);
  gather_array ga1;
  (**)bigstar_uneta () #0 #0 #dp2_size #(shared_array #dp2_size ga2 #v2);
  gather_array ga2;

  bigstar_if_elim #_ #0 #dp2_size 0 (fun _ -> HR.gpu_pts_to_slice_sum #dp2_size gr 0 dp2_size (pmul v1 v2));

  unfold HR.gpu_pts_to_slice_sum;
  if_elim_true _;
  unfold HR.gpu_pts_to_slice_sum_inner;
  with res. assert (gpu_pts_to_array_slice gr 0 dp2_size res);
  fold (gpu_pts_to_array #u64 #dp2_size gr #1.0R res);

  // TODO: don't copy whole array
  Kuiper.Array.gpu_memcpy_device_to_host ar gr dp2_size;

  gpu_array_free ga1;
  gpu_array_free ga2;
  gpu_array_free gr;

  let dp = ar.(0sz);

  (* Finally, ensure that the reduction must be sum *)
  u64_comm_semigroup ();
  IsReduction.ac_eq_foldl HR.neu HR.op (pmul v1 v2) dp;

  A.free ar;
  dp
}
