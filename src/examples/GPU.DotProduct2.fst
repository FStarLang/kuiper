module GPU.DotProduct2

#lang-pulse

open GPU
open GPU.Barrier.RPM
open GPU.Math

module A = Pulse.Lib.Array
module SZ = FStar.SizeT
module U32 = FStar.UInt32
module U64 = FStar.UInt64

module HR = GPU.HReduce_U64_Plus

#set-options "--z3rlimit 20"

let size : sz = 1024sz

let mul (s1 s2: seq u64)
  : Ghost (seq u64)
          (requires Seq.length s1 == Seq.length s2)
          (ensures fun _ -> True)
  = Seq.init_ghost (Seq.length s1)
      (fun i -> U64.mul_mod (Seq.index s1 i) (Seq.index s2 i))

let kpre (nth: nat) (ga1 ga2 r : gpu_array u64 nth) (s1 s2: erased (seq u64))
  (#_: squash ( Seq.length s1 == nth /\ Seq.length s2 == nth )) (tid:nat{tid < nth})
  : slprop =
    (gpu_pts_to_array #u64 #nth ga1 #(1.0R /. Real.of_int nth) s1 **
    gpu_pts_to_array #u64 #nth ga2 #(1.0R /. Real.of_int nth) s2) **
    gpu_pts_to_array1 r tid

let kpost (nth: nat) (ga1 ga2 r : gpu_array u64 nth) (s1 s2: erased (seq u64))
  (#_: squash ( Seq.length s1 == nth /\ Seq.length s2 == nth )) (tid:nat{tid < nth})
  : slprop =
    ((gpu_pts_to_array #u64 #nth ga1 #(1.0R /. Real.of_int nth) s1 **
    gpu_pts_to_array #u64 #nth ga2 #(1.0R /. Real.of_int nth) s2) **
    if_ (tid = 0) (HR.gpu_pts_to_slice_sum r 0 nth (mul s1 s2)))

// #set-options "--ext pulse:env_on_err=1"

[@@ CPrologue "__global__"]
fn kernel
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
  (ga1 ga2 : gpu_array u64 nth)
  (r : gpu_array u64 nth)
  (#s1 #s2: erased (seq u64))
  (#_: squash ( Seq.length s1 == nth /\ Seq.length s2 == nth ))
  (etid : erased tid_t { (gdim_x etid <: nat) == 1ul /\ (bdim_x etid <: nat) == SZ.sizet_to_uint32 nth })
  requires
    gpu **
    thread_id etid **
    mbarrier_tok nth (HR.barrier_matrix nth r (mul s1 s2)) 0 (tidx_x etid) **
    kpre nth ga1 ga2 r s1 s2 (thread_index etid)
  ensures
    gpu **
    thread_id etid **
    (exists* it. mbarrier_tok nth (HR.barrier_matrix nth r (mul s1 s2)) it (tidx_x etid)) **
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
  (**)let dot_v = hide (mul s1 s2);
  
  (**)unfold (gpu_pts_to_array1 r tid);
  gpu_array_write #u64 #(SZ.v nth) #(SZ.v tid) #(hide (SZ.v tid+1)) r tid vm;
  
  (* sigh... this is terrible. It's a one element sequence. *)
  with s'. assert (gpu_pts_to_array_slice r tid (tid+1) s');
  assert (pure (vm == Seq.index dot_v tid));
  assert (pure (Seq.index s' 0 == vm));
  assert (pure (Seq.length s' == 1));
  GPU.Seq.Common.lem_one_elem s' vm; (* oof *)
  assert (pure (s' == seq![vm <: u64])); (* the freaking refinement made this very difficult. *)
  rewrite each s' as seq![vm <: u64];
  
  (* Reduction *)
  GPU.HReduce_U64_Plus.kernel nth r #dot_v #() etid;
  
  fold (kpost nth ga1 ga2 r s1 s2 tid);
}

let shared_array (#nth : nat { nth <> 0 }) (ga : gpu_array u64 nth) (#v: seq u64 { Seq.length v == nth }) (_: nat): slprop =
  gpu_pts_to_array ga #(1.0R /. Real.of_int nth) v

ghost
fn share_array
  (#nth : nat { nth <> 0 })
  (ga : gpu_array u64 nth)
  (#v: erased (seq u64) { reveal (Seq.length v) == nth })
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
  (#v: erased (seq u64) { reveal (Seq.length v) == nth })
  requires bigstar 0 nth (shared_array #nth ga #v)
  ensures  gpu_pts_to_array ga #1.0R v
{
  admit();
}

fn main
  (a1 a2: array u64)
  (v1 v2: erased (seq u64))
  (#_: squash (Seq.length v1 = size /\ Seq.length v2 = size))
  requires cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2
  returns  dp: u64
  ensures  cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2 ** pure (dp == HR.sum (mul v1 v2))
{
  let ar = A.alloc #u64 0UL size;

  let ga1 = gpu_array_alloc #u64 size;
  let ga2 = gpu_array_alloc #u64 size;

  GPU.Array.gpu_memcpy_host_to_device a1 ga1 size;
  GPU.Array.gpu_memcpy_host_to_device a2 ga2 size;
  
  let gr = gpu_array_alloc #u64 size;

  // Slicing the arrays
  (**)share_array ga1;
  (**)share_array ga2;
  (**)gpu_array_slice_1_underspec gr;

  // Boring combination of resources
  (**)bigstar_zip 0 size (shared_array ga1) (shared_array ga2);
  (**)bigstar_zip 0 size _ (gpu_pts_to_array1 gr);
  (**)rewrite
    (bigstar 0 size
      (fun i -> ((shared_array #size ga1 #v1 i **
                 shared_array #size ga2 #v2 i) **
                 gpu_pts_to_array1 gr i)))
  as
    (bigstar 0 size (fun i -> kpre size ga1 ga2 gr v1 v2 i))
    by tadmit ();
  (**)bigstar_uneta ();

  rewrite
    bigstar 0 size
      (kpre size ga1 ga2 gr v1 v2)
  as
    bigstar 0 (1 * SZ.v size)
      (kpre size ga1 ga2 gr v1 v2);

  launch_kernel_n_m_barrier #0 1sz size
    #(kpre size ga1 ga2 gr v1 v2)
    #(kpost size ga1 ga2 gr v1 v2)
    #(HR.barrier_matrix size gr (mul v1 v2))
    (fun etid -> kernel size ga1 ga2 gr #v1 #v2 etid);

  (**)bigstar_eta ();
  // TODO:
  (**)drop_
        (bigstar 0 (1 * SZ.v size) (fun i -> kpost size ga1 ga2 gr v1 v2 i));
  (**)assume_
        (bigstar 0 size
          (fun i -> ((gpu_pts_to_array #u64 #size ga1 #(1.0R /. Real.of_int size) v1 **
                    gpu_pts_to_array #u64 #size ga2 #(1.0R /. Real.of_int size) v2) **
                    if_ (i = 0) (HR.gpu_pts_to_slice_sum gr 0 size (mul v1 v2)))
        ));
  
  (**)bigstar_unzip 0 size _ _;
  (**)bigstar_unzip 0 size _ _;
  
  (**)bigstar_uneta () #0 #0 #size #(shared_array #size ga1 #v1);
  gather_array ga1;
  (**)bigstar_uneta () #0 #0 #size #(shared_array #size ga2 #v2);
  gather_array ga2;

  bigstar_if_elim #_ #0 #size 0 (fun _ -> HR.gpu_pts_to_slice_sum #size gr 0 size (mul v1 v2));

  unfold HR.gpu_pts_to_slice_sum;
  if_elim_true _;
  unfold HR.gpu_pts_to_slice_sum_inner;
  with res. assert (gpu_pts_to_array_slice gr 0 size res);
  fold (gpu_pts_to_array #u64 #size gr #1.0R res);

  // TODO: don't copy whole array
  GPU.Array.gpu_memcpy_device_to_host ar gr size;

  gpu_array_free ga1;
  gpu_array_free ga2;
  gpu_array_free gr;

  let dp = ar.(0sz);
  A.free ar;
  dp
}
