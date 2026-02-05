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

friend Kuiper.Poly.HReduce (* use gpu_pts_to_slice_sum, refactor ! *)

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Approximates

module V = Pulse.Lib.Vec
module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module HR = Kuiper.Poly.HReduce


let rsmul (s1 s2 : seq real{ Seq.length s1 == Seq.length s2 }) : GTot (seq real)
  = Seq.init_ghost (Seq.length s1) (fun i -> (s1 @! i) *. (s2 @! i))

let pmul_approximates_rsmul (#et:Type) {| scalar et, real_like et |}
  (s1 s2 : seq et)
  (vr1 vr2 : seq real{ s1 %~ vr1 /\ s2 %~ vr2 })
  : Lemma (requires Seq.length s1 == Seq.length s2)
          (ensures pmul s1 s2 %~ rsmul vr1 vr2)
          [SMTPat (pmul s1 s2 %~ rsmul vr1 vr2)]
= let aux (i : natlt (Seq.length s1))
    : Lemma (((s1 @! i) `mul` (s2 @! i)) %~ ((vr1 @! i) *. (vr2 @! i)))
  = a_mul (s1 @! i) (s2 @! i) (vr1 @! i) (vr2 @! i)
  in
  Classical.forall_intro aux

(* Pre and post conditions for the kernel *)

(* - Mutable permission over single cell in ga1
   - Read permission over same cell in ga2
   - Barrier for operating over ga1. *)
let kpre
  (#et:Type0) {| scalar et, real_like et |}
  (lena : nat)
  (ga1 ga2 : gpu_array et lena)
  (s1 s2 : erased (seq et))
  (vr1 vr2 : erased (seq real) { s1 %~ vr1 /\ s2 %~ vr2 })
  (#_: squash ( len s1 == lena /\ len s2 == lena ))
  (tid : natlt lena)
  : slprop
  = gpu_pts_to_slice ga1 tid (tid + 1) seq![s1 @! tid] **
    gpu_pts_to_slice ga2 tid (tid + 1) seq![s2 @! tid]

(* ^ Note how the instantiation of the barrier states (pmul s1 s2) regarless
     of whatever is in sr. This is since the first thing the kernel will do is
     write to r without any synchronization happening before that. Or, in other words,
    we are going to call hreduce after writing into r with the components of (pmul s1 s2),
    so that is the "contract" that the barrier must use. *)

let kpost
  (#et:Type0) {| scalar et, real_like et |}
  (lena : nat)
  (ga1 ga2 : gpu_array et lena)
  (s1 s2 : erased (seq et))
  (vr1 vr2 : erased (seq real) { s1 %~ vr1 /\ s2 %~ vr2 })
  (#_: squash ( len s1 == lena /\ len s2 == lena ))
  (tid : natlt lena)
  : slprop
  = gpu_pts_to_slice ga2 tid (tid + 1) seq![s2 @! tid] **
    (if_ (tid = 0) (HR.gpu_pts_to_slice_sum ga1 0 lena (pmul s1 s2) (rsmul vr1 vr2)))

inline_for_extraction noextract
fn kf
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp{lena <= max_threads})
  (ga1 ga2 : gpu_array et lena)
  (#s1 #s2 : erased (seq et))
  (#vr1 #vr2 : erased (seq real) { s1 %~ vr1 /\ s2 %~ vr2 })
  (#_: squash ( len s1 == lena /\ len s2 == lena ))
  (tid : szlt lena)
  ()
  requires
    gpu **
    kpre lena ga1 ga2 s1 s2 vr1 vr2 tid **
    thread_id lena tid **
    B.barrier_tok (mbarrier_contract #lena (HR.barrier_matrix lena ga1 (pmul s1 s2) (rsmul vr1 vr2))) **
    B.barrier_state 0
  ensures
    gpu **
    kpost lena ga1 ga2 s1 s2 vr1 vr2 tid **
    thread_id lena tid
{
  (**)unfold (kpre lena ga1 ga2 s1 s2 vr1 vr2 tid);

  let v1 = gpu_array_read ga1 tid;
  let v2 = gpu_array_read ga2 tid;

  let vm = mul v1 v2;

  gpu_array_write ga1 tid vm;

  (* Convince the SMT solver that these sequences are equal *)
  with s'.
    assert (gpu_pts_to_slice ga1 tid (tid+1) s');
  assert (pure (Seq.equal s' seq![pmul s1 s2 @! tid]));
  assert (gpu_pts_to_slice ga1 tid (tid+1) (seq![pmul s1 s2 @! tid]));

  (* Reduction *)
  HR.kf lena ga1 #(pmul s1 s2) #_ #() tid ();

  fold (kpost lena ga1 ga2 s1 s2 vr1 vr2 tid);
  ()
}

ghost
fn setup
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp{SZ.v lena <= max_threads})
  (ga1 ga2 : gpu_array et lena)
  (#s1 #s2 : seq et)
  (#vr1 #vr2 : seq real { s1 %~ vr1 /\ s2 %~ vr2 })
  (_: squash ( len s1 == SZ.v lena /\ len s2 == SZ.v lena ))
  norewrite
  requires
    ga2 |-> s2 ** ga1 |-> s1
  ensures
    (forall+ (bid : natlt lena). kpre lena ga1 ga2 s1 s2 vr1 vr2 bid) **
    emp (* frame *)
{
  gpu_array_slice_1 ga2;
  gpu_array_slice_1 ga1;
  forevery_zip (fun (i: natlt (v lena)) -> gpu_pts_to_slice ga1 i (i + 1) seq![Seq.Base.index s1 i]) _;
  forevery_map
   (fun (i: natlt (v lena)) -> //too bad that you have to write this; inference should find it
      gpu_pts_to_slice ga1 i (i + 1) seq![Seq.Base.index s1 i] **
      gpu_pts_to_slice ga2 i (i + 1) seq![Seq.Base.index s2 i])
   (kpre lena ga1 ga2 s1 s2 vr1 vr2) fn bid { fold kpre lena ga1 ga2 s1 s2 vr1 vr2 bid };
}

ghost
fn teardown
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp{SZ.v lena <= max_threads})
  (ga1 ga2 : gpu_array et lena)
  (#s1 #s2 : seq et)
  (#vr1 #vr2 : seq real { s1 %~ vr1 /\ s2 %~ vr2 })
  (_: squash ( len s1 == SZ.v lena /\ len s2 == SZ.v lena ))
  norewrite
  requires
    (forall+ (tid : natlt lena). kpost lena ga1 ga2 s1 s2 vr1 vr2 tid) **
    emp
  ensures
    ga2 |-> s2 **
    (exists* (s1' : seq et{Seq.length s1' > 0}).
      (ga1 |-> s1') **
      pure ((s1' @! 0) %~ real_seq_sum (rsmul vr1 vr2)))
{
  // rewrite_by (forall+ (tid : natlt lena). kpost lena ga1 ga2 s1 s2 vr1 vr2 tid) _
  //            (slprop_equiv_unfold (`%kpost)) ();
  //  rewrite (forall+ (tid : natlt lena). kpost lena ga1 ga2 s1 s2 vr1 vr2 tid) as _ by (norm [delta_only [`kpost]]);
  forevery_map (kpost lena ga1 ga2 s1 s2 vr1 vr2)
    (fun tid ->
      gpu_pts_to_slice ga2 tid (tid + 1) seq![s2 @! tid] **
      (if_ (tid = 0) (HR.gpu_pts_to_slice_sum ga1 0 lena (pmul s1 s2) (rsmul vr1 vr2))))
    fn tid { unfold (kpost lena ga1 ga2 s1 s2 vr1 vr2 tid) };
  forevery_unzip _ _;
  gpu_array_unslice_1 ga2;
  forevery_extract #(natlt lena) 0
    (fun tid -> (if_ (tid = 0) (HR.gpu_pts_to_slice_sum ga1 0 lena (pmul s1 s2) (rsmul vr1 vr2))));
  if_elim_true _;
  drop_ (Pulse.Lib.Trade.trade _ _);
  unfold HR.gpu_pts_to_slice_sum ga1 0 lena (pmul s1 s2) (rsmul vr1 vr2);
}

inline_for_extraction noextract
let dp_kernel
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp{SZ.v lena <= max_threads})
  (ga1 ga2 : gpu_array et lena {is_global_array ga1 /\ is_global_array ga2})
  (#s1 #s2 : erased (seq et))
  (#vr1 #vr2 : erased (seq real) { s1 %~ vr1 /\ s2 %~ vr2 })
  (#_: squash ( len s1 == SZ.v lena /\ len s2 == SZ.v lena ))
  : kernel_desc
      (ga2 |-> s2 ** ga1 |-> s1)
      (ga2 |-> s2 ** (exists* (s1' : seq et{Seq.length s1' > 0}). (ga1 |-> s1') **
                                     pure ((s1' @! 0) %~ real_seq_sum (rsmul vr1 vr2))))
  = {
    nthr = lena;
    f = kf lena ga1 ga2 #s1 #s2;

    barrier_contract = mbarrier_contract #lena (HR.barrier_matrix lena ga1 (pmul s1 s2) (rsmul vr1 vr2));
    barrier_ok       = mbarrier_transform _;

    block_setup    = setup lena ga1 ga2 #s1 #s2;
    block_teardown = teardown lena ga1 ga2 #s1 #s2;

    kpre  = kpre lena ga1 ga2 s1 s2 vr1 vr2;
    kpost = kpost lena ga1 ga2 s1 s2 vr1 vr2;

    frame = emp;

    kpost_sendable=solve;
    kpre_sendable=solve;
    full_post_sendable=solve;
    full_pre_sendable=solve;
  } <: kernel_desc_1_n_barr _ _


inline_for_extraction noextract
fn dotprod
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp{lena <= max_threads})
  (a1 a2: vec et)
  (v1 v2: erased (seq et))
  (vr1 vr2: erased (seq real) { v1 %~ vr1 /\ v2 %~ vr2 })
  (#_: squash (len v1 == lena /\ len v2 == lena))
  norewrite
  preserves
    cpu **
    a1 |-> v1 **
    a2 |-> v2
  returns
    dp: et
  ensures
    pure (dp %~ sum (pmul vr1 vr2))
{
  Pulse.Lib.Vec.pts_to_len a1;
  Pulse.Lib.Vec.pts_to_len a2;

  let ga1 = gpu_array_alloc #et lena;
  let ga2 = gpu_array_alloc #et lena;

  Kuiper.Array.gpu_memcpy_host_to_device ga1 a1 lena;
  Kuiper.Array.gpu_memcpy_host_to_device ga2 a2 lena;

  (* Why are the implicits needed? *)
  launch_sync (dp_kernel lena ga1 ga2 #v1 #v2 #vr1 #vr2);

  gpu_pts_to_ref_located ga1;
  gpu_pts_to_ref_located ga2;

  assert (pure (Seq.length v1 == lena));
  assert (pure (Seq.length v2 == lena));
  assert (pure (0sz + 1sz <= lena));

  (* swap space *)
  let ar = V.alloc #et zero 1sz;
  (* inference sucks here, what's going on? *)
  Kuiper.Array.gpu_memcpy_device_to_host' #_ #_ #1 //the dst_sz cannot be computed by unification;
      ar 0sz //#_
      ga1 0sz 1sz;

  gpu_array_free ga1;
  gpu_array_free ga2;

  let dp = ar.(0sz);
  V.free ar;

  (* Finally, ensure that the reduction must be sum *)
  assert pure (rsmul vr1 vr2 `Seq.equal` pmul vr1 vr2);

  dp
}
