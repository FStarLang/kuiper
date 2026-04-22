module Kuiper.Kernel.DotProduct

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

friend Kuiper.Kernel.HReduce (* use gpu_pts_to_slice_sum, refactor ! *)

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Tensor { ctlayout }

module V = Pulse.Lib.Vec
module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module HR = Kuiper.Kernel.HReduce
module Array1 = Kuiper.Array1

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
  (#l1 : Array1.layout lena) (ga1 : Array1.t et l1)
  (#l2 : Array1.layout lena) (ga2 : Array1.t et l2)
  (s1 s2 : erased (lseq et lena))
  (vr1 vr2 : erased (lseq real lena) { s1 %~ vr1 /\ s2 %~ vr2 })
  (#_: squash ( len s1 == lena /\ len s2 == lena ))
  (tid : natlt lena)
  : slprop
  = HR.array1_pts_to_slice ga1 tid (tid + 1) seq![s1 @! tid] **
    HR.array1_pts_to_slice ga2 tid (tid + 1) seq![s2 @! tid]

(* ^ Note how the instantiation of the barrier states (pmul s1 s2) regarless
     of whatever is in sr. This is since the first thing the kernel will do is
     write to r without any synchronization happening before that. Or, in other words,
    we are going to call hreduce after writing into r with the components of (pmul s1 s2),
    so that is the "contract" that the barrier must use. *)

let kpost
  (#et:Type0) {| scalar et, real_like et |}
  (lena : nat)
  (#l1 : Array1.layout lena) (ga1 : Array1.t et l1)
  (#l2 : Array1.layout lena) (ga2 : Array1.t et l2)
  (s1 s2 : erased (lseq et lena))
  (vr1 vr2 : erased (lseq real lena) { s1 %~ vr1 /\ s2 %~ vr2 })
  (#_: squash ( len s1 == lena /\ len s2 == lena ))
  (tid : natlt lena)
  : slprop
  = HR.array1_pts_to_slice ga2 tid (tid + 1) seq![s2 @! tid] **
    if_ (tid = 0)
      (HR.array1_pts_to_slice_sum ga1 0 lena (rsmul vr1 vr2))

inline_for_extraction noextract
fn kf
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp{lena <= max_threads})
  (#l1 : Array1.layout lena) (ga1 : Array1.t et l1)
  (#l2 : Array1.layout lena) (ga2 : Array1.t et l2)
  {| ctlayout l1, ctlayout l2 |}
  (#s1  #s2 : erased (lseq et lena))
  (#vr1 #vr2 : erased (lseq real lena) { s1 %~ vr1 /\ s2 %~ vr2 })
  (#_: squash ( len s1 == lena /\ len s2 == lena ))
  (tid : szlt lena)
  ()
  requires
    gpu **
    kpre lena ga1 ga2 s1 s2 vr1 vr2 tid **
    thread_id lena tid **
    B.barrier_tok (mbarrier_contract #lena (HR.barrier_matrix lena ga1 (rsmul vr1 vr2))) **
    B.barrier_state 0
  ensures
    gpu **
    kpost lena ga1 ga2 s1 s2 vr1 vr2 tid **
    thread_id lena tid **
    B.barrier_tok (mbarrier_contract #lena (HR.barrier_matrix lena ga1 (rsmul vr1 vr2))) **
    B.barrier_state (HR.hreduce_barrier_count lena)
{
  (**)unfold (kpre lena ga1 ga2 s1 s2 vr1 vr2 tid);

  let v1 = HR.array1_read_from_slice ga1 tid;
  let v2 = HR.array1_read_from_slice ga2 tid;

  let vm = mul v1 v2;

  HR.array1_write_to_slice ga1 tid vm;

  (* Convince the SMT solver that these sequences are equal *)
  with s'.
    assert (HR.array1_pts_to_slice ga1 tid (tid+1) s');
  assert (pure (Seq.equal s' seq![pmul s1 s2 @! tid]));
  assert (HR.array1_pts_to_slice ga1 tid (tid+1) (seq![pmul s1 s2 @! tid]));

  (* Reduction *)
  HR.kf lena ga1 (rsmul vr1 vr2) tid ();

  fold (kpost lena ga1 ga2 s1 s2 vr1 vr2 tid);
  ()
}

ghost
fn block_setup
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp{SZ.v lena <= max_threads})
  (#l1 : Array1.layout lena) (ga1 : Array1.t et l1)
  (#l2 : Array1.layout lena) (ga2 : Array1.t et l2)
  (#s1 #s2 : lseq et lena)
  (#vr1 #vr2 : seq real { s1 %~ vr1 /\ s2 %~ vr2 })
  (_: squash ( len s1 == SZ.v lena /\ len s2 == SZ.v lena ))
  norewrite
  requires
    ga2 |-> s2 ** ga1 |-> s1
  ensures
    (forall+ (bid : natlt lena). kpre lena ga1 ga2 s1 s2 vr1 vr2 bid) **
    emp (* frame *)
{
  Array1.explode ga1;
  Array1.explode ga2;
  forevery_zip
    (fun (i : natlt (v lena)) -> Cell ga1 i |-> (s1 @! i))
    (fun (i : natlt (v lena)) -> Cell ga2 i |-> (s2 @! i));
  forevery_map
    (fun (i: natlt (v lena)) ->
      Cell ga1 i |-> (s1 @! i) **
      Cell ga2 i |-> (s2 @! i))
    (kpre lena ga1 ga2 s1 s2 vr1 vr2)
    fn bid {
      forevery_singleton_intro'
        #(x:nat{bid <= x /\ x < bid + 1})
        (fun x -> Cell ga1 (x <: natlt lena) |-> (seq![s1 @! bid] @! (x - bid)))
        bid;
      fold HR.array1_pts_to_slice ga1 bid (bid + 1) seq![s1 @! bid];
      forevery_singleton_intro'
        #(x:nat{bid <= x /\ x < bid + 1})
        (fun x -> Cell ga2 (x <: natlt lena) |-> (seq![s2 @! bid] @! (x - bid)))
        bid;
      fold HR.array1_pts_to_slice ga2 bid (bid + 1) seq![s2 @! bid];
      fold kpre lena ga1 ga2 s1 s2 vr1 vr2 bid;
    };
}

ghost
fn block_teardown
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp{SZ.v lena <= max_threads})
  (#l1 : Array1.layout lena) (ga1 : Array1.t et l1)
  (#l2 : Array1.layout lena) (ga2 : Array1.t et l2)
  (#s1 #s2 : lseq et lena)
  (#vr1 #vr2 : seq real { s1 %~ vr1 /\ s2 %~ vr2 })
  (_: squash ( len s1 == SZ.v lena /\ len s2 == SZ.v lena /\
              SZ.fits (Array1.layout_size l1) /\ SZ.fits (Array1.layout_size l2) ))
  norewrite
  requires
    (forall+ (tid : natlt lena). kpost lena ga1 ga2 s1 s2 vr1 vr2 tid) **
    emp
  ensures
    ga2 |-> s2 **
    (exists* (s1' : lseq et lena).
      (ga1 |-> s1') **
      pure ((s1' @! 0) %~ rsum (rsmul vr1 vr2)))
{
  (* Step 1: Unfold each kpost into its components *)
  forevery_map (kpost lena ga1 ga2 s1 s2 vr1 vr2)
    (fun tid ->
      HR.array1_pts_to_slice ga2 tid (tid + 1) seq![s2 @! tid] **
      (if_ (tid = 0) (HR.array1_pts_to_slice_sum ga1 0 lena (rsmul vr1 vr2))))
    fn tid { unfold (kpost lena ga1 ga2 s1 s2 vr1 vr2 tid) };

  (* Step 2: Split the forall+ into ga2 slices and ga1 results *)
  forevery_unzip _ _;

  (* Step 3: Reassemble ga2 — convert slices back to cells, then implode *)
  forevery_map
    (fun (tid : natlt lena) -> HR.array1_pts_to_slice ga2 tid (tid + 1) seq![s2 @! tid])
    (fun (tid : natlt lena) -> Cell ga2 tid |-> (s2 @! tid))
    fn tid {
      unfold HR.array1_pts_to_slice ga2 tid (tid + 1) seq![s2 @! tid];
      forevery_extract #(x:nat{tid <= x /\ x < tid + 1}) tid _;
      drop_ (Pulse.Lib.Trade.trade _ _);
    };
  Array1.implode ga2;

  (* Step 4: Extract ga1 result from thread 0 — follow HReduce's teardown *)
  forevery_map
    (fun (j:natlt lena) ->
      if_ (j = 0) (HR.array1_pts_to_slice_sum ga1 0 (SZ.v lena) (rsmul vr1 vr2)))
    (fun (j:natlt lena) ->
      if_ (op_Equality #(natlt lena) j 0) (HR.array1_pts_to_slice_sum ga1 0 (SZ.v lena) (rsmul vr1 vr2)))
    fn j {};
  forevery_if_elim #(natlt lena) 0 (fun (x: natlt lena) ->
    HR.array1_pts_to_slice_sum ga1 0 (v lena) (rsmul vr1 vr2));

  (* Step 5: Unfold slice_sum and convert to ga1 |-> s1' *)
  unfold HR.array1_pts_to_slice_sum ga1 0 lena (rsmul vr1 vr2);
  with s. assert HR.array1_pts_to_slice_sum_inner ga1 0 lena (rsmul vr1 vr2) s;
  unfold HR.array1_pts_to_slice_sum_inner ga1 0 lena (rsmul vr1 vr2) s;
  with ss. assert HR.array1_pts_to_slice ga1 0 lena ss;
  unfold HR.array1_pts_to_slice ga1 0 lena ss;

  (* Convert the type refinement for forall+ index. Again, would prefer
  to use refinement extensionality here but that fails. Work around with
  a bijection. *)
  let bij : Kuiper.Bijection.bijection (k:nat{0 <= k /\ k < lena}) (Array1.ait lena) =
    Kuiper.Bijection.Mkbijection
      #(k:nat{0 <= k /\ k < lena})
      #(Array1.ait lena)
      (fun k -> k)
      (fun k -> k)
      ez ez;
  forevery_iso bij _;
  forevery_ext _ (fun (k : natlt lena) -> Cell (ga1 <: Array1.t et l1) k |-> (ss @! k));
  Array1.implode ga1;
}

inline_for_extraction noextract
let dp_kernel
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp{SZ.v lena <= max_threads})
  (#l1 : Array1.layout lena) (ga1 : Array1.t et l1 { Array1.is_global ga1 })
  (#l2 : Array1.layout lena) (ga2 : Array1.t et l2 { Array1.is_global ga2 })
  {| ctlayout l1, ctlayout l2 |}
  (#s1 #s2 : erased (lseq et lena))
  (vr1 vr2 : erased (lseq real lena) { s1 %~ vr1 /\ s2 %~ vr2 })
  : kernel_desc
      (ga2 |-> s2 ** ga1 |-> s1)
      (ga2 |-> s2 ** (exists* (s1' : lseq et lena). (ga1 |-> s1') **
                                     pure ((s1' @! 0) %~ rsum (rsmul vr1 vr2))))
  = {
    nthr = lena;
    f = kf lena ga1 ga2;

    barrier_contract = mbarrier_contract #lena (HR.barrier_matrix lena ga1 (rsmul vr1 vr2));
    barrier_count    = HR.hreduce_barrier_count lena;
    barrier_ok       = mbarrier_transform _;

    block_setup    = block_setup lena ga1 ga2;
    block_teardown = block_teardown lena ga1 ga2;

    kpre  = kpre  lena ga1 ga2 s1 s2 vr1 vr2;
    kpost = kpost lena ga1 ga2 s1 s2 vr1 vr2;

    frame = emp;

    kpost_sendable     = solve;
    kpre_sendable      = solve;
    full_post_sendable = solve;
    full_pre_sendable  = solve;
  } <: kernel_desc_1_n_barr _ _

inline_for_extraction noextract
fn dotprod
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp{lena <= max_threads})
  (a1 a2 : vec et)
  (#v1 #v2 : erased (lseq et lena))
  (vr1 vr2 : erased (lseq real lena) { v1 %~ vr1 /\ v2 %~ vr2 })
  norewrite
  preserves
    cpu ** a1 |-> v1 ** a2 |-> v2
  returns
    dp: et
  ensures
    pure (dp %~ sum (pmul vr1 vr2))
{
  Pulse.Lib.Vec.pts_to_len a1;
  Pulse.Lib.Vec.pts_to_len a2;

  let ga1 = Array1.alloc0 #et lena (Kuiper.Tensor.Layout.Alg.l1_forward _);
  let ga2 = Array1.alloc0 #et lena (Kuiper.Tensor.Layout.Alg.l1_forward _);

  Array1.memcpy_host_to_device ga1 a1 lena;
  Array1.memcpy_host_to_device ga2 a2 lena;

  (* Why are the implicits needed? *)
  launch_sync (dp_kernel lena ga1 ga2 vr1 vr2);

  // gpu_pts_to_ref_located ga1;
  // gpu_pts_to_ref_located ga2;

  assert (pure (Seq.length v1 == lena));
  assert (pure (Seq.length v2 == lena));
  assert (pure (0sz + 1sz <= lena));

  (* swap space *)
  let ar = V.alloc #et zero 1sz;
  (* inference sucks here, what's going on? *)
  Array1.memcpy_device_to_host' #_ #_ #1 //the dst_sz cannot be computed by unification;
      ar 0sz //#_
      ga1 0sz 1sz;

  Array1.free ga1;
  Array1.free ga2;

  let dp = ar.(0sz);
  V.free ar;

  (* Finally, ensure that the reduction must be sum *)
  assert pure (rsmul vr1 vr2 `Seq.equal` pmul vr1 vr2);

  dp
}
