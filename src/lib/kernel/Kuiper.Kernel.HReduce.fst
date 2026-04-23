module Kuiper.Kernel.HReduce

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Seq.Common
open Kuiper.Tensor { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }

module SZ = Kuiper.SizeT
module RPM = Kuiper.Barrier.RPM
module B = Kuiper.Barrier

(* Plain ownership of a slice of an Array1. *)
let array1_pts_to_slice
  (#et : Type0)
  (#sz : nat)
  (#l : Array1.layout sz)
  ([@@@mkey] r : Array1.t et l)
  ([@@@mkey]i
   [@@@mkey]j : nat{i <= j /\ j <= sz})
  (s : lseq et (j - i))
  : slprop
  = forall+ (k : nat{i <= k /\ k < j}).
      Cell r (k <: natlt sz) |-> (s @! (k - i))

#push-options "--z3rlimit 80"
ghost
fn array1_slice_concat
  (#et : Type0)
  (#sz : nat)
  (#l : Array1.layout sz)
  (r : Array1.t et l)
  (i j k : nat{i <= j /\ j <= k /\ k <= sz})
  (#s1 : lseq et (j - i))
  (#s2 : lseq et (k - j))
  requires
    array1_pts_to_slice r i j s1 **
    array1_pts_to_slice r j k s2
  ensures
    array1_pts_to_slice r i k (s1 @+ s2)
{
  unfold array1_pts_to_slice r i j s1;
  unfold array1_pts_to_slice r j k s2;

  let s = s1 @+ s2;

  (* Rewrite each side to use s *)
  forevery_ext
    (fun (x:nat{i <= x /\ x < j}) -> Cell r (x <: natlt sz) |-> (s1 @! (x - i)))
    (fun (x:nat{i <= x /\ x < j}) -> Cell r (x <: natlt sz) |-> (s @! (x - i)));
  forevery_ext
    (fun (x:nat{j <= x /\ x < k}) -> Cell r (x <: natlt sz) |-> (s2 @! (x - j)))
    (fun (x:nat{j <= x /\ x < k}) -> Cell r (x <: natlt sz) |-> (s @! (x - i)));

  (* Join *)
  forevery_refine_join' #nat
    (fun (x:nat) -> i <= x /\ x < j)
    (fun (x:nat) -> j <= x /\ x < k)
    (fun (x:nat{(i <= x /\ x < j) \/ (j <= x /\ x < k)}) ->
      Cell r (x <: natlt sz) |-> (s @! (x - i)));

  (* Simplify *)
  forevery_refine_ext' #nat
    #(fun (x:nat) -> (i <= x /\ x < j) \/ (j <= x /\ x < k))
    (fun (x:nat) -> i <= x /\ x < k)
    (fun (x:nat{(i <= x /\ x < j) \/ (j <= x /\ x < k)}) ->
      Cell r (x <: natlt sz) |-> (s @! (x - i)));

  fold array1_pts_to_slice r i k s;
}
#pop-options

inline_for_extraction noextract
fn array1_read_from_slice
  (#et : Type0)
  (#len : erased nat)
  (#l : Array1.layout len) {| ctlayout l |}
  (r : Array1.t et l)
  (#i #j : erased nat{i <= j /\ j <= len})
  (idx : sz{i <= idx /\ idx < j})
  (#s : erased (lseq et (j - i)))
  preserves
    array1_pts_to_slice r i j s
  returns
    v : et
  ensures
    pure (v == s @! (idx - i))
{
  unfold array1_pts_to_slice r i j s;
  forevery_extract #(x:nat{i <= x /\ x < j}) (SZ.v idx) _;
  let v = Array1.read_cell r idx;
  Pulse.Lib.Trade.elim_trade _ _;
  fold array1_pts_to_slice r i j s;
  v
}

inline_for_extraction noextract
fn array1_write_to_slice
  (#et : Type0)
  (#len : erased nat)
  (#l : Array1.layout len) {| ctlayout l |}
  (r : Array1.t et l)
  (#i #j : erased nat{i <= j /\ j <= len})
  (idx : sz{i <= idx /\ idx < j})
  (#s : erased (lseq et (j - i)))
  (v : et)
  requires
    array1_pts_to_slice r i j s
  ensures
    array1_pts_to_slice r i j (Seq.upd s (idx - i) v)
{
  unfold array1_pts_to_slice r i j s;
  forevery_extract' #(x:nat{i <= x /\ x < j}) (SZ.v idx) _;
  Array1.write_cell r idx v;
  let s' : erased (lseq et (j - i)) = Seq.upd s (idx - i) v;
  Pulse.Lib.Forall.elim_forall
    (fun (x:nat{i <= x /\ x < j}) ->
      Cell r (x <: natlt len) |-> (s' @! (x - i)));
  Pulse.Lib.Trade.elim_trade _ _;
  fold array1_pts_to_slice r i j s';
  rewrite each s' as Seq.upd s (idx - i) v;
  ()
}

(* Ownership of array r between i and j. The first value of that slice
is the reduction of all the values in the (original) slice v. *)
unfold
let array1_pts_to_slice_sum_inner
  (#et:Type0) {| scalar et, real_like et |}
  (#sz : nat)
  (#l : Array1.layout sz)
  (r : Array1.t et l)
  (i j : nat{i < j /\ j <= sz})
  (rr : lseq real sz)
  (s : lseq et (j - i))
  : slprop
  = array1_pts_to_slice r i j s **
    pure ((s @! 0) %~ rsum (Seq.slice rr i j))

let array1_pts_to_slice_sum
  (#et:Type0) {| scalar et, real_like et |}
  (#sz : nat)
  (#l : Array1.layout sz)
  ([@@@mkey] r : Array1.t et l)
  ([@@@mkey] i : nat)
  (j : nat{i < j /\ j <= sz})
  (rr : lseq real sz)
  : slprop
  = exists* s. array1_pts_to_slice_sum_inner r i j rr s

// Barrier

let barrier_matrix
  (#et:Type0) {| scalar et, real_like et |}
  (nth : szp)
  (#l : Array1.layout nth)
  (r : Array1.t et l)
  (vr : lseq real nth)
  (it : nat)
  (from to : natlt nth)
: slprop
=
  if_ (from = to + pow2 it)
      (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
           (array1_pts_to_slice_sum r from (min (from + pow2 it) nth) vr))

ghost
fn mk_barrier_pre
  (#et:Type0) {| scalar et, real_like et |}
  (nth : szp)
  (#l : Array1.layout nth)
  (r : Array1.t et l)
  (vr : lseq real nth)
  (tid : natlt nth)
  (it: natlt 31)
  requires
    if_ (not (div_pow2 (it + 1) tid) && div_pow2 it tid)
      (array1_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vr)
  ensures
    forall+ (i:natlt nth). barrier_matrix nth r vr it tid i
{
  open FStar.SizeT;
  if (tid >= pow2 it) {
    forevery_if_intro #(natlt nth) (tid - pow2 it) (fun i ->
      if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid))
        (array1_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vr));
    forevery_ext
      (fun (i:natlt nth) ->
        if_ (op_Equality #(natlt nth) i (tid - pow2 it))
          (if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid))
            (array1_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vr)))
      (fun (i:natlt nth) -> barrier_matrix nth r vr it tid i);
  } else {
    if_elim_false _;
    forevery_emp_intro (natlt nth);
    forevery_ext
      (fun (i:natlt nth) -> emp)
      (fun (i:natlt nth) -> barrier_matrix nth r vr it tid i);
  }
}

// RO permission to a, thread 0 also owns output ref, full ownership of own cell
// in shmem array
unfold
let kpre
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp { lena <= max_threads })
  (#l : Array1.layout lena)
  (a : Array1.t et l)
  (va : lseq et lena)
  (vr : lseq real lena)
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et lena])
  (bid : natlt 1)
  (tid : natlt lena)
  : slprop
  = a |-> Frac (1 /. lena) va **
    if_ (op_Equality #nat tid 0) (live out) **
    exists* (v : et). Cell (Array1.from_array (l1_forward lena) shmem._1) tid |-> v

// Same RO permission to a, 1st thread has full ownership of shmem plus of the
// output reference.  No need to specify the contents of the shmem array, it
// will disappear.
unfold
let kpost
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp { lena <= max_threads })
  (#l : Array1.layout lena)
  (a : Array1.t et l)
  (va : lseq et lena)
  (vr : lseq real lena)
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et lena])
  (bid : natlt 1)
  (tid : natlt lena)
  : slprop
  = a |-> Frac (1 /. lena) va **
    if_ (op_Equality #nat tid 0) (
      live (Array1.from_array (l1_forward lena) shmem._1) **
      exists* (v : et). out |-> v ** pure (v %~ rsum vr)
    )

inline_for_extraction
fn iteration
  (#et:Type0) {| scalar et, real_like et |}
  (nth : szp { SZ.v nth <= max_threads })
  (#l : Array1.layout nth) {| Kuiper.Tensor.ctlayout l |}
  (r : Array1.t et l)
  (vr : erased (lseq real nth))
  (tid : szlt nth)
  (it: szlt 31)
  preserves gpu
  preserves thread_id nth tid
  preserves mbarrier_tok nth (barrier_matrix nth r vr)
  requires B.barrier_state it
  requires if_ (div_pow2 it tid) (array1_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vr)
  ensures  B.barrier_state (it + 1)
  ensures  if_ (div_pow2 (it+1) tid) (array1_pts_to_slice_sum r tid (min (tid + pow2 (it + 1)) nth) vr)
{
  case_split (div_pow2 (it + 1) tid)
    (if_ (div_pow2 it tid) (array1_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vr));
  if_flatten #(div_pow2 (it + 1) tid);
  if_flatten #(not (div_pow2 (it + 1) tid));

  div_pow2_lemma it (it + 1) tid;
  rewrite (if_ (div_pow2 (it + 1) tid && div_pow2 it tid)
            (array1_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vr))
      as (if_ (div_pow2 (it + 1) tid)
            (array1_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vr));

  mk_barrier_pre nth r vr tid it;
  fold RPM.row (barrier_matrix nth r vr) it tid;
  mbarrier_wait ();
  unfold RPM.col (barrier_matrix nth r vr) it tid;

  // combine (div_pow2 (it + 1) tid) (array1_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv) _;

  let nextid = FStar.SizeT.(tid +^ spow2 it);

  (* We do not use end_ in extracted code, so we can use a nat and erase it
  so there are no traces in the extracted C. *)
  let end_ : erased nat = hide (min (tid + 2 * pow2 it) nth);

  if (nextid <^ nth) {
    forevery_ext
      (fun (from: natlt nth) ->
        if_ (op_Equality #int from (tid + pow2 it))
          (if_ (not (div_pow2 (it + 1) from) && div_pow2 it from)
            (array1_pts_to_slice_sum r from (min (from + pow2 it) nth) vr)))
      (fun (from: natlt nth) ->
        if_ (op_Equality #(natlt nth) from (tid + pow2 it))
          (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
            (array1_pts_to_slice_sum r from (min (from + pow2 it) nth) vr)));
    forevery_if_elim #(natlt nth)
      (tid + pow2 it)
      (fun (from: natlt nth) -> if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
         (array1_pts_to_slice_sum r from (min (from + pow2 it) nth) vr));

    let b = sdiv_pow2 (it +^ 1sz) tid;

    rewrite each (div_pow2 (it + 1) (SZ.v tid)) as b;

    div_pow2_lemma_2 it tid;
    combine
      b
      (array1_pts_to_slice_sum r nextid (min (tid + pow2 it + pow2 it) nth) vr)
      _;

    if b {
      assert (pure (div_pow2 (SZ.v it + 1) (SZ.v tid)));
      if_elim_true _;

      (**)unfold (array1_pts_to_slice_sum r nextid end_ vr);
      (**)unfold (array1_pts_to_slice_sum r tid nextid vr);
      (**)array1_slice_concat #et #nth r tid nextid end_;

      let s1 = array1_read_from_slice r tid;
      (**)assert (pure (s1 `approximates` rsum (Seq.slice vr tid nextid)));

      let s2 = array1_read_from_slice r nextid;
      (**)assert (pure (s2 `approximates` rsum (Seq.slice vr nextid end_)));

      let s = add s1 s2;
      (**)lem_append_slice vr tid nextid end_;
      (**)seq_approximates_append s1 s2 (Seq.slice vr tid nextid) (Seq.slice vr nextid end_);
      (**)assert (pure ((s1 `add` s2) `approximates` rsum (Seq.append (Seq.slice vr tid nextid) (Seq.slice vr nextid end_))));
      (**)rsum_append (Seq.slice vr tid nextid) (Seq.slice vr nextid end_);
      (**)assert (pure (s `approximates` rsum (Seq.slice vr tid end_)));

      // gpu_array_write r tid s;
      array1_write_to_slice r tid s;

      (**)with seq. assert (array1_pts_to_slice r tid end_ seq);
      (**)fold (array1_pts_to_slice_sum r tid end_ vr);
      (**)if_intro_true (array1_pts_to_slice_sum r tid end_ vr);
      // Step below optional right now, but good practice?
      (**)rewrite
      (**)  if_ true
      (**)      (array1_pts_to_slice_sum r (SZ.v tid) (reveal end_) vr)
      (**)as
      (**)  if_ (div_pow2 (SZ.v it + 1) (SZ.v tid))
      (**)      (array1_pts_to_slice_sum r (SZ.v tid) (reveal end_) vr);
    } else {
      (* no-op *)
      if_elim_false _;
      if_intro_false (array1_pts_to_slice_sum r tid end_ vr);
    }
  } else {
    forevery_map
      (fun (from: natlt nth) ->
        if_ (op_Equality #int from (tid + pow2 it))
          (if_ (not (div_pow2 (it + 1) from) && div_pow2 it from)
            (array1_pts_to_slice_sum r from (min (from + pow2 it) nth) vr)))
      (fun from -> emp)
      fn from {
        if_rewrite_bool (from = tid + pow2 it) false _;
        if_elim_false _;
      };
    forevery_emp_elim _;
  }
}

(* Number of barrier calls in the reduction loop: smallest k s.t. pow2 k >= nth *)
let hreduce_barrier_count (nth : pos) : GTot nat = log2 (2 * nth - 1)

(* If pow2 k <= n < pow2 (k+1), then log2 n = k. *)
private let rec log2_range (n:pos) (k:nat)
  : Lemma (requires pow2 k <= n /\ n < pow2 (k+1))
          (ensures log2 n == k)
          (decreases k)
= if k = 0 then ()
  else begin
    FStar.Math.Lemmas.lemma_div_le (pow2 k) n 2;
    log2_range (n/2) (k-1)
  end

(* The smallest k with pow2 k >= nth equals log2 (2*nth - 1). *)
private let log2_hreduce (nth:pos) (it:nat)
  : Lemma (requires pow2 it >= nth /\ (it == 0 \/ pow2 (it - 1) < nth))
          (ensures it == log2 (2 * nth - 1))
= if it = 0 then ()
  else log2_range (2 * nth - 1) it

inline_for_extraction noextract
fn kf
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp { lena <= max_threads })
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l)
  (va : erased (lseq et lena))
  (vr : erased (lseq real lena){ va %~ vr })
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et lena])
  (bid : szlt 1sz)
  (tid : szlt lena)
  ()
  requires
    gpu **
    kpre lena a va vr out shmem bid tid **
    thread_id lena tid **
    block_id 1 bid **
    mbarrier_tok lena (barrier_matrix lena (Array1.from_array (l1_forward lena) shmem._1) vr) **
    B.barrier_state 0
  ensures
    gpu **
    kpost lena a va vr out shmem bid tid **
    thread_id lena tid **
    block_id 1 bid **
    mbarrier_tok lena (barrier_matrix lena (Array1.from_array (l1_forward lena) shmem._1) vr) **
    B.barrier_state (hreduce_barrier_count lena)
{
  let (gsa, _) = shmem;

  let sa = Array1.from_array (l1_forward lena) gsa;
  rewrite each Array1.from_array (l1_forward lena) gsa as sa;

  (* Copy from input to shmem swap. *)
  let myv = Array1.read a tid;
  Array1.write_cell sa tid myv;

  (* Reduction *)
  let mut n : szlt 32 = 0sz;

  forevery_singleton_intro'
    #(x:nat{tid <= x /\ x < tid + 1})
    (fun x -> Cell sa (x <: natlt lena) |-> (seq![myv] @! (x - tid)))
    tid;
  fold array1_pts_to_slice sa tid (tid+1) seq![myv];

  (**)fold (array1_pts_to_slice_sum sa tid (tid + 1) vr);
  (**)if_intro_true' (div_pow2 !n tid) (array1_pts_to_slice_sum sa tid (min (tid + pow2 !n) lena) vr);

  open FStar.SizeT;
  while (spow2 !n <^ lena)
    invariant
      live n **
      B.barrier_state !n **
      if_ (div_pow2 !n tid) (array1_pts_to_slice_sum sa tid (min (tid + pow2 !n) lena) vr) **
      pure (v !n > 0 ==> pow2 (v !n - 1) < v lena)
    decreases (2 * lena - spow2 !n)
  {
    assert pure (Seq.length va == SZ.v lena);
    iteration lena sa vr tid !n;
    n := !n +^ 1sz;
  };

  with it. assert (B.barrier_state it);

  // After loop exit: pow2 it >= lena, and tid < lena, so div_pow2 it tid <==> tid = 0
  FStar.Math.Lemmas.modulo_lemma tid (pow2 it);
  rewrite
    (if_ (div_pow2 it tid) (array1_pts_to_slice_sum sa tid (min (tid + pow2 it) lena) vr))
  as
    (if_ (op_Equality #nat tid 0) (array1_pts_to_slice_sum sa 0 lena vr));

  log2_hreduce (v lena) it;
  rewrite (B.barrier_state it) as (B.barrier_state (hreduce_barrier_count lena));

  (* Thread zero owns the result at the end, and writes it out. *)
  if (tid = 0sz) {
    if_elim_true' (op_Equality #nat tid 0) (array1_pts_to_slice_sum sa 0 lena vr);
    if_elim_true' (op_Equality #nat tid 0) (live out);
    unfold array1_pts_to_slice_sum sa 0 lena vr;
    gpu_write out (array1_read_from_slice sa 0sz);
    // fold array1_pts_to_slice_sum sa 0 lena vr;
    with ss. assert array1_pts_to_slice sa 0 lena ss;
    unfold array1_pts_to_slice sa;
    (* I would prefer to rewrite the type with refinement extensionality,
       but that does not seem to work. *)
    let bij : Kuiper.Bijection.bijection (k:nat{0 <= k /\ k < lena}) (Array1.ait lena) =
      Kuiper.Bijection.Mkbijection
        #(k:nat{0 <= k /\ k < lena})
        #(Array1.ait lena)
        (fun k -> k)
        (fun k -> k)
        ez ez;
    forevery_iso bij _;
    forevery_ext _ (fun (k : natlt lena) -> Cell sa k |-> (ss @! k));
    Array1.implode sa;
    rewrite each sa as Array1.from_array (l1_forward lena) shmem._1;
    if_intro_true' (op_Equality #nat tid 0) (
      live (Array1.from_array (l1_forward lena) shmem._1) **
      exists* (v : et). out |-> v ** pure (v %~ rsum vr)
    )
  } else {
    (* Nop, convince Pulse. *)
    if_elim_false' (op_Equality #nat tid 0) (array1_pts_to_slice_sum sa 0 lena vr);
    if_elim_false' (op_Equality #nat tid 0) (live out);
    if_intro_false' (op_Equality #nat tid 0) (
      live (Array1.from_array (l1_forward lena) shmem._1) **
      exists* (v : et). out |-> v ** pure (v %~ rsum vr)
    );
    ();
  };
}

ghost
fn block_setup
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp { lena <= max_threads })
  (#l : Array1.layout lena)
  (a : Array1.t et l)
  (#va : lseq et lena)
  (vr : lseq real lena { va %~ vr })
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et lena])
  (bid : natlt 1)
  ()
  norewrite
  requires
    live_c_shmems shmem **
    (a |-> va ** live out)
  ensures
    (forall+ (i : natlt lena). kpre lena a va vr out shmem bid i) **
    emp
{
  unfold_live_c_shmems_cons shmem #_;
  unfold_live_c_shmems_nil shmem._2 #_;
  let gsa = shmem._1; rewrite each fst shmem as gsa;
  unfold live_c_shmem gsa;

  with vgsa. assert gsa |-> vgsa;
  gpu_pts_to_ref gsa;

  (* share input *)
  Array1.share_n a lena;

  (* tid 0 gets the ref *)
  forevery_if_intro #(natlt lena) 0 (fun _ -> live out);
  (* Sad.*)
  forevery_ext
    (fun tid -> if_ (op_Equality #(natlt lena) tid 0) (live out))
    (fun tid -> if_ (op_Equality #nat tid 0) (live out));

  forevery_zip (fun _ -> a |-> Frac (1 /. lena) va) _;

  (* View shmem array as Array1. Explode it. *)
  Array1.raise' (l1_forward lena) gsa;
  Array1.explode (Array1.from_array (l1_forward lena) gsa);

  forevery_zip #(natlt lena)
    (fun tid -> a |-> Frac (1 /. lena) va ** if_ (op_Equality #nat tid 0) (live out))
    _;

  forevery_map
    #(natlt lena)
    (fun tid ->
      (a |-> Frac (1 /. lena) va **
       if_ (op_Equality #nat tid 0) (live out)) **
      Cell (Array1.from_array (l1_forward lena) gsa) tid |-> (Array1.from_seq (l1_forward lena) vgsa @! tid)
    )
    (fun (tid : natlt lena) -> kpre lena a va vr out shmem bid tid)
    fn tid {
      rewrite each gsa as shmem._1;
      ();
    };

  ()
}


ghost
fn block_teardown
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp { lena <= max_threads })
  (#l : Array1.layout lena)
  (a : Array1.t et l)
  (#va : lseq et lena)
  (vr : lseq real lena { va %~ vr })
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et lena])
  (bid : natlt 1)
  ()
  norewrite
  requires
    (forall+ (i : natlt lena). kpost lena a va vr out shmem bid i) **
    emp
  ensures
    live_c_shmems shmem **
    (a |-> va ** (exists* (v : et). out |-> v ** pure (v %~ rsum vr)))
{
  forevery_unzip _ _;

  Array1.gather_n a lena;

  (* Sad.*)
  forevery_ext #(natlt lena)
    (fun tid ->
      if_ (op_Equality #nat tid 0) (
        live (Array1.from_array (l1_forward lena) shmem._1) **
        exists* (v : et). out |-> v ** pure (v %~ rsum vr)))
    (fun tid ->
      if_ (op_Equality #(natlt lena) tid 0) (
        live (Array1.from_array (l1_forward lena) shmem._1) **
        exists* (v : et). out |-> v ** pure (v %~ rsum vr)));

  forevery_if_elim #(natlt lena) 0 (fun tid ->
      live (Array1.from_array (l1_forward lena) shmem._1) **
      exists* (v : et). out |-> v ** pure (v %~ rsum vr)
  );

  Array1.lower (Array1.from_array (l1_forward lena) shmem._1);
  rewrite each Array1.core (Array1.from_array (l1_forward lena) shmem._1) as shmem._1;

  fold_live_c_shmems_nil shmem._2 #_;
  with vgsa. assert shmem._1 |-> vgsa;
  fold_live_c_shmem shmem._1;
  fold_live_c_shmems_cons shmem #_;
}

ghost
fn setup
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp { lena <= max_threads })
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l { Array1.is_global a })
  (#va : erased (lseq et lena))
  (vr : erased (lseq real lena) { va %~ vr })
  (#_ : squash (Seq.length va == SZ.v lena))
  (out : gpu_ref et)
  ()
  norewrite
  requires
    a |-> va ** live out
  ensures
    (forall+ (bid : natlt 1). a |-> va ** live out) **
    emp
{
  forevery_singleton_intro #(natlt 1) (fun _bid -> a |-> va ** live out);
}

ghost
fn teardown
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp { lena <= max_threads })
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l { Array1.is_global a })
  (#va : erased (lseq et lena))
  (vr : erased (lseq real lena) { va %~ vr })
  (#_ : squash (Seq.length va == SZ.v lena))
  (out : gpu_ref et)
  ()
  norewrite
  requires
    (forall+ (bid : natlt 1). a |-> va ** exists* (v : et). out |-> v ** pure (v %~ rsum vr)) **
    emp
  ensures
    a |-> va ** (exists* (v : et). out |-> v ** pure (v %~ rsum vr))
{
  forevery_singleton_elim #(natlt 1) _;
}

inline_for_extraction noextract
let kernel
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp { lena <= max_threads })
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l { Array1.is_global a })
  (#va : erased (lseq et lena))
  (vr : erased (lseq real lena) { va %~ vr })
  (#_ : squash (Seq.length va == SZ.v lena))
  (out : gpu_ref et)
  : kernel_desc
      (a |-> va ** live out)
      (a |-> va ** exists* (v : et). out |-> v ** pure (v %~ rsum vr))
  = {
    nblk = 1sz;
    nthr = lena;

    shmems_desc = [SHArray et lena];

    barrier_contract = (fun _bid shmem ->
      mbarrier_contract (barrier_matrix #et lena (Array1.from_array _ shmem._1) vr));
    barrier_count    = (fun _bid    -> hreduce_barrier_count lena);
    barrier_ok       = (fun _bid shmem ->
      mbarrier_transform (barrier_matrix lena #(l1_forward lena) (Array1.from_array _ shmem._1) vr));

    f = kf lena a va vr out;

    block_pre  = (fun bid -> a |-> va ** live out);
    block_post = (fun bid -> a |-> va ** exists* (v : et). out |-> v ** pure (v %~ rsum vr));
    setup      = setup    lena a #va vr out;
    teardown   = teardown lena a #va vr out;

    block_frame    = (fun _shmem _bid -> emp);
    block_setup    = block_setup    lena a #va vr out;
    block_teardown = block_teardown lena a #va vr out;

    kpre =  kpre  lena a va vr out;
    kpost = kpost lena a va vr out;
    frame = emp;

    // FIXME: kpre and kpost mention a non-global array, but tc resolution tries
    // to apply the instance for global arrays anyway, and fails to prove the
    // refinement.
    kpre_sendable       = magic();
    kpost_sendable      = magic();
    block_post_sendable = solve;
    block_pre_sendable  = solve;
  }

inline_for_extraction noextract
fn reduce
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp { lena <= max_threads })
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l { Array1.is_global a })
  (#va : erased (lseq et lena))
  (vr : erased (lseq real lena))
  norewrite // sigh... spec in fsti is not purified
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    pure (va %~ vr)
  returns
    res : et
  ensures
    pure (res %~ rsum vr)
{
  let out = Kuiper.Ref.gpu_alloc0 #et ();
  launch_sync (kernel lena a vr out);

  (* Bring back out result, free swap. *)
  let mut hout : et = zero #et;
  Kuiper.Ref.gpu_memcpy_device_to_host hout out;
  Kuiper.Ref.gpu_free out;

  !hout;
}
