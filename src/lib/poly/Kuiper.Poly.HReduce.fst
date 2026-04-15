module Kuiper.Poly.HReduce

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Seq.Common
open Kuiper.Tensor { ctlayout }

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
  (* FIXME: implement. *)
  admit();
}

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
    pure ((s @! 0) %~ real_seq_sum (Seq.slice rr i j))

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

unfold
let barrier_matrix
  (#et:Type0) {| scalar et, real_like et |}
  (nth : nat)
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
  (nth : nat)
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

unfold
let kpre
  (#et:Type0) {| scalar et, real_like et |}
  (lena : nat)
  (#l : Array1.layout lena)
  (a : Array1.t et l)
  (s : lseq et lena)
  (vr : seq Real.real)
  (#_: squash (len s == lena))
  (tid : natlt lena)
  : slprop
  = array1_pts_to_slice a tid (tid +1) seq![s @! tid]

unfold
let kpost
  (#et:Type0) {| scalar et, real_like et |}
  (lena : nat)
  (#l : Array1.layout lena)
  (a : Array1.t et l)
  (vr : lseq real lena)
  (tid : natlt lena)
  : slprop
  = if_ (tid = 0) (array1_pts_to_slice_sum a 0 lena vr)

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
          (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
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
      (**)assert (pure (s1 `approximates` real_seq_sum (Seq.slice vr tid nextid)));

      let s2 = array1_read_from_slice r nextid;
      (**)assert (pure (s2 `approximates` real_seq_sum (Seq.slice vr nextid end_)));

      let s = add s1 s2;
      (**)lem_append_slice vr tid nextid end_;
      (**)seq_approximates_append s1 s2 (Seq.slice vr tid nextid) (Seq.slice vr nextid end_);
      (**)assert (pure ((s1 `add` s2) `approximates` real_seq_sum (Seq.append (Seq.slice vr tid nextid) (Seq.slice vr nextid end_))));
      (**)real_seq_sum_append (Seq.slice vr tid nextid) (Seq.slice vr nextid end_);
      (**)assert (pure (s `approximates` real_seq_sum (Seq.slice vr tid end_)));

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
          (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
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
  (nth : szp { nth <= max_threads })
  (#l : Array1.layout nth) {| ctlayout l |}
  (a : Array1.t et l)
  (#s : erased (lseq et nth))
  (vr : erased (lseq real nth){ s %~ vr })
  (tid : szlt nth)
  ()
  requires
    gpu **
    kpre nth a s vr tid **
    thread_id nth tid **
    mbarrier_tok nth (barrier_matrix nth a vr) **
    B.barrier_state 0
  ensures
    gpu **
    kpost nth a vr tid **
    thread_id nth tid **
    mbarrier_tok nth (barrier_matrix nth a vr) **
    B.barrier_state (hreduce_barrier_count nth)
{
  (* Reduction *)
  let mut n : szlt 32 = 0sz;

  (**)with ss. assert (array1_pts_to_slice a tid (tid+1) ss);
  assert (pure (Seq.slice s tid (tid+1) `Seq.equal` seq![ss @! 0])); // sucks
  (**)fold (array1_pts_to_slice_sum a tid (tid + 1) vr);
  (**)if_intro_true (array1_pts_to_slice_sum a tid (tid + 1) vr);

  open FStar.SizeT;
  while (spow2 !n <^ nth)
    invariant
      live n **
      B.barrier_state !n **
      if_ (div_pow2 !n tid) (array1_pts_to_slice_sum a tid (min (tid + pow2 !n) nth) vr) **
      pure (v !n > 0 ==> pow2 (v !n - 1) < v nth)
    decreases (2 * nth - spow2 !n)
  {
    assert pure (Seq.length s == SZ.v nth);
    iteration nth a vr tid !n;
    n := !n +^ 1sz;
  };

  with it. assert (B.barrier_state it);

  // After loop exit: pow2 it >= nth, and tid < nth, so div_pow2 it tid <==> tid = 0
  FStar.Math.Lemmas.modulo_lemma tid (pow2 it);
  rewrite
    (if_ (div_pow2 it tid) (array1_pts_to_slice_sum a tid (min (tid + pow2 it) nth) vr))
  as
    (if_ (op_Equality #nat tid 0) (array1_pts_to_slice_sum a 0 nth vr));

  log2_hreduce (v nth) it;
  rewrite (B.barrier_state it) as (B.barrier_state (hreduce_barrier_count nth));

  ()
}

ghost
fn block_setup
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp { lena <= max_threads })
  (#l : Array1.layout lena)
  (a : Array1.t et l)
  (#va : lseq et lena)
  (#vr : seq real { va %~ vr })
  (#_ : squash (Seq.length va == SZ.v lena))
  ()
  norewrite
  requires
    a |-> va
  ensures
    (forall+ (i : natlt lena). kpre lena a va vr i) **
    emp
{
  Array1.explode a;
  forevery_map
    (fun (i:natlt lena) -> Cell a i |-> (va @! i))
    (fun (i:natlt lena) -> kpre lena a va vr i)
    fn i {
      forevery_singleton_intro'
        #(x:nat{i <= x /\ x < i + 1})
        (fun x -> Cell a (x <: natlt lena) |-> (seq![va @! i] @! (x - i)))
        i;
      fold array1_pts_to_slice a i (i + 1) seq![va @! i];
      fold kpre lena a va vr i;
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
  (#vr : seq real { va %~ vr })
  ()
  norewrite
  requires
    (forall+ (i : natlt lena). kpost lena a vr i) **
    emp
  ensures
    array1_pts_to_slice_sum a 0 lena vr
{
  // Adjust type of equality...
  forevery_map
    (fun (j:natlt lena) ->
      if_ (j = 0) (array1_pts_to_slice_sum a 0 (SZ.v lena) vr))
    (fun (j:natlt lena) ->
      if_ (op_Equality #(natlt lena) j 0) (array1_pts_to_slice_sum a 0 (SZ.v lena) vr))
    fn j {};
  forevery_if_elim #(natlt lena) 0 (fun (x: natlt lena) ->
    array1_pts_to_slice_sum a 0 (v lena) vr);
}


inline_for_extraction noextract
let kernel
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp { lena <= max_threads })
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l { Array1.is_global a })
  (#va : erased (lseq et lena))
  (#vr : erased (lseq real lena) { va %~ vr })
  (#_ : squash (Seq.length va == SZ.v lena))
  : kernel_desc_1_n_barr
      (a |-> va)
      (array1_pts_to_slice_sum a 0 lena vr)
  = {
    nthr = lena;

    barrier_contract = mbarrier_contract (barrier_matrix lena a vr);
    barrier_count    = hreduce_barrier_count lena;
    barrier_ok       = mbarrier_transform (barrier_matrix lena a vr);

    f = kf lena a vr;

    block_setup = block_setup lena a #va;
    block_teardown = block_teardown lena a #va;
    kpre =  kpre lena a va vr;
    kpost = kpost lena a vr;
    frame = emp;

    kpre_sendable      = magic();
    kpost_sendable     = magic();
    full_post_sendable = magic();
    full_pre_sendable  = magic();
  }

inline_for_extraction noextract
fn reduce
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp { lena <= max_threads })
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l { Array1.is_global a })
  (#va : erased (lseq et lena))
  (vr : erased (lseq real lena))
  preserves
    cpu
  requires
    on gpu_loc (a |-> va) **
    pure (va %~ vr)
  ensures
    (exists* (va' : lseq et lena).
      on gpu_loc (a |-> va') **
      pure ((va' @! 0) %~ seq_fold_left (+.) 0.0R vr))
{
  map_loc gpu_loc
    #(a |-> va)
    #(a |-> va ** pure (SZ.fits (Array1.layout_size l)))
    fn _ { Array1.pts_to_ref a; };
  launch_sync (kernel lena a #va #vr);
  map_loc gpu_loc
    #(array1_pts_to_slice_sum a 0 lena vr)
    #(exists* (va' : lseq et lena).
        a |-> va' **
        pure ((va' @! 0) %~ seq_fold_left (+.) 0.0R vr))
    fn _ {
      unfold array1_pts_to_slice_sum a 0 lena vr;
      with s. assert array1_pts_to_slice a 0 lena s;
      unfold array1_pts_to_slice a 0 lena s;
      assert pure (forall (k:nat). 0 <= k /\ k < lena <==> k < lena);
      // FStar.RefinementExtensionality.refext nat (fun (k:nat) -> 0 <= k /\ k < lena) (fun (k:nat) -> k < lena);
      assume pure ((k : nat{0 <= k /\ k < lena}) == Array1.ait lena);
      forevery_rw_type (k : nat{0 <= k /\ k < lena}) (Array1.ait lena) _;
      forevery_ext _ (fun (k : natlt lena) -> Cell (a <: Array1.t et l) k |-> (s @! k));
      Array1.implode a;
      ();
    };
  ();
}
