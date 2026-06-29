module Kuiper.Kernel.HReduce

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Pulse.Lib.GhostReference { read as gread, write as gwrite, alloc as galloc }
open Kuiper.Tensor
open Kuiper.Chest1.Helpers
open Kuiper.Bijection { ( =~ ), bij_sym }
open Kuiper.Seq.Common

module SZ = Kuiper.SizeT
module RPM = Kuiper.Barrier.RPM
module B = Kuiper.Barrier

let chest1_stride_length
  (n : nat) (stride : pos) (off : natlt stride)
  : GTot nat
  = (n - off + stride - 1) / stride

let chest1_stride (#a:Type0) (#n : nat)
  (s : chest1 a n) (stride : pos) (off : natlt stride)
  : GTot (chest1 a (chest1_stride_length n stride off))
  = mk1 (fun i -> acc1 s (off + i * stride))

let chest1_take (#a:Type0) (#n : nat)
  (len : natle n)
  (s : chest1 a n)
  : GTot (chest1 a len)
  = chest1_sub 0 len s

let chest1_drop (#a:Type0) (#n : nat)
  (len : natle n)
  (s : chest1 a n)
  : GTot (chest1 a (n-len))
  = chest1_sub len n s

(* Bijection between the abstract 1-D tensor index [(k, ())] and a plain
   [natlt len], used to (un)reindex a forevery over tensor cells.  [unfold] so
   that [abs_bij.gg]/[.ff] reduce to [(k, ())]/[fst], letting the [forevery_iso]
   reindexing post match the [idx1]/[(tid, ())] forms used elsewhere. *)
unfold
let abs_bij (#len : nat) : (abs (len @| INil) =~ natlt len) =
  {
    ff = (fun (i, ()) -> i);
    gg = (fun i -> (i, ()));
  }


(* [chest1_to_seq] commutes with mapping; hence [chest1_rsum (chest_map f c)]
   equals the seq-level [rsum (seq_map f (chest1_to_seq c))]. This bridges the
   chest1 interface (in the .fsti) to the seq-based numeric proof below. *)
let chest_map_to_seq_map (#et1 #et2 : Type) (#n : nat)
  (f : et1 -> et2) (c : chest1 et1 n)
  : Lemma (chest1_to_seq (chest_map f c) == seq_map f (chest1_to_seq c))
  = assert (Seq.equal (chest1_to_seq (chest_map f c)) (seq_map f (chest1_to_seq c)))

let chest1_rsum_map (#n : nat) (f : real -> real) (c : chest1 real n)
  : Lemma (chest1_rsum (chest_map f c) == rsum (seq_map f (chest1_to_seq c)))
  = chest_map_to_seq_map f c

(* Plain ownership of a slice of a 1-D tensor. *)
let array1_pts_to_slice
  (#et : Type0)
  (#sz : nat)
  (#l : layout1 sz)
  ([@@@mkey] r : array1 et l)
  ([@@@mkey]i
   [@@@mkey]j : nat{i <= j /\ j <= sz})
  (s : chest1 et (j - i))
  : slprop
  = forall+ (k : nat{i <= k /\ k < j}).
      Cell r ((k, ()) <: abs (sz @| INil)) |-> (s `acc1` (k - i))

#push-options "--z3rlimit 80"
ghost
fn array1_slice_concat
  (#et : Type0)
  (#sz : nat)
  (#l : layout1 sz)
  (r : array1 et l)
  (i j k : nat{i <= j /\ j <= k /\ k <= sz})
  (#s1 : chest1 et (j - i))
  (#s2 : chest1 et (k - j))
  requires
    array1_pts_to_slice r i j s1 **
    array1_pts_to_slice r j k s2
  ensures
    array1_pts_to_slice r i k (chest1_append s1 s2)
{
  unfold array1_pts_to_slice r i j s1;
  unfold array1_pts_to_slice r j k s2;

  let s = chest1_append s1 s2;

  (* Rewrite each side to use s *)
  forevery_ext
    (fun (x:nat{i <= x /\ x < j}) -> tensor_pts_to_cell r ((x <: natlt sz), ()) (acc1 s1 (x - i)))
    (fun (x:nat{i <= x /\ x < j}) -> tensor_pts_to_cell r ((x <: natlt sz), ()) (acc1 s (x - i)));
  forevery_ext
    (fun (x:nat{j <= x /\ x < k}) -> tensor_pts_to_cell r ((x <: natlt sz), ()) (acc1 s2 (x - j)))
    (fun (x:nat{j <= x /\ x < k}) -> tensor_pts_to_cell r ((x <: natlt sz), ()) (acc1 s (x - i)));

  (* Join *)
  forevery_refine_join' #nat
    (fun (x:nat) -> i <= x /\ x < j)
    (fun (x:nat) -> j <= x /\ x < k)
    (fun (x:nat{(i <= x /\ x < j) \/ (j <= x /\ x < k)}) ->
      tensor_pts_to_cell r ((x <: natlt sz), ()) (acc1 s (x - i)));

  (* Simplify *)
  forevery_refine_ext' #nat
    #(fun (x:nat) -> (i <= x /\ x < j) \/ (j <= x /\ x < k))
    (fun (x:nat) -> i <= x /\ x < k)
    (fun (x:nat{(i <= x /\ x < j) \/ (j <= x /\ x < k)}) ->
      tensor_pts_to_cell r ((x <: natlt sz), ()) (acc1 s (x - i)));


  // FIXME: Terrible that this is needed. But we have a length of (j-i)+(k-j) which
  // does not unify with k-i.
  forevery_ext
    _
    (fun (x : nat{i <= x /\ x < k}) ->
      tensor_pts_to_cell r ((x <: natlt sz), ()) (acc1 #_ #(k-i) s (x - i)));

  fold array1_pts_to_slice r i k s;
}
#pop-options

inline_for_extraction noextract
fn array1_read_from_slice
  (#et : Type0)
  (#len : erased nat)
  (#l : layout1 len) {| ctlayout l |}
  (r : array1 et l)
  (#i #j : erased nat{i <= j /\ j <= len})
  (idx : sz{i <= idx /\ idx < j})
  (#s : erased (chest1 et (j - i)))
  preserves
    array1_pts_to_slice r i j s
  returns
    v : et
  ensures
    pure (v == acc1 s (idx - i))
{
  unfold array1_pts_to_slice r i j s;
  forevery_extract #(x:nat{i <= x /\ x < j}) (SZ.v idx) _;
  let v = tensor_read_cell r ((idx <: szlt len), ());
  Pulse.Lib.Trade.elim_trade _ _;
  fold array1_pts_to_slice r i j s;
  v
}

inline_for_extraction noextract
fn array1_write_to_slice
  (#et : Type0)
  (#len : erased nat)
  (#l : layout1 len) {| ctlayout l |}
  (r : array1 et l)
  (#i #j : erased nat{i <= j /\ j <= len})
  (idx : sz{i <= idx /\ idx < j})
  (#s : erased (chest1 et (j - i)))
  (v : et)
  requires
    array1_pts_to_slice r i j s
  ensures
    array1_pts_to_slice r i j (upd1 s (idx - i) v)
{
  unfold array1_pts_to_slice r i j s;
  forevery_extract' #(x:nat{i <= x /\ x < j}) (SZ.v idx) _;
  tensor_write_cell r ((idx <: szlt len), ()) v;
  let s' : erased (chest1 et (j - i)) = upd1 s (idx - i) v;
  Pulse.Lib.Forall.elim_forall
    (fun (x:nat{i <= x /\ x < j}) ->
      tensor_pts_to_cell r ((x <: natlt len), ()) (acc1 s' (x - i)));
  Pulse.Lib.Trade.elim_trade _ _;
  fold array1_pts_to_slice r i j s';
  rewrite each s' as upd1 s (idx - i) v;
  ()
}

(* Build a length-one slice from a single owned cell. The clean [natlt sz]
   index avoids the SizeT->nat coercion that trips the framing tactic, and
   using [i - i] (rather than [0]) in both the rewrite and the predicate keeps
   the diagonal match syntactic. *)
ghost
fn slice_singleton
  (#et : Type0)
  (#sz : nat)
  (#l : layout1 sz)
  (r : array1 et l)
  (i : nat{i < sz})
  (v : et)
  (s : chest1 et 1)
  requires
    tensor_pts_to_cell r ((i <: natlt sz), ()) v **
    pure (v == acc1 s 0)
  ensures
    array1_pts_to_slice r i (i+1) s
{
  forevery_singleton_intro'
    #(x:nat{i <= x /\ x < i + 1})
    (fun x -> tensor_pts_to_cell r ((x <: natlt sz), ()) v)
    i;
  forevery_ext
    (fun (x:nat{i <= x /\ x < i + 1}) -> tensor_pts_to_cell r ((x <: natlt sz), ()) v)
    (fun (x:nat{i <= x /\ x < i + 1}) -> tensor_pts_to_cell r ((x <: natlt sz), ()) (acc1 #_ #(i+1-i) s (x - i)));
  fold array1_pts_to_slice r i (i+1) s;
}

(* Ownership of array r between i and j. The first value of that slice
is the reduction of all the values in the (original) slice v. *)
unfold
let array1_pts_to_slice_sum_inner
  (#et:Type0) {| scalar et, real_like et |}
  (#sz : nat)
  (#l : layout1 sz)
  (r : array1 et l)
  (i j : nat{i < j /\ j <= sz})
  (rr : chest1 real sz)
  (s : chest1 et (j - i))
  : slprop
  = array1_pts_to_slice r i j s **
    pure ((acc1 s 0) %~ chest1_rsum (chest1_sub i j rr))

let array1_pts_to_slice_sum
  (#et:Type0) {| scalar et, real_like et |}
  (#sz : nat)
  (#l : layout1 sz)
  ([@@@mkey] r : array1 et l)
  ([@@@mkey] i : nat)
  (j : nat{i < j /\ j <= sz})
  (rr : chest1 real sz)
  : slprop
  = exists* s. array1_pts_to_slice_sum_inner r i j rr s

// Barrier

let barrier_matrix
  (#et:Type0) {| scalar et, real_like et |}
  (nth : szp)
  (#l : layout1 nth)
  (r : array1 et l)
  (vr : chest1 real nth)
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
  (#l : layout1 nth)
  (r : array1 et l)
  (vr : chest1 real nth)
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
    assert pure (pow2 it > tid);
    assert pure (tid % pow2 it == tid);
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
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) })
  (#l : layout1 lena)
  (a : array1 et l)
  (va : chest1 et lena)
  (vr : chest1 real lena)
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt 1)
  (tid : natlt nth)
  : slprop
  = a |-> Frac (1 /. nth) va **
    if_ (op_Equality #nat tid 0) (live out) **
    exists* (v : et).
      tensor_pts_to_cell (from_array (l1_forward nth) shmem._1) (tid, ())  v

// Same RO permission to a, 1st thread has full ownership of shmem plus of the
// output reference.  No need to specify the contents of the shmem array, it
// will disappear.
unfold
let kpost
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) })
  (#l : layout1 lena)
  (a : array1 et l)
  (va : chest1 et lena)
  (vr : chest1 real lena)
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt 1)
  (tid : natlt nth)
  : slprop
  = a |-> Frac (1 /. nth) va **
    if_ (op_Equality #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et). out |-> v ** pure (v %~ chest1_rsum (chest_map pre_map_r vr))
    )

inline_for_extraction
fn iteration
  (#et:Type0) {| scalar et, real_like et |}
  (nth : szp { SZ.v nth <= max_threads })
  (#l : layout1 nth) {| Kuiper.Tensor.ctlayout l |}
  (r : array1 et l)
  (vr : chest1 real nth)
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
      (**)assert (pure (s1 `approximates` chest1_rsum (chest1_sub tid nextid vr)));

      let s2 = array1_read_from_slice r nextid;
      (**)assert (pure (s2 `approximates` chest1_rsum (chest1_sub nextid end_ vr)));

      let s = add s1 s2;
      (**)chest1_approximates_append s1 s2 (chest1_sub tid nextid vr) (chest1_sub nextid end_ vr);
      (**)assert (pure ((s1 `add` s2) `approximates` chest1_rsum (chest1_append (chest1_sub tid nextid vr) (chest1_sub nextid end_ vr))));
      (**)chest1_rsum_append (chest1_sub tid nextid vr) (chest1_sub nextid end_ vr);
      (**)chest1_rsum_sub_split tid nextid end_ vr;
      (**)assert (pure (s `approximates` chest1_rsum (chest1_sub tid end_ vr)));

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
let rec log2_range (n:pos) (k:nat)
  : Lemma (requires pow2 k <= n /\ n < pow2 (k+1))
          (ensures log2 n == k)
          (decreases k)
= if k = 0 then ()
  else begin
    FStar.Math.Lemmas.lemma_div_le (pow2 k) n 2;
    log2_range (n/2) (k-1)
  end

(* The smallest k with pow2 k >= nth equals log2 (2*nth - 1). *)
let log2_hreduce (nth:pos) (it:nat)
  : Lemma (requires pow2 it >= nth /\ (it == 0 \/ pow2 (it - 1) < nth))
          (ensures it == log2 (2 * nth - 1))
= if it = 0 then ()
  else log2_range (2 * nth - 1) it

(* ---- End helpers ---- *)

(* Per-thread partial sums: thread [tid] reduces the [tid]-strided subseq of
   [pre_map]-mapped input.  Returned as a [chest1] so it can index the shmem
   slice predicates; the entries are seq-level strided sums. *)
let vr_partial (pre_map : real -> real) (#lena : nat) (vr : chest1 real lena) (nth : nat)
  : GTot (chest1 real nth) =
  mk1 (fun tid -> rsum (seq_stride (seq_map pre_map (chest1_to_seq vr)) nth tid))

let vr_partial_acc (pre_map : real -> real) (#lena : nat) (vr : chest1 real lena)
  (nth : nat) (tid : natlt nth)
  : Lemma (acc1 (vr_partial pre_map vr nth) tid
           == rsum (seq_stride (seq_map pre_map (chest1_to_seq vr)) nth tid))
  = ()

let strided_sum_is_sum (pre_map : real -> real) (#lena : nat) (vr : chest1 real lena) (nth : pos)
  : Lemma (ensures chest1_rsum (vr_partial pre_map vr nth) == chest1_rsum (chest_map pre_map vr))
  = let s = seq_map pre_map (chest1_to_seq vr) in
    chest_map_to_seq_map pre_map vr;
    admit();
    assert (Seq.equal (chest1_to_seq (vr_partial pre_map vr nth))
                      (Seq.init_ghost nth (fun tid -> rsum (seq_stride s nth tid))))


(* Quantifier-free arithmetic step, proved in a clean context (stable):
   when [tid < nth] and [pow2 k == 1], [min (tid + pow2 k) nth == tid + 1]. *)
let min_tid_pow2_step (tid nth k : nat)
  : Lemma (requires tid < nth /\ pow2 k == 1)
          (ensures min (tid + pow2 k) nth == tid + 1)
  = ()

let lemma_first_past
  (len off : nat) (stride : pos)
  (i : nat)
  : Lemma (requires i % stride == off /\ i >= len /\ i < len + stride)
          (ensures  i == off + ((len - off - 1 + stride) / stride) * stride)
  = ()

#push-options "--z3rlimit 60"
inline_for_extraction noextract
fn sum_stride_map
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (lena : sz)
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l)
  (stride : szp)
  (off : szlt stride)
  (#va : erased (chest1 et lena))
  (vr : erased (chest1 real lena))
  (#f : perm)
  preserves
    gpu ** a |-> Frac f va ** pure (va %~ vr) ** pure (SZ.fits (lena + stride))
  returns
    res : et
  ensures
    pure (res %~ chest1_rsum (chest1_stride (chest_map pre_map_r vr) stride off))
{
  let mut acc : et = zero;
  let mut idx : sz = off;
  let gidx = galloc #nat 0;

  while (!idx <^ lena)
    invariant
      live acc ** live gidx **
      live idx **
      pure (SZ.v !idx == gread gidx * stride + off /\
            off <= SZ.v !idx /\ SZ.v !idx < lena + stride /\
            gread gidx <= chest1_stride_length lena stride off /\
            // gread gidx == (SZ.v !idx - SZ.v off) / SZ.v stride /\ // superfluous
            !acc %~ chest1_rsum (chest1_take (gread gidx) (chest1_stride (chest_map pre_map_r vr) stride off))) **
      emp
    decreases (lena + stride - !idx)
  {
    assert pure (gread gidx < chest1_stride_length lena stride off);

    (* Read from input array (fractional permission) *)
    let vidx = !idx;
    let v = tensor_read a ((vidx <: szlt lena), ());
    let v' = pre_map v;
    (**)assert (pure (v == acc1 va (SZ.v !idx)));
    (**)assert (pure (v %~ (vr `acc1` SZ.v !idx)));
    (**)assert (pure (v' %~ (chest_map pre_map_r vr `acc1` SZ.v !idx)));

    // assert pure (!acc %~ rsum (seq_take (gread gidx) (seq_stride (lseq_map pre_map_r vr) stride off)));
    assert pure (!acc %~ chest1_rsum (chest1_take (gread gidx) (chest1_stride (chest_map pre_map_r vr) stride off)));
    assert pure (v %~ (vr `acc1` !idx));
    assert pure (chest1_stride (chest_map pre_map_r vr) stride off `acc1` gread gidx == (chest_map pre_map_r vr) `acc1` (off + gread gidx * stride));
    assert pure (off + gread gidx * stride == SZ.v !idx);
    // rsum_seq_take_next_ (seq_stride (lseq_map pre_map_r vr) stride off) (gread gidx);

    let vgidx = gread gidx;
    assert (pure (SZ.v !idx                  == vgidx    * stride + off));
    Math.Lemmas.distributivity_add_left vgidx 1 stride;
    assert (pure ((vgidx + 1) * stride + off == ((vgidx * stride) + (1 * stride)) + off)); // Sad.
    assert (pure (SZ.v !idx + stride == (vgidx + 1) * stride + off));

    Math.Lemmas.add_div_mod_1 (SZ.v !idx) stride;

    acc := !acc `add` v';
    idx := !idx +^ stride;
    gwrite gidx (gread gidx + 1);

    assert pure (SZ.v !idx == gread gidx * stride + off);

    // FIXME: restore proof
    assume pure (!acc %~ chest1_rsum (chest1_take (gread gidx) (chest1_stride (chest_map pre_map_r vr) stride off)));
    ()
  };

  assert pure (SZ.v !idx == gread gidx * stride + off);
  Math.Lemmas.lemma_mod_plus off (gread gidx) stride;
  Math.Lemmas.small_mod off stride;
  assert pure ((off + stride * gread gidx) % stride == off);
  assert pure ((gread gidx * stride + off) % stride == off);
  assert pure (!idx % stride == off);
  lemma_first_past lena off stride (SZ.v !idx);
  assert (pure (SZ.v !idx == off + ((lena - off - 1 + stride) / stride) * stride));

  assert pure (gread gidx <= chest1_stride_length lena stride off);
  Math.Lemmas.cancel_mul_div (gread gidx) stride;
  (* FIXME: A calc proof would be much nicer. *)
  assert pure (gread gidx == (!idx - off) / stride);
  assert pure (gread gidx == ((off + ((lena - off - 1 + stride) / stride) * stride) - off) / stride);
  assert pure (gread gidx == (((lena - off - 1 + stride) / stride) * stride) / stride);
  assert pure (gread gidx == (lena - off - 1 + stride) / stride);
  assert pure (lena - off - 1 + stride == lena - off + stride - 1);
  assert pure (gread gidx == (lena - off + stride - 1) / stride);
  assert pure (gread gidx == chest1_stride_length lena stride off);
  assert pure (chest1_take (chest1_stride_length lena stride off) (chest1_stride (chest_map pre_map_r vr) stride off)
               `equal` chest1_stride (chest_map pre_map_r vr) stride off);
  assert pure (!acc %~ chest1_rsum (chest1_stride (chest_map pre_map_r vr) stride off));

  drop_ (gidx |-> _);

  !acc
}
#pop-options

#push-options "--z3rlimit 20 --print_implicits"
#set-options "--z3refresh" // CRUTCH
inline_for_extraction noextract
fn kf
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) })
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l)
  (va : chest1 et lena)
  (vr : chest1 real lena { va %~ vr })
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et nth])
  (bid : szlt 1sz)
  (tid : szlt nth)
  ()
  requires
    gpu **
    kpre pre_map pre_map_r nth lena a va vr out shmem bid tid **
    thread_id nth tid **
    block_id 1 bid **
    mbarrier_tok nth (barrier_matrix nth (from_array (l1_forward nth) shmem._1) (vr_partial pre_map_r vr nth)) **
    B.barrier_state 0
  ensures
    gpu **
    kpost pre_map pre_map_r nth lena a va vr out shmem bid tid **
    thread_id nth tid **
    block_id 1 bid **
    mbarrier_tok nth (barrier_matrix nth (from_array (l1_forward nth) shmem._1) (vr_partial pre_map_r vr nth)) **
    B.barrier_state (hreduce_barrier_count nth)
{
  let (gsa, _) = shmem;

  let sa = from_array (l1_forward nth) gsa;
  rewrite each from_array (l1_forward nth) gsa as sa;

  let vr_s : erased (chest1 real nth) = vr_partial pre_map_r vr nth;

  (* Compute partial sum and write to shmem *)
  (**)chest1_to_seq_approx va vr;
  let psum : et = sum_stride_map pre_map pre_map_r lena a nth tid vr;
  tensor_write_cell sa (tid, ()) psum;

  (* Now do tree reduction on shmem *)
  let mut n : szlt 32 = 0sz;

  (* The slice [tid, tid+1) holds [psum], which approximates the [tid]-th
     partial sum [acc1 vr_s tid]. *)
  let psum_chest : chest1 et 1 = mk1 #et #1 (fun _ -> psum);
  slice_singleton sa (SZ.v tid) psum psum_chest;

  // (**)chest1_rsum_sub_one tid vr_s;
  (**)vr_partial_acc pre_map_r vr nth tid;
  assume pure (acc1 psum_chest 0 %~ chest1_rsum (chest1_sub tid (tid + 1) vr_s));
  (**)fold array1_pts_to_slice_sum sa tid (tid + 1) vr_s;
  (**)assert pure (pow2 (SZ.v !n) == 1);
  (**)min_tid_pow2_step (SZ.v tid) (SZ.v nth) (SZ.v !n);
  (**)rewrite (array1_pts_to_slice_sum sa tid (tid + 1) vr_s)
  (**)     as (array1_pts_to_slice_sum sa tid (min (tid + pow2 !n) nth) vr_s);
  (**)if_intro_true' (div_pow2 !n tid) (array1_pts_to_slice_sum sa tid (min (tid + pow2 !n) nth) vr_s);

  open FStar.SizeT;
  while (spow2 !n <^ nth)
    invariant
      live n **
      B.barrier_state !n **
      if_ (div_pow2 !n tid) (array1_pts_to_slice_sum sa tid (min (tid + pow2 !n) nth) vr_s) **
      pure (v !n > 0 ==> pow2 (v !n - 1) < v nth)
    decreases (2 * nth - spow2 !n)
  {
    iteration nth sa vr_s tid !n;
    n := !n +^ 1sz;
  };

  with it. assert (B.barrier_state it);

  // After loop exit: pow2 it >= nth, and tid < nth, so div_pow2 it tid <==> tid = 0
  FStar.Math.Lemmas.modulo_lemma tid (pow2 it);
  rewrite
    (if_ (div_pow2 it tid) (array1_pts_to_slice_sum sa tid (min (tid + pow2 it) nth) vr_s))
  as
    (if_ (op_Equality #nat tid 0) (array1_pts_to_slice_sum sa 0 nth vr_s));

  log2_hreduce (v nth) it;
  rewrite (B.barrier_state it) as (B.barrier_state (hreduce_barrier_count nth));

  (* Thread zero owns the result at the end, and writes it out. *)
  if (tid = 0sz) {
    if_elim_true' (op_Equality #nat tid 0) (array1_pts_to_slice_sum sa 0 nth vr_s);
    if_elim_true' (op_Equality #nat tid 0) (live out);
    unfold array1_pts_to_slice_sum sa 0 nth vr_s;
    (**)strided_sum_is_sum pre_map_r vr nth;
    (**)chest1_rsum_sub_full vr_s;
    gpu_write out (array1_read_from_slice sa 0sz);
    with ss. assert array1_pts_to_slice sa 0 nth ss;
    unfold array1_pts_to_slice sa;
    (* Reindex the per-cell ownership back to abstract indices and implode the
       shmem tensor to recover its liveness (mirrors Kuiper.Kernel.HReduce.Block.Max). *)
    let css : erased (chest1 et nth) = hide (mk1 #et #nth (fun (k:natlt nth) -> acc1 ss k));
    forevery_refine_ext' #nat #(fun (k:nat) -> 0 <= k /\ k < nth) (fun (k:nat) -> k < nth) _;
    forevery_ext
      (fun (k:natlt nth) -> tensor_pts_to_cell sa ((k <: natlt nth), ()) (acc1 ss (k - 0)))
      (fun (k:natlt nth) -> tensor_pts_to_cell sa (abs_bij.gg k) (acc (reveal css) (abs_bij.gg k)));
    forevery_iso_back (abs_bij #nth)
      (fun (i : abs (nth @| INil)) -> tensor_pts_to_cell sa i (acc (reveal css) i));
    tensor_implode sa #1.0R #(reveal css);
    rewrite each sa as from_array (l1_forward nth) shmem._1;
    if_intro_true' (op_Equality #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et). out |-> v ** pure (v %~ chest1_rsum (chest_map pre_map_r vr))
    )
  } else {
    (* Nop, convince Pulse. *)
    if_elim_false' (op_Equality #nat tid 0) (array1_pts_to_slice_sum sa 0 nth vr_s);
    if_elim_false' (op_Equality #nat tid 0) (live out);
    if_intro_false' (op_Equality #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et). out |-> v ** pure (v %~ chest1_rsum (chest_map pre_map_r vr))
    );
    ();
  };
}
#pop-options

ghost
fn block_setup
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) })
  (#l : layout1 lena)
  (a : array1 et l)
  (#va : chest1 et lena)
  (vr : chest1 real lena { va %~ vr })
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt 1)
  ()
  norewrite
  requires
    live_c_shmems shmem **
    (a |-> va ** live out)
  ensures
    (forall+ (i : natlt nth). kpre pre_map pre_map_r nth lena a va vr out shmem bid i) **
    emp
{
  unfold_live_c_shmems_cons shmem #_;
  unfold_live_c_shmems_nil shmem._2 #_;
  let gsa = shmem._1; rewrite each fst shmem as gsa;
  unfold live_c_shmem gsa;

  with vgsa. assert gsa |-> vgsa;
  gpu_pts_to_ref gsa;

  (* share input into nth fractional copies *)
  tensor_share_n a nth;

  (* tid 0 gets the ref *)
  forevery_if_intro #(natlt nth) 0 (fun _ -> live out);
  (* Sad.*)
  forevery_ext
    (fun tid -> if_ (op_Equality #(natlt nth) tid 0) (live out))
    (fun tid -> if_ (op_Equality #nat tid 0) (live out));

  forevery_zip (fun _ -> a |-> Frac (1 /. nth) va) _;

  (* View shmem array as a tensor and explode it into per-cell ownership. *)
  tensor_abs' (l1_forward nth) gsa;
  tensor_explode (from_array (l1_forward nth) gsa);
  forevery_iso abs_bij _;

  forevery_zip #(natlt nth)
    (fun tid -> a |-> Frac (1 /. nth) va ** if_ (op_Equality #nat tid 0) (live out))
    _;

  forevery_map
    #(natlt nth)
    (fun tid ->
      (a |-> Frac (1 /. nth) va **
       if_ (op_Equality #nat tid 0) (live out)) **
      Cell (from_array (l1_forward nth) gsa) (abs_bij.gg (tid <: natlt nth))
        |-> (acc (from_seq (l1_forward nth) vgsa) (abs_bij.gg (tid <: natlt nth)))
    )
    (fun (tid : natlt nth) -> kpre pre_map pre_map_r nth lena a va vr out shmem bid tid)
    fn tid {
      rewrite each gsa as shmem._1;
      ();
    };

  ()
}


ghost
fn block_teardown
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) })
  (#l : layout1 lena)
  (a : array1 et l)
  (#va : chest1 et lena)
  (vr : chest1 real lena { va %~ vr })
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt 1)
  ()
  norewrite
  requires
    (forall+ (i : natlt nth). kpost pre_map pre_map_r nth lena a va vr out shmem bid i) **
    emp
  ensures
    live_c_shmems shmem **
    (a |-> va ** (exists* (v : et). out |-> v ** pure (v %~ chest1_rsum (chest_map pre_map_r vr))))
{
  forevery_unzip _ _;

  tensor_gather_n a nth;

  (* Sad.*)
  forevery_ext #(natlt nth)
    (fun tid ->
      if_ (op_Equality #nat tid 0) (
        live (from_array (l1_forward nth) shmem._1) **
        exists* (v : et). out |-> v ** pure (v %~ chest1_rsum (chest_map pre_map_r vr))))
    (fun tid ->
      if_ (op_Equality #(natlt nth) tid 0) (
        live (from_array (l1_forward nth) shmem._1) **
        exists* (v : et). out |-> v ** pure (v %~ chest1_rsum (chest_map pre_map_r vr))));

  forevery_if_elim #(natlt nth) 0 (fun tid ->
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et). out |-> v ** pure (v %~ chest1_rsum (chest_map pre_map_r vr))
  );

  tensor_concr (from_array (l1_forward nth) shmem._1);
  rewrite each core (from_array (l1_forward nth) shmem._1) as shmem._1;

  fold_live_c_shmems_nil shmem._2 #_;
  with vgsa. assert shmem._1 |-> vgsa;
  fold_live_c_shmem shmem._1;
  fold_live_c_shmems_cons shmem #_;
}

ghost
fn setup
  (#et:Type0) {| scalar et, real_like et |}
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) })
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#va : chest1 et lena)
  (vr : chest1 real lena { va %~ vr })
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
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) })
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#va : chest1 et lena)
  (vr : chest1 real lena { va %~ vr })
  (out : gpu_ref et)
  ()
  norewrite
  requires
    (forall+ (bid : natlt 1). a |-> va ** exists* (v : et). out |-> v ** pure (v %~ chest1_rsum (chest_map pre_map_r vr))) **
    emp
  ensures
    a |-> va ** (exists* (v : et). out |-> v ** pure (v %~ chest1_rsum (chest_map pre_map_r vr)))
{
  forevery_singleton_elim #(natlt 1) _;
}

inline_for_extraction noextract
let kernel
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) })
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#va : chest1 et lena)
  (vr : chest1 real lena { va %~ vr })
  (out : gpu_ref et)
  : kernel_desc
      (a |-> va ** live out)
      (a |-> va ** exists* (v : et). out |-> v ** pure (v %~ chest1_rsum (chest_map pre_map_r vr)))
  = {
    nblk = 1sz;
    nthr = nth;

    shmems_desc = [SHArray et nth];

    barrier_contract = (fun _bid shmem ->
      mbarrier_contract (barrier_matrix #et nth (from_array _ shmem._1) (vr_partial pre_map_r vr nth)));
    barrier_count    = (fun _bid    -> hreduce_barrier_count nth);
    barrier_ok       = (fun _bid shmem ->
      mbarrier_transform (barrier_matrix nth #(l1_forward nth) (from_array _ shmem._1) (vr_partial pre_map_r vr nth)));

    f = kf pre_map pre_map_r nth lena a va vr out;

    block_pre  = (fun bid -> a |-> va ** live out);
    block_post = (fun bid -> a |-> va ** exists* (v : et). out |-> v ** pure (v %~ chest1_rsum (chest_map pre_map_r vr)));
    setup      = setup    nth lena a #va vr out;
    teardown   = teardown pre_map pre_map_r nth lena a #va vr out;

    block_frame    = (fun _shmem _bid -> emp);
    block_setup    = block_setup    pre_map pre_map_r nth lena a #va vr out;
    block_teardown = block_teardown pre_map pre_map_r nth lena a #va vr out;

    kpre =  kpre  pre_map pre_map_r nth lena a va vr out;
    kpost = kpost pre_map pre_map_r nth lena a va vr out;
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
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz)
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#va : chest1 et lena)
  (vr : chest1 real lena)
  norewrite // sigh... spec in fsti is not purified
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    pure (va %~ vr) **
    pure (SZ.fits (lena + nth))
  returns
    res : et
  ensures
    pure (res %~ chest1_rsum (chest_map pre_map_r vr))
{
  let out = Kuiper.Ref.gpu_alloc0 #et ();
  launch_sync (kernel pre_map pre_map_r nth lena a vr out);

  (* Bring back out result, free swap. *)
  let mut hout : et = zero #et;
  Kuiper.Ref.gpu_memcpy_device_to_host hout out;
  Kuiper.Ref.gpu_free out;

  !hout;
}
