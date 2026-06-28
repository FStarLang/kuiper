module Kuiper.Kernel.HReduce.Max

(* Parallel single-block max reduction.

   This module is a port of Kuiper.Kernel.HReduce from a (+)/rsum reduction to
   an fmax/seq_max reduction. The overall structure (strided per-thread fold +
   shared-memory tree reduction guarded by a row/column barrier) is identical.

   The key difference: addition has a unit (zero), so the sum reduction can
   initialize accumulators with zero and reduce over all nth strided buckets,
   including empty ones. Max has no real-number unit. So this module requires
   `0 < nth /\ nth <= lena`, which guarantees every strided bucket is non-empty,
   and the per-thread fold (`max_stride_map`) initializes its accumulator with
   its first strided element instead of a unit. *)

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Seq.Common
open Kuiper.Math.OnlineSoftmax { seq_max, seq_max_cons_lem }
open Kuiper.Tensor { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Pulse.Lib.GhostReference { read as gread, write as gwrite, alloc as galloc }

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

#push-options "--z3rlimit 40"
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
let array1_pts_to_slice_max_inner
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (#sz : nat)
  (#l : Array1.layout sz)
  (r : Array1.t et l)
  (i j : nat{i < j /\ j <= sz})
  (rr : lseq real sz)
  (s : lseq et (j - i))
  : slprop
  = array1_pts_to_slice r i j s **
    pure ((s @! 0) %~ seq_max (Seq.slice rr i j))

let array1_pts_to_slice_max
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (#sz : nat)
  (#l : Array1.layout sz)
  ([@@@mkey] r : Array1.t et l)
  ([@@@mkey] i : nat)
  (j : nat{i < j /\ j <= sz})
  (rr : lseq real sz)
  : slprop
  = exists* s. array1_pts_to_slice_max_inner r i j rr s

(* ---- Helpers for strided_max_is_max ---- *)

(* seq_max of a singleton. *)
let seq_max_singleton (x : real)
  : Lemma (seq_max (seq![x]) == x)
  = assert (seq_max (seq![x]) == x)

(* seq_max of a length-one sequence equals its element. *)
let seq_max_len1 (s : seq real { len s == 1 })
  : Lemma (seq_max s == (s @! 0))
  = lem_one_elem s (s @! 0);
    seq_max_singleton (s @! 0)

(* Every element is below the max. *)
private let rec seq_max_ub (s : seq real { len s > 0 }) (i : nat { i < len s })
  : Lemma (ensures (s @! i) <=. seq_max s)
          (decreases len s)
  = let n = len s in
    if n = 1 then seq_max_len1 s
    else begin
      let s' = Seq.slice s 0 (n - 1) in
      let last = s @! (n - 1) in
      assert (Seq.equal s (s' @+ seq![last]));
      Seq.lemma_eq_elim s (s' @+ seq![last]);
      seq_max_cons_lem s' last;   // seq_max s == rmax last (seq_max s')
      if i = n - 1 then
        assert (s @! i == last)
      else begin
        assert (Seq.index s' i == Seq.index s i);
        seq_max_ub s' i
      end
    end

(* The max is below any upper bound of all elements. *)
private let rec seq_max_le (s : seq real { len s > 0 }) (c : real)
  : Lemma (requires (forall (i : nat { i < len s }). (s @! i) <=. c))
          (ensures seq_max s <=. c)
          (decreases len s)
  = let n = len s in
    if n = 1 then seq_max_len1 s
    else begin
      let s' = Seq.slice s 0 (n - 1) in
      let last = s @! (n - 1) in
      assert (Seq.equal s (s' @+ seq![last]));
      Seq.lemma_eq_elim s (s' @+ seq![last]);
      seq_max_cons_lem s' last;   // seq_max s == rmax last (seq_max s')
      let aux (i : nat { i < len s' }) : Lemma ((s' @! i) <=. c)
        = assert (Seq.index s' i == Seq.index s i)
      in
      Classical.forall_intro aux;
      seq_max_le s' c
    end

(* seq_max distributes over append of two non-empty sequences. *)
private let rec seq_max_append (s1 s2 : seq real { len s1 > 0 /\ len s2 > 0 })
  : Lemma (ensures seq_max (s1 @+ s2) == rmax (seq_max s1) (seq_max s2))
          (decreases len s2)
  = let n2 = len s2 in
    if n2 = 1 then begin
      assert (Seq.equal s2 (seq![s2 @! 0]));
      Seq.lemma_eq_elim s2 (seq![s2 @! 0]);
      seq_max_singleton (s2 @! 0);          // seq_max s2 == s2 @! 0
      assert (Seq.equal (s1 @+ s2) (s1 @+ seq![s2 @! 0]));
      Seq.lemma_eq_elim (s1 @+ s2) (s1 @+ seq![s2 @! 0]);
      seq_max_cons_lem s1 (s2 @! 0);        // seq_max (s1 @+ s2) == rmax (s2@!0) (seq_max s1)
      lem_rmax_comm (s2 @! 0) (seq_max s1)
    end else begin
      let s2' = Seq.slice s2 0 (n2 - 1) in
      let last = s2 @! (n2 - 1) in
      assert (Seq.equal s2 (s2' @+ seq![last]));
      Seq.lemma_eq_elim s2 (s2' @+ seq![last]);
      seq_max_cons_lem s2' last;            // seq_max s2 == rmax last (seq_max s2')
      assert (Seq.equal (s1 @+ s2) ((s1 @+ s2') @+ seq![last]));
      Seq.lemma_eq_elim (s1 @+ s2) ((s1 @+ s2') @+ seq![last]);
      seq_max_cons_lem (s1 @+ s2') last;    // seq_max (s1@+s2) == rmax last (seq_max (s1@+s2'))
      seq_max_append s1 s2';                // seq_max (s1@+s2') == rmax (seq_max s1) (seq_max s2')
      let a = seq_max s1 in
      let b = seq_max s2' in
      // goal: rmax last (rmax a b) == rmax a (rmax last b)
      lem_rmax_assoc last a b;              // rmax last (rmax a b) == rmax (rmax last a) b
      lem_rmax_comm last a;                 // rmax last a == rmax a last
      lem_rmax_assoc a last b               // rmax (rmax a last) b == rmax a (rmax last b)
    end

(* One step of a left-to-right running max over a non-empty prefix:
   extending [seq_take k s] by one element [s @! k] maxes that element in. *)
private let seq_max_take_step (s : seq real) (k : nat { 1 <= k /\ k < len s })
  : Lemma (ensures seq_max (seq_take (k + 1) s) == rmax (seq_max (seq_take k s)) (s @! k))
  = let pre  = seq_take k s in
    let pre1 = seq_take (k + 1) s in
    assert (Seq.equal pre1 (pre @+ seq![s @! k]));
    Seq.lemma_eq_elim pre1 (pre @+ seq![s @! k]);
    seq_max_cons_lem pre (s @! k);             // seq_max pre1 == rmax (s@!k) (seq_max pre)
    lem_rmax_comm (s @! k) (seq_max pre)

(* ---- End helpers ---- *)

(* A strided bucket is non-empty as soon as its offset is in range.  Stated as
   an SMTPat so that the well-typedness of `seq_max (seq_stride ...)` (which
   needs the bucket non-empty) is discharged automatically. *)
private let stride_nonempty (#a:Type) (s : seq a) (stride : pos) (off : natlt stride)
  : Lemma (requires off < Seq.length s)
          (ensures seq_stride_length s stride off > 0 /\
                   Seq.length (seq_stride s stride off) == seq_stride_length s stride off /\
                   Seq.length (seq_stride s stride off) > 0)
          [SMTPat (seq_stride s stride off)]
  = let n = Seq.length s in
    // numerator = n - off + stride - 1 >= stride
    FStar.Math.Lemmas.lemma_div_le stride (n - off + stride - 1) stride;
    FStar.Math.Lemmas.cancel_mul_div 1 stride

(* Partial maxima of the nth strided buckets.  Requires nth <= len vr so that
   every bucket is non-empty (and seq_max is well-defined). *)
let vr_partial_max (pre_map : real -> real) (vr : seq real) (nth : pos { nth <= Seq.length vr })
  : GTot (lseq real nth) =
  let w = seq_map pre_map vr in
  Seq.init_ghost nth (fun tid ->
    stride_nonempty w nth tid;
    seq_max (seq_stride w nth tid))

(* The max over the per-bucket maxima equals the max over the whole sequence.
   Proven by antisymmetry, with each direction in its own lemma: the monolithic
   proof is brittle (it only went through via F*'s fuel escalation, which is
   slow), whereas the two halves are small and stable. *)

(* Direction 1: every bucket max is below the global max, so their max is too. *)
#push-options "--fuel 2 --ifuel 1 --z3rlimit 30"
let strided_max_le (pre_map : real -> real) (vr : seq real) (nth : pos { nth <= Seq.length vr })
  : Lemma (ensures seq_max (vr_partial_max pre_map vr nth) <=. seq_max (seq_map pre_map vr))
  = let w = seq_map pre_map vr in
    let lenw = Seq.length w in
    let vp = vr_partial_max pre_map vr nth in
    let aux1 (tid : nat { tid < nth }) : Lemma ((vp @! tid) <=. seq_max w)
      = stride_nonempty w nth tid;
        let bucket = seq_stride w nth tid in
        let aux1a (i : nat { i < Seq.length bucket }) : Lemma ((bucket @! i) <=. seq_max w)
          = // bucket @! i == w @! (tid + i*nth)
            assert (tid + i * nth < lenw);
            seq_max_ub w (tid + i * nth)
        in
        Classical.forall_intro aux1a;
        seq_max_le bucket (seq_max w)
    in
    Classical.forall_intro aux1;
    seq_max_le vp (seq_max w)
#pop-options

(* Direction 2: every element lies in some bucket, so it is below that bucket's
   max, hence below the max of the partial maxima. *)
#push-options "--fuel 2 --ifuel 1 --z3rlimit 30"
let strided_max_ge (pre_map : real -> real) (vr : seq real) (nth : pos { nth <= Seq.length vr })
  : Lemma (ensures seq_max (seq_map pre_map vr) <=. seq_max (vr_partial_max pre_map vr nth))
  = let w = seq_map pre_map vr in
    let lenw = Seq.length w in
    let vp = vr_partial_max pre_map vr nth in
    let aux2 (g : nat { g < lenw }) : Lemma ((w @! g) <=. seq_max vp)
      = let off : nat = g % nth in
        let i : nat = g / nth in
        FStar.Math.Lemmas.euclidean_division_definition g nth;  // g == nth*(g/nth) + g%nth
        assert (off < nth);
        assert (g == off + i * nth);
        stride_nonempty w nth off;
        // i < seq_stride_length w nth off
        assert (i * nth <= lenw - off - 1);
        FStar.Math.Lemmas.lemma_div_le (i * nth) (lenw - off - 1) nth;
        FStar.Math.Lemmas.cancel_mul_div i nth;
        assert (i <= (lenw - off - 1) / nth);
        FStar.Math.Lemmas.lemma_div_plus (lenw - off - 1) 1 nth;
        assert (seq_stride_length w nth off == (lenw - off - 1) / nth + 1);
        assert (i < Seq.length (seq_stride w nth off));
        // (seq_stride w nth off) @! i == w @! (off + i*nth) == w @! g  (init_ghost_index_)
        assert (seq_stride w nth off @! i == w @! g);
        seq_max_ub (seq_stride w nth off) i;  // w@!g == bucket@!i <=. seq_max bucket == vp@!off
        seq_max_ub vp off                     // vp@!off <=. seq_max vp
    in
    Classical.forall_intro aux2;
    seq_max_le w (seq_max vp)
#pop-options

let strided_max_is_max (pre_map : real -> real) (vr : seq real) (nth : pos { nth <= Seq.length vr })
  : Lemma (ensures seq_max (vr_partial_max pre_map vr nth) == seq_max (seq_map pre_map vr))
  = strided_max_le pre_map vr nth;
    strided_max_ge pre_map vr nth

// Barrier

let barrier_matrix
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
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
           (array1_pts_to_slice_max r from (min (from + pow2 it) nth) vr))

ghost
fn mk_barrier_pre
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp)
  (#l : Array1.layout nth)
  (r : Array1.t et l)
  (vr : lseq real nth)
  (tid : natlt nth)
  (it: natlt 31)
  requires
    if_ (not (div_pow2 (it + 1) tid) && div_pow2 it tid)
      (array1_pts_to_slice_max r tid (min (tid + pow2 it) nth) vr)
  ensures
    forall+ (i:natlt nth). barrier_matrix nth r vr it tid i
{
  open FStar.SizeT;
  if (tid >= pow2 it) {
    forevery_if_intro #(natlt nth) (tid - pow2 it) (fun i ->
      if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid))
        (array1_pts_to_slice_max r tid (min (tid + pow2 it) nth) vr));
    forevery_ext
      (fun (i:natlt nth) ->
        if_ (op_Equality #(natlt nth) i (tid - pow2 it))
          (if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid))
            (array1_pts_to_slice_max r tid (min (tid + pow2 it) nth) vr)))
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
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= lena })
  (#l : Array1.layout lena)
  (a : Array1.t et l)
  (va : lseq et lena)
  (vr : lseq real lena)
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt 1)
  (tid : natlt nth)
  : slprop
  = a |-> Frac (1 /. nth) va **
    if_ (op_Equality #nat tid 0) (live out) **
    exists* (v : et). Cell (Array1.from_array (l1_forward nth) shmem._1) tid |-> v

// Same RO permission to a, 1st thread has full ownership of shmem plus of the
// output reference.  No need to specify the contents of the shmem array, it
// will disappear.
unfold
let kpost
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= lena })
  (#l : Array1.layout lena)
  (a : Array1.t et l)
  (va : lseq et lena)
  (vr : lseq real lena)
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt 1)
  (tid : natlt nth)
  : slprop
  = a |-> Frac (1 /. nth) va **
    if_ (op_Equality #nat tid 0) (
      live (Array1.from_array (l1_forward nth) shmem._1) **
      exists* (v : et). out |-> v ** pure (v %~ seq_max (seq_map pre_map_r vr))
    )

inline_for_extraction
fn iteration
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
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
  requires if_ (div_pow2 it tid) (array1_pts_to_slice_max r tid (min (tid + pow2 it) nth) vr)
  ensures  B.barrier_state (it + 1)
  ensures  if_ (div_pow2 (it+1) tid) (array1_pts_to_slice_max r tid (min (tid + pow2 (it + 1)) nth) vr)
{
  case_split (div_pow2 (it + 1) tid)
    (if_ (div_pow2 it tid) (array1_pts_to_slice_max r tid (min (tid + pow2 it) nth) vr));
  if_flatten #(div_pow2 (it + 1) tid);
  if_flatten #(not (div_pow2 (it + 1) tid));

  div_pow2_lemma it (it + 1) tid;
  rewrite (if_ (div_pow2 (it + 1) tid && div_pow2 it tid)
            (array1_pts_to_slice_max r tid (min (tid + pow2 it) nth) vr))
      as (if_ (div_pow2 (it + 1) tid)
            (array1_pts_to_slice_max r tid (min (tid + pow2 it) nth) vr));

  mk_barrier_pre nth r vr tid it;
  fold RPM.row (barrier_matrix nth r vr) it tid;
  mbarrier_wait ();
  unfold RPM.col (barrier_matrix nth r vr) it tid;

  // combine (div_pow2 (it + 1) tid) (array1_pts_to_slice_max r tid (min (tid + pow2 it) nth) vv) _;

  let nextid = FStar.SizeT.(tid +^ spow2 it);

  (* We do not use end_ in extracted code, so we can use a nat and erase it
  so there are no traces in the extracted C. *)
  let end_ : erased nat = hide (min (tid + 2 * pow2 it) nth);

  if (nextid <^ nth) {
    forevery_ext
      (fun (from: natlt nth) ->
        if_ (op_Equality #int from (tid + pow2 it))
          (if_ (not (div_pow2 (it + 1) from) && div_pow2 it from)
            (array1_pts_to_slice_max r from (min (from + pow2 it) nth) vr)))
      (fun (from: natlt nth) ->
        if_ (op_Equality #(natlt nth) from (tid + pow2 it))
          (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
            (array1_pts_to_slice_max r from (min (from + pow2 it) nth) vr)));
    forevery_if_elim #(natlt nth)
      (tid + pow2 it)
      (fun (from: natlt nth) -> if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
         (array1_pts_to_slice_max r from (min (from + pow2 it) nth) vr));

    let b = sdiv_pow2 (it +^ 1sz) tid;

    rewrite each (div_pow2 (it + 1) (SZ.v tid)) as b;

    div_pow2_lemma_2 it tid;
    combine
      b
      (array1_pts_to_slice_max r nextid (min (tid + pow2 it + pow2 it) nth) vr)
      _;

    if b {
      assert (pure (div_pow2 (SZ.v it + 1) (SZ.v tid)));
      if_elim_true _;

      (**)unfold (array1_pts_to_slice_max r nextid end_ vr);
      (**)unfold (array1_pts_to_slice_max r tid nextid vr);
      (**)array1_slice_concat #et #nth r tid nextid end_;

      let s1 = array1_read_from_slice r tid;
      (**)assert (pure (s1 `approximates` seq_max (Seq.slice vr tid nextid)));

      let s2 = array1_read_from_slice r nextid;
      (**)assert (pure (s2 `approximates` seq_max (Seq.slice vr nextid end_)));

      (* Both slices are non-empty: tid < nextid < end_. *)
      (**)assert (pure (tid < nextid));
      (**)assert (pure (nextid < end_));

      let s = fmax s1 s2;
      (**)lem_append_slice vr tid nextid end_;
      (* fmax_approx_pat lifts s1, s2 approximations through fmax/rmax. *)
      (**)assert (pure ((s1 `fmax` s2) `approximates`
      (**)              rmax (seq_max (Seq.slice vr tid nextid)) (seq_max (Seq.slice vr nextid end_))));
      (**)seq_max_append (Seq.slice vr tid nextid) (Seq.slice vr nextid end_);
      (**)assert (pure (seq_max (Seq.append (Seq.slice vr tid nextid) (Seq.slice vr nextid end_))
      (**)              == rmax (seq_max (Seq.slice vr tid nextid)) (seq_max (Seq.slice vr nextid end_))));
      (**)assert (pure (s `approximates` seq_max (Seq.slice vr tid end_)));

      // gpu_array_write r tid s;
      array1_write_to_slice r tid s;

      (**)with seq. assert (array1_pts_to_slice r tid end_ seq);
      (**)fold (array1_pts_to_slice_max r tid end_ vr);
      (**)if_intro_true (array1_pts_to_slice_max r tid end_ vr);
      // Step below optional right now, but good practice?
      (**)rewrite
      (**)  if_ true
      (**)      (array1_pts_to_slice_max r (SZ.v tid) (reveal end_) vr)
      (**)as
      (**)  if_ (div_pow2 (SZ.v it + 1) (SZ.v tid))
      (**)      (array1_pts_to_slice_max r (SZ.v tid) (reveal end_) vr);
    } else {
      (* no-op *)
      if_elim_false _;
      if_intro_false (array1_pts_to_slice_max r tid end_ vr);
    }
  } else {
    forevery_map
      (fun (from: natlt nth) ->
        if_ (op_Equality #int from (tid + pow2 it))
          (if_ (not (div_pow2 (it + 1) from) && div_pow2 it from)
            (array1_pts_to_slice_max r from (min (from + pow2 it) nth) vr)))
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


let lemma_first_past
  (len off : nat) (stride : pos)
  (i : nat)
  : Lemma (requires i % stride == off /\ i >= len /\ i < len + stride)
          (ensures  i == off + ((len - off - 1 + stride) / stride) * stride)
  = ()

(* ── Pure arithmetic of the strided fold, factored out ─────────────────────
   The strided fold ([max_stride_map] here, [max_stride_map_2d] in
   [Kuiper.Kernel.HReduce.Block.Max]) carries a quantified approximation
   hypothesis in its ambient context. Proving the (nonlinear) index arithmetic
   of the loop *in that context* is pathologically slow because the solver keeps
   instantiating that irrelevant quantifier. We discharge the arithmetic here,
   in a clean context, and feed the results back as ground facts. (Block.Max
   reuses both lemmas through its [friend] of this module.) *)

(* One loop step: from [idx == gidx*stride + off] (and [idx] still in range),
   the next index is [(gidx+1)*stride + off], and the bucket length bound is
   maintained. *)
#push-options "--z3rlimit 20"
let max_stride_step_arith (#a:Type) (s : seq a) (stride : pos) (off : natlt stride)
  (gidx idx : nat)
  : Lemma (requires idx == gidx * stride + off /\ idx < Seq.length s)
          (ensures  off + gidx * stride == idx /\
                    idx + stride == (gidx + 1) * stride + off /\
                    gidx + 1 <= seq_stride_length s stride off)
  = Math.Lemmas.distributivity_add_left gidx 1 stride;
    Math.Lemmas.cancel_mul_div (gidx + 1) stride;
    Math.Lemmas.lemma_div_le ((gidx + 1) * stride) (Seq.length s - off + stride - 1) stride
#pop-options

(* Loop exit: once [idx] has just passed [Seq.length s] in [stride]-sized steps,
   the step count [gidx] equals the number of strided buckets. *)
#push-options "--z3rlimit 20"
let max_stride_post_arith (#a:Type) (s : seq a) (stride : pos) (off : natlt stride)
  (gidx idx : nat)
  : Lemma (requires idx == gidx * stride + off /\
                    idx >= Seq.length s /\ idx < Seq.length s + stride)
          (ensures  gidx == seq_stride_length s stride off)
  = let len = Seq.length s in
    Math.Lemmas.lemma_mod_plus off gidx stride;
    Math.Lemmas.small_mod off stride;
    lemma_first_past len off stride idx;
    Math.Lemmas.cancel_mul_div ((len - off - 1 + stride) / stride) stride;
    Math.Lemmas.cancel_mul_div gidx stride
#pop-options

#push-options "--fuel 2 --ifuel 1 --z3rlimit 20"
inline_for_extraction noextract
fn max_stride_map
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (lena : sz)
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l)
  (stride : szp)
  (off : szlt stride { SZ.v off < lena })
  (#va : erased (lseq et lena))
  (vr : erased (lseq real lena))
  (#f : perm)
  preserves
    gpu ** a |-> Frac f va ** pure (va %~ vr) ** pure (SZ.fits (lena + stride))
  returns
    res : et
  ensures
    pure (res %~ seq_max (seq_stride (lseq_map pre_map_r vr) stride off))
{
  (* off < lena, so the strided bucket at `off` is non-empty. *)
  stride_nonempty (lseq_map pre_map_r vr) stride off;

  (* Initialize the accumulator with the first strided element (index off),
     so the running max is over a non-empty prefix.  seq_take 1 of the bucket
     is the singleton [w @! off]. *)
  let v0 = Array1.read a off;
  let acc0 = pre_map v0;
  (**)assert (pure (v0 == va @! SZ.v off));
  (**)assert (pure (acc0 %~ (lseq_map pre_map_r vr @! SZ.v off)));
  (**)assert (pure (seq_stride (lseq_map pre_map_r vr) stride off @! 0 == (lseq_map pre_map_r vr) @! (off + 0 * stride)));
  (**)assert (pure (acc0 %~ (seq_stride (lseq_map pre_map_r vr) stride off @! 0)));
  (**)seq_max_singleton (seq_stride (lseq_map pre_map_r vr) stride off @! 0);
  (**)assert (pure (Seq.equal (seq_take 1 (seq_stride (lseq_map pre_map_r vr) stride off))
  (**)                        (seq![seq_stride (lseq_map pre_map_r vr) stride off @! 0])));
  (**)assert (pure (acc0 %~ seq_max (seq_take 1 (seq_stride (lseq_map pre_map_r vr) stride off))));

  let mut acc : et = acc0;
  let mut idx : sz = off +^ stride;
  let gidx = galloc #nat 1;

  while (!idx <^ lena)
    invariant
      live acc ** live gidx **
      live idx **
      pure (SZ.v !idx == gread gidx * stride + off /\
            gread gidx >= 1 /\
            off <= SZ.v !idx /\ SZ.v !idx < lena + stride /\
            gread gidx <= seq_stride_length (lseq_map pre_map_r vr) stride off /\
            !acc %~ seq_max (seq_take (gread gidx) (seq_stride (lseq_map pre_map_r vr) stride off))) **
      emp
    decreases (lena + stride - !idx)
  {
    assert pure (gread gidx < seq_stride_length vr stride off);

    (* Read from input array (fractional permission) *)
    let v = Array1.read a !idx;
    let v' = pre_map v;
    (**)assert (pure (v == va @! SZ.v !idx));
    (**)assert (pure (v %~ (vr @! SZ.v !idx)));
    (**)assert (pure (v' %~ (lseq_map pre_map_r vr @! SZ.v !idx)));

    assert pure (!acc %~ seq_max (seq_take (gread gidx) (seq_stride (lseq_map pre_map_r vr) stride off)));
    assert pure (v %~ (vr @! !idx));

    let vgidx = gread gidx;
    (* All loop-step index arithmetic, discharged in a clean context so the
       ambient approximation quantifier doesn't blow up the solver. *)
    max_stride_step_arith (lseq_map pre_map_r vr) stride off vgidx (SZ.v !idx);

    assert pure (seq_stride (lseq_map pre_map_r vr) stride off @! vgidx == (lseq_map pre_map_r vr) @! (off + vgidx * stride));

    (* seq_take (k+1) maxes in the bucket's k-th element; combine with fmax. *)
    (**)seq_max_take_step (seq_stride (lseq_map pre_map_r vr) stride off) vgidx;

    acc := !acc `fmax` v';
    idx := !idx +^ stride;
    gwrite gidx (gread gidx + 1);

    assert pure (SZ.v !idx == gread gidx * stride + off);
    ()
  };

  assert pure (SZ.v !idx == gread gidx * stride + off);
  (* Loop-exit index arithmetic, again discharged off to the side. *)
  max_stride_post_arith (lseq_map pre_map_r vr) stride off (gread gidx) (SZ.v !idx);
  assert pure (gread gidx == seq_stride_length (lseq_map pre_map_r vr) stride off);
  assert pure (seq_take (seq_stride_length (lseq_map pre_map_r vr) stride off) (seq_stride (lseq_map pre_map_r vr) stride off) == seq_stride (lseq_map pre_map_r vr) stride off);
  assert pure (!acc %~ seq_max (seq_stride (lseq_map pre_map_r vr) stride off));

  drop_ (gidx |-> _);

  !acc
}
#pop-options

#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn kf
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= lena })
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l)
  (va : erased (lseq et lena))
  (vr : erased (lseq real lena) { va %~ vr })
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
    mbarrier_tok nth (barrier_matrix nth (Array1.from_array (l1_forward nth) shmem._1) (vr_partial_max pre_map_r vr nth)) **
    B.barrier_state 0
  ensures
    gpu **
    kpost pre_map pre_map_r nth lena a va vr out shmem bid tid **
    thread_id nth tid **
    block_id 1 bid **
    mbarrier_tok nth (barrier_matrix nth (Array1.from_array (l1_forward nth) shmem._1) (vr_partial_max pre_map_r vr nth)) **
    B.barrier_state (hreduce_barrier_count nth)
{
  let (gsa, _) = shmem;

  let sa = Array1.from_array (l1_forward nth) gsa;
  rewrite each Array1.from_array (l1_forward nth) gsa as sa;

  let vr_s : erased (lseq real nth) = vr_partial_max pre_map_r vr nth;

  (* Compute partial max and write to shmem *)
  let psum : et = max_stride_map pre_map pre_map_r lena a nth tid vr;
  Array1.write_cell sa tid psum;

  (* Now do tree reduction on shmem *)
  let mut n : szlt 32 = 0sz;

  forevery_singleton_intro'
    #(x:nat{tid <= x /\ x < tid + 1})
    (fun x -> Cell sa (x <: natlt nth) |-> (seq![psum] @! (x - tid)))
    tid;
  fold array1_pts_to_slice sa tid (tid+1) seq![psum];

  (* psum %~ vr_s @! tid == seq_max (single-element slice [tid, tid+1)) *)
  (**)stride_nonempty (lseq_map pre_map_r vr) nth tid;
  (**)assert (pure (psum %~ (reveal vr_s @! SZ.v tid)));
  (**)seq_max_singleton (reveal vr_s @! SZ.v tid);
  (**)assert (pure (Seq.equal (Seq.slice (reveal vr_s) tid (tid + 1)) (seq![reveal vr_s @! SZ.v tid])));
  (**)assert (pure (psum %~ seq_max (Seq.slice (reveal vr_s) tid (tid + 1))));
  (**)fold (array1_pts_to_slice_max sa tid (tid + 1) vr_s);
  (**)assert (pure (pow2 (SZ.v !n) == 1));
  (**)assert (pure (min (SZ.v tid + pow2 (SZ.v !n)) (SZ.v nth) == SZ.v tid + 1));
  (**)rewrite (array1_pts_to_slice_max sa tid (tid + 1) vr_s)
  (**)     as (array1_pts_to_slice_max sa tid (min (tid + pow2 !n) nth) vr_s);
  (**)if_intro_true' (div_pow2 !n tid) (array1_pts_to_slice_max sa tid (min (tid + pow2 !n) nth) vr_s);

  open FStar.SizeT;
  while (spow2 !n <^ nth)
    invariant
      live n **
      B.barrier_state !n **
      if_ (div_pow2 !n tid) (array1_pts_to_slice_max sa tid (min (tid + pow2 !n) nth) vr_s) **
      pure (v !n > 0 ==> pow2 (v !n - 1) < v nth)
    decreases (2 * nth - spow2 !n)
  {
    assert pure (Seq.length va == SZ.v lena);
    iteration nth sa vr_s tid !n;
    n := !n +^ 1sz;
  };

  with it. assert (B.barrier_state it);

  // After loop exit: pow2 it >= nth, and tid < nth, so div_pow2 it tid <==> tid = 0
  FStar.Math.Lemmas.modulo_lemma tid (pow2 it);
  rewrite
    (if_ (div_pow2 it tid) (array1_pts_to_slice_max sa tid (min (tid + pow2 it) nth) vr_s))
  as
    (if_ (op_Equality #nat tid 0) (array1_pts_to_slice_max sa 0 nth vr_s));

  log2_hreduce (v nth) it;
  rewrite (B.barrier_state it) as (B.barrier_state (hreduce_barrier_count nth));

  (* Thread zero owns the result at the end, and writes it out. *)
  if (tid = 0sz) {
    if_elim_true' (op_Equality #nat tid 0) (array1_pts_to_slice_max sa 0 nth vr_s);
    if_elim_true' (op_Equality #nat tid 0) (live out);
    unfold array1_pts_to_slice_max sa 0 nth vr_s;
    (**)strided_max_is_max pre_map_r vr nth;
    (**)assert (pure (Seq.equal (Seq.slice (reveal vr_s) 0 nth) (reveal vr_s)));
    gpu_write out (array1_read_from_slice sa 0sz);
    with ss. assert array1_pts_to_slice sa 0 nth ss;
    unfold array1_pts_to_slice sa;
    let bij : Kuiper.Bijection.bijection (k:nat{0 <= k /\ k < nth}) (Array1.ait nth) =
      Kuiper.Bijection.Mkbijection
        #(k:nat{0 <= k /\ k < nth})
        #(Array1.ait nth)
        (fun k -> k)
        (fun k -> k);
    forevery_iso bij _;
    forevery_ext _ (fun (k : natlt nth) -> Cell sa k |-> (ss @! k));
    Array1.implode sa;
    rewrite each sa as Array1.from_array (l1_forward nth) shmem._1;
    if_intro_true' (op_Equality #nat tid 0) (
      live (Array1.from_array (l1_forward nth) shmem._1) **
      exists* (v : et). out |-> v ** pure (v %~ seq_max (seq_map pre_map_r vr))
    )
  } else {
    (* Nop, convince Pulse. *)
    if_elim_false' (op_Equality #nat tid 0) (array1_pts_to_slice_max sa 0 nth vr_s);
    if_elim_false' (op_Equality #nat tid 0) (live out);
    if_intro_false' (op_Equality #nat tid 0) (
      live (Array1.from_array (l1_forward nth) shmem._1) **
      exists* (v : et). out |-> v ** pure (v %~ seq_max (seq_map pre_map_r vr))
    );
    ();
  };
}
#pop-options

ghost
fn block_setup
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= lena })
  (#l : Array1.layout lena)
  (a : Array1.t et l)
  (#va : lseq et lena)
  (vr : lseq real lena { va %~ vr })
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
  Array1.share_n a nth;

  (* tid 0 gets the ref *)
  forevery_if_intro #(natlt nth) 0 (fun _ -> live out);
  (* Sad.*)
  forevery_ext
    (fun tid -> if_ (op_Equality #(natlt nth) tid 0) (live out))
    (fun tid -> if_ (op_Equality #nat tid 0) (live out));

  forevery_zip (fun _ -> a |-> Frac (1 /. nth) va) _;

  (* View shmem array as Array1. Explode it. *)
  Array1.raise' (l1_forward nth) gsa;
  Array1.explode (Array1.from_array (l1_forward nth) gsa);

  forevery_zip #(natlt nth)
    (fun tid -> a |-> Frac (1 /. nth) va ** if_ (op_Equality #nat tid 0) (live out))
    _;

  forevery_map
    #(natlt nth)
    (fun tid ->
      (a |-> Frac (1 /. nth) va **
       if_ (op_Equality #nat tid 0) (live out)) **
      Cell (Array1.from_array (l1_forward nth) gsa) tid |-> (Array1.from_seq (l1_forward nth) vgsa @! tid)
    )
    (fun (tid : natlt nth) -> kpre pre_map pre_map_r nth lena a va vr out shmem bid tid)
    fn tid {
      rewrite each gsa as shmem._1;
      ();
    };

  ()
}


#push-options "--z3rlimit 40 --fuel 2 --ifuel 2"
ghost
fn block_teardown
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= lena })
  (#l : Array1.layout lena)
  (a : Array1.t et l)
  (#va : lseq et lena)
  (vr : lseq real lena { va %~ vr })
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
    (a |-> va ** (exists* (v : et). out |-> v ** pure (v %~ seq_max (lseq_map pre_map_r vr))))
{
  forevery_unzip _ _;

  Array1.gather_n a nth;

  (* Sad.*)
  forevery_ext #(natlt nth)
    (fun tid ->
      if_ (op_Equality #nat tid 0) (
        live (Array1.from_array (l1_forward nth) shmem._1) **
        exists* (v : et). out |-> v ** pure (v %~ seq_max (lseq_map pre_map_r vr))))
    (fun tid ->
      if_ (op_Equality #(natlt nth) tid 0) (
        live (Array1.from_array (l1_forward nth) shmem._1) **
        exists* (v : et). out |-> v ** pure (v %~ seq_max (lseq_map pre_map_r vr))));

  forevery_if_elim #(natlt nth) 0 (fun tid ->
      live (Array1.from_array (l1_forward nth) shmem._1) **
      exists* (v : et). out |-> v ** pure (v %~ seq_max (lseq_map pre_map_r vr))
  );

  Array1.lower (Array1.from_array (l1_forward nth) shmem._1);
  rewrite each Array1.core (Array1.from_array (l1_forward nth) shmem._1) as shmem._1;

  fold_live_c_shmems_nil shmem._2 #_;
  with vgsa. assert shmem._1 |-> vgsa;
  fold_live_c_shmem shmem._1;
  fold_live_c_shmems_cons shmem #_;
}
#pop-options

ghost
fn setup
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= lena })
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
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= lena })
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l { Array1.is_global a })
  (#va : erased (lseq et lena))
  (vr : erased (lseq real lena) { va %~ vr })
  (#_ : squash (Seq.length va == SZ.v lena))
  (out : gpu_ref et)
  ()
  norewrite
  requires
    (forall+ (bid : natlt 1). a |-> va ** exists* (v : et). out |-> v ** pure (v %~ seq_max (lseq_map pre_map_r vr))) **
    emp
  ensures
    a |-> va ** (exists* (v : et). out |-> v ** pure (v %~ seq_max (lseq_map pre_map_r vr)))
{
  forevery_singleton_elim #(natlt 1) _;
}

inline_for_extraction noextract
let kernel
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= lena })
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l { Array1.is_global a })
  (#va : erased (lseq et lena))
  (vr : erased (lseq real lena) { va %~ vr })
  (#_ : squash (Seq.length va == SZ.v lena))
  (out : gpu_ref et)
  : kernel_desc
      (a |-> va ** live out)
      (a |-> va ** exists* (v : et). out |-> v ** pure (v %~ seq_max (lseq_map pre_map_r vr)))
  = {
    nblk = 1sz;
    nthr = nth;

    shmems_desc = [SHArray et nth];

    barrier_contract = (fun _bid shmem ->
      mbarrier_contract (barrier_matrix #et nth (Array1.from_array _ shmem._1) (vr_partial_max pre_map_r vr nth)));
    barrier_count    = (fun _bid    -> hreduce_barrier_count nth);
    barrier_ok       = (fun _bid shmem ->
      mbarrier_transform (barrier_matrix nth #(l1_forward nth) (Array1.from_array _ shmem._1) (vr_partial_max pre_map_r vr nth)));

    f = kf pre_map pre_map_r nth lena a va vr out;

    block_pre  = (fun bid -> a |-> va ** live out);
    block_post = (fun bid -> a |-> va ** exists* (v : et). out |-> v ** pure (v %~ seq_max (lseq_map pre_map_r vr)));
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
fn reduce_max
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : szp)
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l { Array1.is_global a })
  (#va : erased (lseq et lena))
  (vr : erased (lseq real lena))
  norewrite // sigh... spec in fsti is not purified
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    pure (va %~ vr) **
    pure (0 < SZ.v nth /\ SZ.v nth <= lena) **
    pure (SZ.fits (lena + nth))
  returns
    res : et
  ensures
    pure (res %~ seq_max (seq_map pre_map_r vr))
{
  let out = Kuiper.Ref.gpu_alloc0 #et ();
  launch_sync (kernel pre_map pre_map_r nth lena a vr out);

  (* Bring back out result, free swap. *)
  let mut hout : et = zero #et;
  Kuiper.Ref.gpu_memcpy_device_to_host hout out;
  Kuiper.Ref.gpu_free out;

  !hout;
}
