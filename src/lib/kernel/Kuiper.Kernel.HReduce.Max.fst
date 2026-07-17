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
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Pulse.Lib.GhostReference { read as gread, write as gwrite, alloc as galloc }
open Kuiper.Tensor
open Kuiper.Chest1.Helpers
open Kuiper.Bijection { ( =~ ), bij_sym }
// Re-open after Kuiper.Tensor so the seq-level `@!`/`seq![..]`/`@+` notations
// shadow the shape-indexing `@!` pulled in via Kuiper.Shape.
open Kuiper.Seq.Common
open Kuiper.Math.OnlineSoftmax { seq_max, seq_max_cons_lem }

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
   [natlt len], used to (un)reindex a forevery over tensor cells. *)
let abs_bij (#len : nat) : (abs (len @| INil) =~ natlt len) =
  {
    ff = (fun (i, ()) -> i);
    gg = (fun i -> (i, ()));
  }

(* Same bijection but with a refined [nat] domain, matching the binder produced
   by unfolding [array1_pts_to_slice r 0 nth]. *)
let nat_abs_bij (nth : nat) : ((k:nat{0 <= k /\ k < nth}) =~ abs (nth @| INil)) =
  {
    ff = (fun k -> ((k, ()) <: abs (nth @| INil)));
    gg = (fun (i, ()) -> i);
  }

(* [chest1_to_seq] commutes with mapping; bridges the chest1 interface (in the
   .fsti) to the seq-based numeric proof below. *)
let chest_map_to_seq_map (#et1 #et2 : Type) (#n : nat)
  (f : et1 -> et2) (c : chest1 et1 n)
  : Lemma (chest1_to_seq (chest_map f c) == seq_map f (chest1_to_seq c))
  = assert (Seq.equal (chest1_to_seq (chest_map f c)) (seq_map f (chest1_to_seq c)))

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

(* Build a length-one slice from a single owned cell. *)
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
is the max of all the values in the (original) slice rr. *)
unfold
let array1_pts_to_slice_max_inner
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (#sz : nat)
  (#l : layout1 sz)
  (r : array1 et l)
  (i j : nat{i < j /\ j <= sz})
  (rr : lseq real sz)
  (s : chest1 et (j - i))
  : slprop
  = array1_pts_to_slice r i j s **
    pure ((acc1 s 0) %~ seq_max (Seq.slice rr i j))

let array1_pts_to_slice_max
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (#sz : nat)
  (#l : layout1 sz)
  ([@@@mkey] r : array1 et l)
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

(* The max over the per-bucket maxima equals the max over the whole sequence. *)
(* The two directions of [strided_max_is_max], as separate top-level lemmas.
   Each is proved in its own clean context so it verifies at a low rlimit
   (the monolithic version sat at ~100% of rlimit 40 and flaked). *)
let strided_max_le (pre_map : real -> real) (vr : seq real) (nth : pos { nth <= Seq.length vr })
  : Lemma (ensures seq_max (vr_partial_max pre_map vr nth) <=. seq_max (seq_map pre_map vr))
  = let w = seq_map pre_map vr in
    let vp = vr_partial_max pre_map vr nth in
    let aux1 (tid : nat { tid < nth }) : Lemma ((vp @! tid) <=. seq_max w)
      = stride_nonempty w nth tid;
        let bucket = seq_stride w nth tid in
        let aux1a (i : nat { i < Seq.length bucket }) : Lemma ((bucket @! i) <=. seq_max w)
          = // bucket @! i == w @! (tid + i*nth)
            assert (tid + i * nth < Seq.length w);
            seq_max_ub w (tid + i * nth)
        in
        Classical.forall_intro aux1a;
        seq_max_le bucket (seq_max w)
    in
    Classical.forall_intro aux1;
    seq_max_le vp (seq_max w)

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

let strided_max_is_max (pre_map : real -> real) (vr : seq real) (nth : pos { nth <= Seq.length vr })
  : Lemma (ensures seq_max (vr_partial_max pre_map vr nth) == seq_max (seq_map pre_map vr))
  = strided_max_le pre_map vr nth;
    strided_max_ge pre_map vr nth
    // antisymmetry of <=. on reals closes the equality

// Barrier

let barrier_matrix
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp)
  (#l : layout1 nth)
  (r : array1 et l)
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
  (#l : layout1 nth)
  (r : array1 et l)
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
      tensor_pts_to_cell (from_array (l1_forward nth) shmem._1) (tid, ()) v

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
      exists* (v : et). out |-> v ** pure (v %~ seq_max (chest1_to_seq (chest_map pre_map_r vr)))
    )

inline_for_extraction
fn iteration
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp { SZ.v nth <= max_threads })
  (#l : layout1 nth) {| Kuiper.Tensor.ctlayout l |}
  (r : array1 et l)
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
  = let q = i / stride in
    let m = len - off - 1 in
    (* i == stride*q + off, so the bounds on i are linear in the opaque [stride*q]:
         m = len-off-1 < stride*q <= m + stride. *)
    FStar.Math.Lemmas.euclidean_division_definition i stride;
    assert (m < stride * q);
    assert (stride * q <= m + stride);
    (* goal reduces to q == m/stride + 1; prove by antisymmetry of the two bounds. *)
    FStar.Math.Lemmas.lemma_div_plus m 1 stride;                    // (m+stride)/stride == m/stride+1
    (* upper bound: q == (stride*q)/stride <= (m+stride)/stride == m/stride+1 *)
    FStar.Math.Lemmas.cancel_mul_div q stride;                      // (q*stride)/stride == q
    FStar.Math.Lemmas.lemma_div_le (stride * q) (m + stride) stride;
    (* lower bound: m/stride <= (stride*q-1)/stride == q-1, so m/stride+1 <= q *)
    FStar.Math.Lemmas.lemma_div_plus (stride - 1) (q - 1) stride;   // (stride*q-1)/stride == q-1
    FStar.Math.Lemmas.lemma_div_le m (stride * q - 1) stride

(* ── Extracted nat / strided-bucket arithmetic ─────────────────────────────
   The [max_stride_map] kernels below run in a context carrying an ambient
   quantified hypothesis [forall j. <input> j %~ <real> j].  Proving the
   (nonlinear / modular) stride arithmetic inline forces Z3 to interleave that
   quantifier with the arithmetic, which is pathologically slow.  We extract the
   pure-nat / generic-seq facts into top-level lemmas proven in a clean context
   and just call them (shared with [Kuiper.Kernel.HReduce.Block.Max]). *)

(* Loop step: advancing the flat index by one stride. *)
let stride_step_arith (idx vgidx stride off : nat)
  : Lemma (requires idx == vgidx * stride + off)
          (ensures  idx + stride == (vgidx + 1) * stride + off)
  = FStar.Math.Lemmas.distributivity_add_left vgidx 1 stride

(* Loop body: the running bucket index is strictly below the bucket count. *)
let stride_idx_in_bounds (cols off stride gidx idx : nat)
  : Lemma (requires off < stride /\ idx == gidx * stride + off /\ idx < cols)
          (ensures  gidx < (cols - off + stride - 1) / stride)
  = FStar.Math.Lemmas.cancel_mul_div gidx stride;
    FStar.Math.Lemmas.lemma_div_le (gidx * stride) (cols - off - 1) stride;
    FStar.Math.Lemmas.lemma_div_plus (cols - off - 1) 1 stride

(* Loop body: the [gidx]-th strided bucket element is the flat index [idx]. *)
let stride_bucket_index (#a:Type) (s : seq a) (stride : pos) (off : nat{off < stride})
                        (gidx idx : nat)
  : Lemma (requires idx == gidx * stride + off /\ gidx < seq_stride_length s stride off)
          (ensures  off + gidx * stride == idx /\
                    seq_stride s stride off @! gidx == s @! idx)
  = ()

(* Loop exit: the first strided index at/after [len s] pins down the bucket
   count exactly.  Stated over the concrete sequence so the caller gets the
   [seq_stride_length] equality directly (without re-deriving the modular
   arithmetic in its quantifier-polluted context). *)
let max_stride_post_arith (#a:Type) (s : seq a) (off stride gidx idx : nat)
  : Lemma (requires off < stride /\ off < Seq.length s /\
                    idx == gidx * stride + off /\
                    Seq.length s <= idx /\ idx < Seq.length s + stride)
          (ensures  gidx == seq_stride_length s stride off)
  = let cols = Seq.length s in
    FStar.Math.Lemmas.lemma_mod_plus off gidx stride;
    FStar.Math.Lemmas.small_mod off stride;
    assert (idx % stride == off);
    lemma_first_past cols off stride idx;
    FStar.Math.Lemmas.cancel_mul_div gidx stride;
    FStar.Math.Lemmas.cancel_mul_div ((cols - off - 1 + stride) / stride) stride

#push-options "--z3rlimit 60"
inline_for_extraction noextract
fn max_stride_map
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (lena : sz)
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l)
  (stride : szp)
  (off : szlt stride { SZ.v off < lena })
  (#va : chest1 et lena)
  (vr : chest1 real lena)
  (#f : perm)
  preserves
    gpu ** a |-> Frac f va ** pure (va %~ vr) ** pure (SZ.fits (lena + stride))
  returns
    res : et
  ensures
    pure (res %~ seq_max (seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off))
{
  (* off < lena, so the strided bucket at `off` is non-empty. *)
  stride_nonempty (seq_map pre_map_r (chest1_to_seq vr)) stride off;

  (* Initialize the accumulator with the first strided element (index off),
     so the running max is over a non-empty prefix.  seq_take 1 of the bucket
     is the singleton [w @! off]. *)
  let off_s : sz = off;
  let v0 = tensor_read a ((off_s <: szlt lena), ());
  let acc0 = pre_map v0;
  (**)assert (pure (v0 == acc1 va (SZ.v off)));
  (**)assert (pure (acc0 %~ (seq_map pre_map_r (chest1_to_seq vr) @! SZ.v off)));
  (**)assert (pure (seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off @! 0 == (seq_map pre_map_r (chest1_to_seq vr)) @! (off + 0 * stride)));
  (**)assert (pure (acc0 %~ (seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off @! 0)));
  (**)seq_max_singleton (seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off @! 0);
  (**)assert (pure (Seq.equal (seq_take 1 (seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off))
  (**)                        (seq![seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off @! 0])));
  (**)assert (pure (acc0 %~ seq_max (seq_take 1 (seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off))));

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
            gread gidx <= seq_stride_length (seq_map pre_map_r (chest1_to_seq vr)) stride off /\
            !acc %~ seq_max (seq_take (gread gidx) (seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off))) **
      emp
    decreases (lena + stride - !idx)
  {
    stride_idx_in_bounds (SZ.v lena) (SZ.v off) (SZ.v stride) (gread gidx) (SZ.v !idx);
    assert pure (gread gidx < seq_stride_length (seq_map pre_map_r (chest1_to_seq vr)) stride off);

    (* Read from input array (fractional permission) *)
    let vidx = !idx;
    let v = tensor_read a ((vidx <: szlt lena), ());
    let v' = pre_map v;
    (**)assert (pure (v == acc1 va (SZ.v !idx)));
    (**)assert (pure (v %~ (chest1_to_seq vr @! SZ.v !idx)));
    (**)assert (pure (v' %~ (seq_map pre_map_r (chest1_to_seq vr) @! SZ.v !idx)));

    assert pure (!acc %~ seq_max (seq_take (gread gidx) (seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off)));
    assert pure (v %~ (chest1_to_seq vr @! !idx));
    stride_bucket_index (seq_map pre_map_r (chest1_to_seq vr)) (SZ.v stride) (SZ.v off) (gread gidx) (SZ.v !idx);
    assert pure (seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off @! gread gidx == (seq_map pre_map_r (chest1_to_seq vr)) @! (SZ.v !idx));

    (* seq_take (k+1) maxes in the bucket's k-th element; combine with fmax. *)
    (**)seq_max_take_step (seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off) (gread gidx);

    let vgidx = gread gidx;
    assert (pure (SZ.v !idx == vgidx * stride + off));
    stride_step_arith (SZ.v !idx) vgidx (SZ.v stride) (SZ.v off);
    assert (pure (SZ.v !idx + stride == (vgidx + 1) * stride + off));

    Math.Lemmas.add_div_mod_1 (SZ.v !idx) stride;

    acc := !acc `fmax` v';
    idx := !idx +^ stride;
    gwrite gidx (gread gidx + 1);

    assert pure (SZ.v !idx == gread gidx * stride + off);
    ()
  };

  assert pure (SZ.v !idx == gread gidx * stride + off);
  max_stride_post_arith (seq_map pre_map_r (chest1_to_seq vr)) (SZ.v off) (SZ.v stride) (gread gidx) (SZ.v !idx);
  assert pure (gread gidx == seq_stride_length (seq_map pre_map_r (chest1_to_seq vr)) stride off);
  assert pure (seq_take (seq_stride_length (seq_map pre_map_r (chest1_to_seq vr)) stride off) (seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off) == seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off);
  assert pure (!acc %~ seq_max (seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off));

  drop_ (gidx |-> _);

  !acc
}
#pop-options

#push-options "--z3rlimit 20"
(* Pure, quantifier-free arithmetic for the first tree-reduction step, proved in
   a clean context so it does not consume [kf]'s (monolithic) VC budget under the
   ambient quantified slprops. *)
let kf_first_step_arith (tid nth k : nat)
  : Lemma (requires tid < nth /\ pow2 k == 1)
          (ensures tid + 1 <= nth /\ min (tid + pow2 k) nth == tid + 1)
  = ()

inline_for_extraction noextract
fn kf
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= lena })
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
    mbarrier_tok nth (barrier_matrix nth (from_array (l1_forward nth) shmem._1) (vr_partial_max pre_map_r (chest1_to_seq vr) nth)) **
    B.barrier_state 0
  ensures
    gpu **
    kpost pre_map pre_map_r nth lena a va vr out shmem bid tid **
    thread_id nth tid **
    block_id 1 bid **
    mbarrier_tok nth (barrier_matrix nth (from_array (l1_forward nth) shmem._1) (vr_partial_max pre_map_r (chest1_to_seq vr) nth)) **
    B.barrier_state (hreduce_barrier_count nth)
{
  let (gsa, _) = shmem;

  let sa = from_array (l1_forward nth) gsa;
  rewrite each from_array (l1_forward nth) gsa as sa;

  let vr_s : erased (lseq real nth) = vr_partial_max pre_map_r (chest1_to_seq vr) nth;

  (* Compute partial max and write to shmem *)
  (**)chest1_to_seq_approx va vr;
  let psum : et = max_stride_map pre_map pre_map_r lena a nth tid vr;
  tensor_write_cell sa (tid, ()) psum;

  (* Now do tree reduction on shmem *)
  let mut n : szlt 32 = 0sz;

  let psum_chest : chest1 et 1 = mk1 #et #1 (fun _ -> psum);
  slice_singleton sa (SZ.v tid) psum psum_chest;

  (* psum %~ vr_s @! tid == seq_max (single-element slice [tid, tid+1)) *)
  (**)stride_nonempty (seq_map pre_map_r (chest1_to_seq vr)) nth tid;
  (**)assert (pure (psum %~ (reveal vr_s @! SZ.v tid)));
  (**)seq_max_singleton (reveal vr_s @! SZ.v tid);
  (**)assert (pure (Seq.equal (Seq.slice (reveal vr_s) tid (tid + 1)) (seq![reveal vr_s @! SZ.v tid])));
  (**)assert (pure (acc1 psum_chest 0 %~ seq_max (Seq.slice (reveal vr_s) tid (tid + 1))));
  (**)fold (array1_pts_to_slice_max sa tid (tid + 1) vr_s);
  (**)kf_first_step_arith (SZ.v tid) (SZ.v nth) (SZ.v !n);
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
    (**)strided_max_is_max pre_map_r (chest1_to_seq vr) nth;
    (**)chest_map_to_seq_map pre_map_r vr;
    (**)assert (pure (Seq.equal (Seq.slice (reveal vr_s) 0 nth) (reveal vr_s)));
    gpu_write out (array1_read_from_slice sa 0sz);
    with ss. assert array1_pts_to_slice sa 0 nth ss;
    unfold array1_pts_to_slice sa;
    let css : erased (chest1 et nth) = hide (mk1 #et #nth (fun (k:natlt nth) -> acc1 ss k));
    (* Clean the index refinement [0<=k /\ k<nth] down to [k<nth] (= natlt nth),
       then reindex to the abstract tensor index and implode. *)
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
      exists* (v : et). out |-> v ** pure (v %~ seq_max (chest1_to_seq (chest_map pre_map_r vr)))
    )
  } else {
    (* Nop, convince Pulse. *)
    if_elim_false' (op_Equality #nat tid 0) (array1_pts_to_slice_max sa 0 nth vr_s);
    if_elim_false' (op_Equality #nat tid 0) (live out);
    if_intro_false' (op_Equality #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et). out |-> v ** pure (v %~ seq_max (chest1_to_seq (chest_map pre_map_r vr)))
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
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= lena })
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
    (a |-> va ** (exists* (v : et). out |-> v ** pure (v %~ seq_max (chest1_to_seq (chest_map pre_map_r vr)))))
{
  forevery_unzip _ _;

  tensor_gather_n a nth;

  (* Sad.*)
  forevery_ext #(natlt nth)
    (fun tid ->
      if_ (op_Equality #nat tid 0) (
        live (from_array (l1_forward nth) shmem._1) **
        exists* (v : et). out |-> v ** pure (v %~ seq_max (chest1_to_seq (chest_map pre_map_r vr)))))
    (fun tid ->
      if_ (op_Equality #(natlt nth) tid 0) (
        live (from_array (l1_forward nth) shmem._1) **
        exists* (v : et). out |-> v ** pure (v %~ seq_max (chest1_to_seq (chest_map pre_map_r vr)))));

  forevery_if_elim #(natlt nth) 0 (fun tid ->
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et). out |-> v ** pure (v %~ seq_max (chest1_to_seq (chest_map pre_map_r vr)))
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
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= lena })
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
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= lena })
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#va : chest1 et lena)
  (vr : chest1 real lena { va %~ vr })
  (out : gpu_ref et)
  ()
  norewrite
  requires
    (forall+ (bid : natlt 1). a |-> va ** exists* (v : et). out |-> v ** pure (v %~ seq_max (chest1_to_seq (chest_map pre_map_r vr)))) **
    emp
  ensures
    a |-> va ** (exists* (v : et). out |-> v ** pure (v %~ seq_max (chest1_to_seq (chest_map pre_map_r vr))))
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
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#va : chest1 et lena)
  (vr : chest1 real lena { va %~ vr })
  (out : gpu_ref et)
  : kernel_desc
      (a |-> va ** live out)
      (a |-> va ** exists* (v : et). out |-> v ** pure (v %~ seq_max (chest1_to_seq (chest_map pre_map_r vr))))
  = {
    nblk = 1sz;
    nthr = nth;

    shmems_desc = [SHArray et nth];

    barrier_contract = (fun _bid shmem ->
      mbarrier_contract (barrier_matrix #et nth (from_array _ shmem._1) (vr_partial_max pre_map_r (chest1_to_seq vr) nth)));
    barrier_count    = (fun _bid    -> hreduce_barrier_count nth);
    barrier_ok       = (fun _bid shmem ->
      mbarrier_transform (barrier_matrix nth #(l1_forward nth) (from_array _ shmem._1) (vr_partial_max pre_map_r (chest1_to_seq vr) nth)));

    f = kf pre_map pre_map_r nth lena a va vr out;

    block_pre  = (fun bid -> a |-> va ** live out);
    block_post = (fun bid -> a |-> va ** exists* (v : et). out |-> v ** pure (v %~ seq_max (chest1_to_seq (chest_map pre_map_r vr))));
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
    pure (0 < SZ.v nth /\ SZ.v nth <= lena) **
    pure (SZ.fits (lena + nth))
  returns
    res : et
  ensures
    pure (res %~ seq_max (chest1_to_seq (chest_map pre_map_r vr)))
{
  let out = Kuiper.Ref.gpu_alloc0 #et ();
  launch_sync (kernel pre_map pre_map_r nth lena a vr out);

  (* Bring back out result, free swap. *)
  let mut hout : et = zero #et;
  Kuiper.Ref.gpu_memcpy_device_to_host hout out;
  Kuiper.Ref.gpu_free out;

  !hout;
}
