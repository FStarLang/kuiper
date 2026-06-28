module Kuiper.Kernel.HReduce

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Seq.Common
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
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) })
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
      exists* (v : et). out |-> v ** pure (v %~ rsum (seq_map pre_map_r vr))
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

(* ---- Helpers for strided_sum_is_sum ---- *)

private let rsum_snoc_ (s : seq real) (x : real)
  : Lemma (rsum (Seq.snoc s x) == rsum s +. x)
  = assert (Seq.equal (Seq.snoc s x) (Seq.append s (Seq.create 1 x)));
    rsum_append s (Seq.create 1 x)

#push-options "--z3rlimit 20"
private let stride_len_snoc_ (n : nat) (nth : pos) (off : natlt nth)
  : Lemma (ensures (
      let old_len = (n - off + nth - 1) / nth in
      let new_len = (n + 1 - off + nth - 1) / nth in
      if n % nth = off then new_len == old_len + 1
      else new_len == old_len))
  = let m = n - off + nth - 1 in
    FStar.Math.Lemmas.euclidean_division_definition m nth;
    FStar.Math.Lemmas.euclidean_division_definition (m+1) nth;
    FStar.Math.Lemmas.euclidean_division_definition n nth;
    FStar.Math.Lemmas.modulo_addition_lemma (n % nth + nth - 1 - off) nth (n / nth);
    assert (m % nth == (n % nth + nth - 1 - off) % nth);
    if n % nth <= off then begin
      FStar.Math.Lemmas.small_mod (n % nth + nth - 1 - off) nth
    end else begin
      FStar.Math.Lemmas.modulo_addition_lemma (n % nth - 1 - off) nth 1;
      FStar.Math.Lemmas.small_mod (n % nth - 1 - off) nth
    end
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 30"
private let seq_stride_snoc_ (s : seq real) (x : real) (nth : pos) (off : natlt nth)
  : Lemma (ensures (
      let s' = Seq.snoc s x in
      let n = Seq.length s in
      if n % nth = off
      then Seq.equal (seq_stride s' nth off) (Seq.snoc (seq_stride s nth off) x)
      else Seq.equal (seq_stride s' nth off) (seq_stride s nth off)))
  = let n = Seq.length s in
    let s' = Seq.snoc s x in
    stride_len_snoc_ n nth off;
    let old_stride = seq_stride s nth off in
    let new_stride = seq_stride s' nth off in
    if n % nth = off then begin
      let old_sl = seq_stride_length s nth off in
      let new_sl = seq_stride_length s' nth off in
      assert (new_sl == old_sl + 1);
      let goal = Seq.snoc old_stride x in
      let aux (i : nat{i < new_sl}) : Lemma (Seq.index new_stride i == Seq.index goal i)
        = if i < old_sl then
            assert (off + i * nth < n)
          else begin
            assert (i == old_sl);
            assert (off + old_sl * nth == n)
          end
      in
      Classical.forall_intro aux;
      assert (Seq.equal new_stride goal)
    end else begin
      let old_sl = seq_stride_length s nth off in
      let aux (i : nat{i < old_sl}) : Lemma (Seq.index new_stride i == Seq.index old_stride i)
        = assert ((off + i * nth) % nth == off);
          assert (off + i * nth <> n);
          assert (off + i * nth < n)
      in
      Classical.forall_intro aux;
      assert (Seq.equal new_stride old_stride)
    end
#pop-options

#push-options "--z3rlimit 10"
private let rsum_seq_take_next_ (s : seq real) (n : nat{n < Seq.length s})
  : Lemma (rsum (seq_take n s) +. (s @! n) == rsum (seq_take (n + 1) s))
  = assert (Seq.equal (seq_take (n + 1) s) (Seq.snoc (seq_take n s) (s @! n)));
    rsum_snoc_ (seq_take n s) (s @! n)

private let rsum_singleton_ (x : real)
  : Lemma (rsum (Seq.create 1 x) == x)
  = let SCons hd tl = view_seq (Seq.create 1 x) in
    assert (Seq.equal tl (Seq.empty #real))

private let rsum_upd_ (s : seq real) (k : nat{k < Seq.length s}) (v : real)
  : Lemma (rsum (Seq.upd s k v) == rsum s +. (v -. (s @! k)))
  = let s1 = Seq.slice s 0 k in
    let s2 = Seq.slice s (k+1) (Seq.length s) in
    assert (Seq.equal s (s1 @+ Seq.create 1 (s @! k) @+ s2));
    assert (Seq.equal (Seq.upd s k v) (s1 @+ Seq.create 1 v @+ s2));
    rsum_append s1 (Seq.create 1 (s @! k) @+ s2);
    rsum_append (Seq.create 1 (s @! k)) s2;
    rsum_append s1 (Seq.create 1 v @+ s2);
    rsum_append (Seq.create 1 v) s2;
    rsum_singleton_ (s @! k);
    rsum_singleton_ v
#pop-options

private let rec rsum_zeros_ (n : nat)
  : Lemma (ensures rsum (Seq.init_ghost n (fun _ -> 0.0R)) == 0.0R)
          (decreases n)
  = if n = 0 then ()
    else begin
      let s = Seq.init_ghost n (fun (_:nat{_ < n}) -> 0.0R) in
      let SCons hd tl = view_seq s in
      assert (hd == 0.0R);
      assert (Seq.equal tl (Seq.init_ghost (n-1) (fun (_:nat{_ < n-1}) -> 0.0R)));
      rsum_zeros_ (n-1)
    end

#push-options "--z3rlimit 20"
private let rec strided_sum_is_sum_core_ (s : seq real) (nth : pos)
  : Lemma (ensures rsum (Seq.init_ghost nth (fun tid -> rsum (seq_stride s nth tid))) == rsum s)
          (decreases Seq.length s)
  = if Seq.length s = 0 then begin
      assert (Seq.equal s (Seq.empty #real));
      let aux (tid : natlt nth) : Lemma (rsum (seq_stride s nth tid) == 0.0R)
        = assert (seq_stride_length s nth tid == 0);
          assert (Seq.equal (seq_stride s nth tid) (Seq.empty #real))
      in
      Classical.forall_intro aux;
      let ig = Seq.init_ghost nth (fun tid -> rsum (seq_stride s nth tid)) in
      rsum_zeros_ nth;
      let z = Seq.init_ghost nth (fun _ -> 0.0R) in
      assert (forall (tid:natlt nth). ig @! tid == 0.0R);
      assert (forall (tid:natlt nth). z @! tid == 0.0R);
      Seq.lemma_eq_elim ig z;
      assert (rsum ig == 0.0R);
      assert (rsum s == 0.0R)
    end else begin
      let s', last = Seq.un_snoc s in
      let n = Seq.length s' in
      let off : natlt nth = n % nth in

      strided_sum_is_sum_core_ s' nth;
      assert (Seq.equal s (Seq.snoc s' last));
      rsum_snoc_ s' last;

      let f  (tid : natlt nth) : GTot real = rsum (seq_stride s  nth tid) in
      let f' (tid : natlt nth) : GTot real = rsum (seq_stride s' nth tid) in

      let aux (tid : natlt nth) : Lemma (
        if tid = off then f tid == f' tid +. last
        else f tid == f' tid)
        = seq_stride_snoc_ s' last nth tid;
          if tid = off then
            rsum_snoc_ (seq_stride s' nth tid) last
          else ()
      in
      Classical.forall_intro aux;

      let ig  = Seq.init_ghost nth (fun tid -> rsum (seq_stride s nth tid)) in
      let ig' = Seq.init_ghost nth (fun tid -> rsum (seq_stride s' nth tid)) in

      let upd_ig' = Seq.upd ig' off (rsum (seq_stride s' nth off) +. last) in
      let eq_aux (i : natlt nth) : Lemma (ig @! i == upd_ig' @! i)
        = if i = off then
            assert (f i == f' i +. last)
          else
            assert (f i == f' i)
      in
      Classical.forall_intro eq_aux;
      assert (Seq.equal ig upd_ig');

      rsum_upd_ ig' off (rsum (seq_stride s' nth off) +. last);
      assert (ig' @! off == rsum (seq_stride s' nth off));
      rsum_singleton_ (rsum (seq_stride s' nth off) +. last);
      rsum_singleton_ (rsum (seq_stride s' nth off))
    end
#pop-options

(* ---- End helpers ---- *)

let vr_partial (pre_map : real -> real) (vr : seq real) (nth : nat) : GTot (seq real) =
  Seq.init_ghost nth (fun tid -> rsum (seq_stride (seq_map pre_map vr) nth tid))

let strided_sum_is_sum (pre_map : real -> real) (vr : seq real) (nth : pos)
  : Lemma (ensures rsum (vr_partial pre_map vr nth) == rsum (seq_map pre_map vr))
  = strided_sum_is_sum_core_ (seq_map pre_map vr) nth

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
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l)
  (stride : szp)
  (off : szlt stride)
  (#va : erased (lseq et lena))
  (vr : erased (lseq real lena))
  (#f : perm)
  preserves
    gpu ** a |-> Frac f va ** pure (va %~ vr) ** pure (SZ.fits (lena + stride))
  returns
    res : et
  ensures
    pure (res %~ rsum (seq_stride (lseq_map pre_map_r vr) stride off))
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
            gread gidx <= seq_stride_length (lseq_map pre_map_r vr) stride off /\
            // gread gidx == (SZ.v !idx - SZ.v off) / SZ.v stride /\ // superfluous
            !acc %~ rsum (seq_take (gread gidx) (seq_stride (lseq_map pre_map_r vr) stride off))) **
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

    assert pure (!acc %~ rsum (seq_take (gread gidx) (seq_stride (lseq_map pre_map_r vr) stride off)));
    assert pure (v %~ (vr @! !idx));
    assert pure (seq_stride (lseq_map pre_map_r vr) stride off @! gread gidx == (lseq_map pre_map_r vr) @! (off + gread gidx * stride));
    assert pure (off + gread gidx * stride == SZ.v !idx);
    rsum_seq_take_next_ (seq_stride (lseq_map pre_map_r vr) stride off) (gread gidx);

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

  assert pure (gread gidx <= seq_stride_length (lseq_map pre_map_r vr) stride off);
  Math.Lemmas.cancel_mul_div (gread gidx) stride;
  (* A calc proof would be much nicer. *)
  assert pure (gread gidx == (!idx - off) / stride);
  assert pure (gread gidx == ((off + ((lena - off - 1 + stride) / stride) * stride) - off) / stride);
  assert pure (gread gidx == (((lena - off - 1 + stride) / stride) * stride) / stride);
  assert pure (gread gidx == (lena - off - 1 + stride) / stride);
  assert pure (lena - off - 1 + stride == lena - off + stride - 1);
  assert pure (gread gidx == (lena - off + stride - 1) / stride);
  assert pure (gread gidx == seq_stride_length (lseq_map pre_map_r vr) stride off);
  assert pure (seq_take (seq_stride_length (lseq_map pre_map_r vr) stride off) (seq_stride (lseq_map pre_map_r vr) stride off) == seq_stride (lseq_map pre_map_r vr) stride off);
  assert pure (!acc %~ rsum (seq_stride (lseq_map pre_map_r vr) stride off));

  drop_ (gidx |-> _);

  !acc
}
#pop-options

#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn kf
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) })
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
    mbarrier_tok nth (barrier_matrix nth (Array1.from_array (l1_forward nth) shmem._1) (vr_partial pre_map_r vr nth)) **
    B.barrier_state 0
  ensures
    gpu **
    kpost pre_map pre_map_r nth lena a va vr out shmem bid tid **
    thread_id nth tid **
    block_id 1 bid **
    mbarrier_tok nth (barrier_matrix nth (Array1.from_array (l1_forward nth) shmem._1) (vr_partial pre_map_r vr nth)) **
    B.barrier_state (hreduce_barrier_count nth)
{
  let (gsa, _) = shmem;

  let sa = Array1.from_array (l1_forward nth) gsa;
  rewrite each Array1.from_array (l1_forward nth) gsa as sa;

  let vr_s : erased (lseq real nth) = vr_partial pre_map_r vr nth;

  (* Compute partial sum and write to shmem *)
  let psum : et = sum_stride_map pre_map pre_map_r lena a nth tid vr;
  Array1.write_cell sa tid psum;

  (* Now do tree reduction on shmem *)
  let mut n : szlt 32 = 0sz;

  forevery_singleton_intro'
    #(x:nat{tid <= x /\ x < tid + 1})
    (fun x -> Cell sa (x <: natlt nth) |-> (seq![psum] @! (x - tid)))
    tid;
  fold array1_pts_to_slice sa tid (tid+1) seq![psum];

  (**)fold (array1_pts_to_slice_sum sa tid (tid + 1) vr_s);
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
    assert pure (Seq.length va == SZ.v lena);
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
      exists* (v : et). out |-> v ** pure (v %~ rsum (seq_map pre_map_r vr))
    )
  } else {
    (* Nop, convince Pulse. *)
    if_elim_false' (op_Equality #nat tid 0) (array1_pts_to_slice_sum sa 0 nth vr_s);
    if_elim_false' (op_Equality #nat tid 0) (live out);
    if_intro_false' (op_Equality #nat tid 0) (
      live (Array1.from_array (l1_forward nth) shmem._1) **
      exists* (v : et). out |-> v ** pure (v %~ rsum (seq_map pre_map_r vr))
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


ghost
fn block_teardown
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) })
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
    (a |-> va ** (exists* (v : et). out |-> v ** pure (v %~ rsum (lseq_map pre_map_r vr))))
{
  forevery_unzip _ _;

  Array1.gather_n a nth;

  (* Sad.*)
  forevery_ext #(natlt nth)
    (fun tid ->
      if_ (op_Equality #nat tid 0) (
        live (Array1.from_array (l1_forward nth) shmem._1) **
        exists* (v : et). out |-> v ** pure (v %~ rsum (lseq_map pre_map_r vr))))
    (fun tid ->
      if_ (op_Equality #(natlt nth) tid 0) (
        live (Array1.from_array (l1_forward nth) shmem._1) **
        exists* (v : et). out |-> v ** pure (v %~ rsum (lseq_map pre_map_r vr))));

  forevery_if_elim #(natlt nth) 0 (fun tid ->
      live (Array1.from_array (l1_forward nth) shmem._1) **
      exists* (v : et). out |-> v ** pure (v %~ rsum (lseq_map pre_map_r vr))
  );

  Array1.lower (Array1.from_array (l1_forward nth) shmem._1);
  rewrite each Array1.core (Array1.from_array (l1_forward nth) shmem._1) as shmem._1;

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
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) })
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l { Array1.is_global a })
  (#va : erased (lseq et lena))
  (vr : erased (lseq real lena) { va %~ vr })
  (#_ : squash (Seq.length va == SZ.v lena))
  (out : gpu_ref et)
  ()
  norewrite
  requires
    (forall+ (bid : natlt 1). a |-> va ** exists* (v : et). out |-> v ** pure (v %~ rsum (lseq_map pre_map_r vr))) **
    emp
  ensures
    a |-> va ** (exists* (v : et). out |-> v ** pure (v %~ rsum (lseq_map pre_map_r vr)))
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
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l { Array1.is_global a })
  (#va : erased (lseq et lena))
  (vr : erased (lseq real lena) { va %~ vr })
  (#_ : squash (Seq.length va == SZ.v lena))
  (out : gpu_ref et)
  : kernel_desc
      (a |-> va ** live out)
      (a |-> va ** exists* (v : et). out |-> v ** pure (v %~ rsum (lseq_map pre_map_r vr)))
  = {
    nblk = 1sz;
    nthr = nth;

    shmems_desc = [SHArray et nth];

    barrier_contract = (fun _bid shmem ->
      mbarrier_contract (barrier_matrix #et nth (Array1.from_array _ shmem._1) (vr_partial pre_map_r vr nth)));
    barrier_count    = (fun _bid    -> hreduce_barrier_count nth);
    barrier_ok       = (fun _bid shmem ->
      mbarrier_transform (barrier_matrix nth #(l1_forward nth) (Array1.from_array _ shmem._1) (vr_partial pre_map_r vr nth)));

    f = kf pre_map pre_map_r nth lena a va vr out;

    block_pre  = (fun bid -> a |-> va ** live out);
    block_post = (fun bid -> a |-> va ** exists* (v : et). out |-> v ** pure (v %~ rsum (lseq_map pre_map_r vr)));
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
    pure (SZ.fits (lena + nth))
  returns
    res : et
  ensures
    pure (res %~ rsum (seq_map pre_map_r vr))
{
  let out = Kuiper.Ref.gpu_alloc0 #et ();
  launch_sync (kernel pre_map pre_map_r nth lena a vr out);

  (* Bring back out result, free swap. *)
  let mut hout : et = zero #et;
  Kuiper.Ref.gpu_memcpy_device_to_host hout out;
  Kuiper.Ref.gpu_free out;

  !hout;
}
