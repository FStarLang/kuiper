module Kuiper.Kernel.RowReduce

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Seq.Common
open Kuiper.Tensor { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Pulse.Lib.GhostReference { read as gread, write as gwrite, alloc as galloc }
open Kuiper.Bijection { ( =~ ), bij_sym }
open Kuiper.Kernel.HReduce

module SZ = Kuiper.SizeT
module RPM = Kuiper.Barrier.RPM
module B = Kuiper.Barrier
module C = Kuiper.Matrix.Casts

(* ── Chest <-> seq bridges (ported from Kuiper.Kernel.HReduce) ───────────── *)

(* [abs (n @| INil)] is definitionally [natlt n & unit]; expose this to the SMT
   solver so that the abstract 1-D tensor index unifies with the explicit
   [(i, ())] tuples produced by reads/writes and [forevery] reindexings. *)
let abs_cons_nil_eq (n:nat)
  : Lemma (Kuiper.Shape.abs (n @| INil) == (natlt n & unit))
          [SMTPat (Kuiper.Shape.abs (n @| INil))]
  = ()

unfold
let abs_bij (#len : nat) : (Kuiper.Shape.abs (len @| INil) =~ natlt len) =
  {
    ff = (fun (i, ()) -> i);
    gg = (fun i -> (i, ()));
  }

let chest_map_to_seq_map (#et1 #et2 : Type) (#n : nat)
  (f : et1 -> et2) (c : chest1 et1 n)
  : Lemma (chest1_to_seq (chest_map f c) == seq_map f (chest1_to_seq c))
  = assert (Seq.equal (chest1_to_seq (chest_map f c)) (seq_map f (chest1_to_seq c)))

let chest1_rsum_map (#n : nat) (f : real -> real) (c : chest1 real n)
  : Lemma (chest1_rsum (chest_map f c) == rsum (seq_map f (chest1_to_seq c)))
  = chest_map_to_seq_map f c

(* ── Slice ownership of a 1-D tensor (World B: lseq content) ─────────────── *)

let array1_pts_to_slice
  (#et : Type0)
  (#sz : nat)
  (#l : layout1 sz)
  ([@@@mkey] r : array1 et l)
  ([@@@mkey]i
   [@@@mkey]j : nat{i <= j /\ j <= sz})
  (s : lseq et (j - i))
  : slprop
  = forall+ (k : nat{i <= k /\ k < j}).
      tensor_pts_to_cell r ((k <: natlt sz), ()) (s @! (k - i))

#push-options "--z3rlimit 40"
ghost
fn array1_slice_concat
  (#et : Type0)
  (#sz : nat)
  (#l : layout1 sz)
  (r : array1 et l)
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

  forevery_ext
    (fun (x:nat{i <= x /\ x < j}) -> tensor_pts_to_cell r ((x <: natlt sz), ()) (s1 @! (x - i)))
    (fun (x:nat{i <= x /\ x < j}) -> tensor_pts_to_cell r ((x <: natlt sz), ()) (s @! (x - i)));
  forevery_ext
    (fun (x:nat{j <= x /\ x < k}) -> tensor_pts_to_cell r ((x <: natlt sz), ()) (s2 @! (x - j)))
    (fun (x:nat{j <= x /\ x < k}) -> tensor_pts_to_cell r ((x <: natlt sz), ()) (s @! (x - i)));

  forevery_refine_join' #nat
    (fun (x:nat) -> i <= x /\ x < j)
    (fun (x:nat) -> j <= x /\ x < k)
    (fun (x:nat{(i <= x /\ x < j) \/ (j <= x /\ x < k)}) ->
      tensor_pts_to_cell r ((x <: natlt sz), ()) (s @! (x - i)));

  forevery_refine_ext' #nat
    #(fun (x:nat) -> (i <= x /\ x < j) \/ (j <= x /\ x < k))
    (fun (x:nat) -> i <= x /\ x < k)
    (fun (x:nat{(i <= x /\ x < j) \/ (j <= x /\ x < k)}) ->
      tensor_pts_to_cell r ((x <: natlt sz), ()) (s @! (x - i)));

  forevery_ext
    _
    (fun (x : nat{i <= x /\ x < k}) ->
      tensor_pts_to_cell r ((x <: natlt sz), ()) (s @! (x - i)));

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
  (#s : erased (lseq et (j - i)))
  (v : et)
  requires
    array1_pts_to_slice r i j s
  ensures
    array1_pts_to_slice r i j (Seq.upd s (idx - i) v)
{
  unfold array1_pts_to_slice r i j s;
  forevery_extract' #(x:nat{i <= x /\ x < j}) (SZ.v idx) _;
  tensor_write_cell r ((idx <: szlt len), ()) v;
  let s' : erased (lseq et (j - i)) = Seq.upd s (idx - i) v;
  Pulse.Lib.Forall.elim_forall
    (fun (x:nat{i <= x /\ x < j}) ->
      tensor_pts_to_cell r ((x <: natlt len), ()) (s' @! (x - i)));
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
  (#l : layout1 sz)
  (r : array1 et l)
  (i j : nat{i < j /\ j <= sz})
  (rr : lseq real sz)
  (s : lseq et (j - i))
  : slprop
  = array1_pts_to_slice r i j s **
    pure ((s @! 0) %~ rsum (Seq.slice rr i j))

let array1_pts_to_slice_sum
  (#et:Type0) {| scalar et, real_like et |}
  (#sz : nat)
  (#l : layout1 sz)
  ([@@@mkey] r : array1 et l)
  ([@@@mkey] i : nat)
  (j : nat{i < j /\ j <= sz})
  (rr : lseq real sz)
  : slprop
  = exists* s. array1_pts_to_slice_sum_inner r i j rr s

(* Barrier *)

let barrier_matrix
  (#et:Type0) {| scalar et, real_like et |}
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
           (array1_pts_to_slice_sum r from (min (from + pow2 it) nth) vr))

ghost
fn mk_barrier_pre
  (#et:Type0) {| scalar et, real_like et |}
  (nth : szp)
  (#l : layout1 nth)
  (r : array1 et l)
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

inline_for_extraction
fn iteration
  (#et:Type0) {| scalar et, real_like et |}
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

  let nextid = FStar.SizeT.(tid +^ spow2 it);

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

      array1_write_to_slice r tid s;

      (**)with seq. assert (array1_pts_to_slice r tid end_ seq);
      (**)fold (array1_pts_to_slice_sum r tid end_ vr);
      (**)if_intro_true (array1_pts_to_slice_sum r tid end_ vr);
      (**)rewrite
      (**)  if_ true
      (**)      (array1_pts_to_slice_sum r (SZ.v tid) (reveal end_) vr)
      (**)as
      (**)  if_ (div_pow2 (SZ.v it + 1) (SZ.v tid))
      (**)      (array1_pts_to_slice_sum r (SZ.v tid) (reveal end_) vr);
    } else {
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

(* ---- Helpers for strided_sum_is_sum ---- *)

let rsum_snoc_ (s : seq real) (x : real)
  : Lemma (rsum (Seq.snoc s x) == rsum s +. x)
  = assert (Seq.equal (Seq.snoc s x) (Seq.append s (Seq.create 1 x)));
    rsum_append s (Seq.create 1 x)

#push-options "--z3rlimit 20"
let stride_len_snoc_ (n : nat) (nth : pos) (off : natlt nth)
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
let seq_stride_snoc_ (s : seq real) (x : real) (nth : pos) (off : natlt nth)
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

let rsum_singleton_ (x : real)
  : Lemma (rsum (Seq.create 1 x) == x)
  = let SCons hd tl = view_seq (Seq.create 1 x) in
    assert (Seq.equal tl (Seq.empty #real))

#push-options "--z3rlimit 20"
let rsum_upd_ (s : seq real) (k : nat{k < Seq.length s}) (v : real)
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

let rec rsum_zeros_ (n : nat)
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

#push-options "--z3rlimit 30"
let rec strided_sum_is_sum_core_ (s : seq real) (nth : pos)
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

(* Per-thread partial sums (chest-friendly boundary, seq internals). *)
let vr_partial (pre_map : real -> real) (#lena : nat) (vr : chest1 real lena) (nth : nat)
  : GTot (lseq real nth) =
  Seq.init_ghost nth (fun tid -> rsum (seq_stride (seq_map pre_map (chest1_to_seq vr)) nth tid))

let strided_sum_is_sum (pre_map : real -> real) (#lena : nat) (vr : chest1 real lena) (nth : pos)
  : Lemma (ensures rsum (vr_partial pre_map vr nth) == chest1_rsum (chest_map pre_map vr))
  = chest1_rsum_map pre_map vr;
    strided_sum_is_sum_core_ (seq_map pre_map (chest1_to_seq vr)) nth

let lemma_first_past
  (len off : nat) (stride : pos)
  (i : nat)
  : Lemma (requires i % stride == off /\ i >= len /\ i < len + stride)
          (ensures  i == off + ((len - off - 1 + stride) / stride) * stride)
  = ()


(* Per-element step lemma for the strided reduction in [sum_stride_map_2d]:
   appending the [k]-th strided element to the running sum of the first [k]
   elements yields the running sum of the first [k+1] elements. Analogous to
   [rsum_seq_take_next_] used by the legacy [Kuiper.Kernel.HReduce.sum_stride_map]. *)
let rsum_seq_stride_step
  (rs : seq real)
  (stride : pos)
  (off : nat{off < stride})
  (k : nat)
  : Lemma (requires k < seq_stride_length rs stride off /\
                    k * stride + off < Seq.length rs)
          (ensures
            rsum (seq_take k (seq_stride rs stride off)) +. (rs @! (k * stride + off)) ==
            rsum (seq_take (k + 1) (seq_stride rs stride off)))
  = let ss = seq_stride rs stride off in
    let a = seq_take k ss in
    let single = Seq.slice ss k (k+1) in
    let v = rs @! (off + k * stride) in
    Kuiper.Seq.Common.lem_append_slice ss 0 k (k+1);
    assert (Seq.equal (seq_take (k+1) ss) (Seq.append a single));
    assert (Seq.length single == 1);
    assert (Seq.index single 0 == Seq.index ss k);
    assert (Seq.index ss k == v);
    Kuiper.Seq.Common.lem_one_elem single v;
    Kuiper.Approximates.rsum_append a single

inline_for_extraction noextract
fn read_at
  (#et:Type0) {| scalar et |}
  (rows : szp)
  (cols : szp)
  (#lin : layout2 rows cols) {| ctlayout lin |}
  (x : array2 et lin)
  (row : szlt rows)
  (col : szlt cols)
  (#sx : chest2 et rows cols)
  (#f : perm)
  preserves
    x |-> Frac f sx
  returns
    res : et
  ensures
    pure (res == acc2 sx row col)
{
  tensor_read x (row, (col, ()))
}

(* Drop a per-element [pure] clause from a [forevery] predicate. The
   global fact [forall x. q x] should typically be extracted beforehand
   via [forevery_extract_pure]; this helper just discards the per-cell
   [pure (q x)] clause so a subsequent [forevery_ext] only needs to
   match the residual slprop. *)
ghost
fn forevery_drop_pure
  (#a:Type0)
  (p : a -> slprop)
  (q : a -> prop)
  requires
    forall+ (x:a). p x ** pure (q x)
  ensures
    forall+ (x:a). p x
{
  forevery_map
    (fun (x:a) -> p x ** pure (q x))
    p
    fn x { drop_ (pure (q x)) }
}

// If k <= (n - off + stride - 1) / stride, k * stride + off >= n, and off < stride,
// then k == (n - off + stride - 1) / stride.
let stride_length_exact (k n stride off : nat)
  : Lemma
      (requires
        stride > 0 /\ off < stride /\
        k <= (n - off + stride - 1) / stride /\
        k * stride + off >= n)
      (ensures k == (n - off + stride - 1) / stride)
  = Math.Lemmas.lemma_div_le (k * stride) (n - off + stride - 1) stride;
    Math.Lemmas.cancel_mul_div k stride

#push-options "--fuel 2 --ifuel 2 --z3rlimit 40"
inline_for_extraction noextract
fn sum_stride_map_2d
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp)
  (cols : szp)
  (#lin : layout2 rows cols) {| ctlayout lin |}
  (x : array2 et lin)
  (row : szlt rows)
  (stride : szp)
  (off : szlt stride)
  (#sx : chest2 et (SZ.v rows) (SZ.v cols))
  (vr_row : erased (lseq real (SZ.v cols)))
  (#f : perm)
  preserves
    gpu ** x |-> Frac f sx **
    pure (forall (j:nat). j < SZ.v cols ==> acc2 sx (SZ.v row) j %~ (vr_row @! j)) **
    pure (SZ.fits (SZ.v cols + stride))
  returns
    res : et
  ensures
    pure (res %~ rsum (seq_stride (lseq_map pre_map_r vr_row) stride off))
{
  let mut acc : et = zero;
  let mut idx : sz = off;
  let gidx = galloc #nat 0;

  while (!idx <^ cols)
    invariant
      live acc ** live gidx ** live idx **
      pure (gread gidx <= seq_stride_length (lseq_map pre_map_r vr_row) stride off /\
            !idx < cols + stride /\
            !acc %~ rsum (seq_take (gread gidx) (seq_stride (lseq_map pre_map_r vr_row) stride off)) /\
            SZ.v !idx == gread gidx * stride + off
      ) **
      emp
    decreases (cols + stride - !idx)
  {
    assert pure (gread gidx < seq_stride_length vr_row stride off);

    let idx_raw : sz = !idx;
    assert pure (SZ.v idx_raw < SZ.v cols);
    let idx_v : szlt cols = idx_raw;
    let v = read_at rows cols x row idx_v;
    let v' = pre_map v;
    (**)assert (pure (v == acc2 sx (SZ.v row) (SZ.v idx_v)));
    (**)assert (pure (v %~ (vr_row @! SZ.v idx_v)));
    (**)assert (pure (v' %~ (lseq_map pre_map_r vr_row @! SZ.v idx_v)));

    a_add !acc v'
      (rsum (seq_take (gread gidx) (seq_stride (lseq_map pre_map_r vr_row) stride off)))
      ((lseq_map pre_map_r vr_row) @! SZ.v idx_v);
    rsum_seq_stride_step (lseq_map pre_map_r vr_row) stride off (gread gidx);

    let vgidx = gread gidx;
    Math.Lemmas.distributivity_add_left vgidx 1 stride;

    acc := !acc `add` v';
    idx := !idx +^ stride;
    gwrite gidx (gread gidx + 1);
    ()
  };

  stride_length_exact (gread gidx) (Seq.length (lseq_map pre_map_r vr_row)) (SZ.v stride) (SZ.v off);
  assert pure (gread gidx == seq_stride_length (lseq_map pre_map_r vr_row) stride off);
  assert pure (seq_take (seq_stride_length (lseq_map pre_map_r vr_row) stride off)
                       (seq_stride (lseq_map pre_map_r vr_row) stride off)
              == seq_stride (lseq_map pre_map_r vr_row) stride off);
  drop_ (gidx |-> _);
  !acc
}
#pop-options

(* ── Per-thread predicates for the per-block kernel ────────────────────── *)

unfold
let kpre_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols)
  (sout : chest1 et rows)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt (SZ.v rows))
  (tid : natlt nth)
  : slprop
  = x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
    if_ (op_Equality #nat tid 0) (
      tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) (sout `acc1` bid)) **
    exists* (v : et). tensor_pts_to_cell (from_array (l1_forward nth) shmem._1) (abs_bij.gg (tid <: natlt nth)) v

unfold
let kpost_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols)
  (sout : chest1 et rows)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  (tid : natlt nth)
  : slprop
  = x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
    if_ (op_Equality #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et).
        tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) (v) **
        pure (v %~ chest1_rsum (chest_map pre_map_r (chest2_row vr bid)))
    )

(* ── Per-thread kernel function ────────────────────────────────────────── *)

#push-options "--z3rlimit 40 --z3seed 1"
inline_for_extraction noextract
fn kf_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)      {| ctlayout lout |}
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols { sx %~ vr })
  (sout : chest1 et (SZ.v rows))
  (shmem : c_shmems [SHArray et nth])
  (bid : szlt rows)
  (tid : szlt nth)
  ()
  requires
    gpu **
    kpre_block pre_map pre_map_r rows cols nth x output sx vr sout shmem (SZ.v bid) (SZ.v tid) **
    thread_id nth tid **
    block_id rows bid **
    mbarrier_tok nth (barrier_matrix nth (from_array (l1_forward nth) shmem._1)
                       (vr_partial pre_map_r (chest2_row vr (SZ.v bid)) nth)) **
    B.barrier_state 0
  ensures
    gpu **
    kpost_block pre_map pre_map_r rows cols nth x output sx vr sout shmem (SZ.v bid) (SZ.v tid) **
    thread_id nth tid **
    block_id rows bid **
    mbarrier_tok nth (barrier_matrix nth (from_array (l1_forward nth) shmem._1)
                       (vr_partial pre_map_r (chest2_row vr (SZ.v bid)) nth)) **
    B.barrier_state (hreduce_barrier_count nth)
{
  unfold kpre_block pre_map pre_map_r rows cols nth x output sx vr sout shmem (SZ.v bid) (SZ.v tid);

  let (gsa, _) = shmem;
  let sa = from_array (l1_forward nth) gsa;
  rewrite each from_array (l1_forward nth) gsa as sa;

  (* Row of vr at bid, as a chest1 / lseq real cols. *)
  let vr_chest : chest1 real (SZ.v cols) = chest2_row (reveal vr) (SZ.v bid);
  let vr_row : erased (lseq real (SZ.v cols)) = hide (chest1_to_seq (reveal vr_chest));
  let vr_s : erased (lseq real nth) = vr_partial pre_map_r (reveal vr_chest) nth;

  (* Bridge from (sx %~ vr) to row-level approximation. *)
  assert pure (forall (j:nat). j < SZ.v cols ==>
                 (vr_row @! j) == acc2 (reveal vr) (SZ.v bid) j);
  assert pure (forall (j:nat). j < SZ.v cols ==>
                 acc2 sx (SZ.v bid) j %~ (vr_row @! j));

  (* Compute partial sum over stride and write to shmem. *)
  let psum : et = sum_stride_map_2d pre_map pre_map_r rows cols x bid nth tid vr_row;
  tensor_write_cell sa (tid, ()) psum;

  (* Set up tree reduction state. *)
  let mut n : szlt 32 = 0sz;

  (* psum approximates the [tid]-th partial sum [vr_s @! tid]. *)
  (**)assert pure (psum %~ rsum (seq_stride (lseq_map pre_map_r vr_row) nth (SZ.v tid)));
  (**)assert pure (reveal vr_s @! SZ.v tid
  (**)             == rsum (seq_stride (seq_map pre_map_r (chest1_to_seq (reveal vr_chest))) nth (SZ.v tid)));
  (**)assert pure (psum %~ (reveal vr_s @! SZ.v tid));
  (**)assert pure (Seq.equal (Seq.slice (reveal vr_s) (SZ.v tid) (SZ.v tid + 1))
  (**)                       (seq![reveal vr_s @! SZ.v tid]));
  (**)rsum_singleton_ (reveal vr_s @! SZ.v tid);
  (**)assert pure (rsum (Seq.slice (reveal vr_s) (SZ.v tid) (SZ.v tid + 1)) == (reveal vr_s @! SZ.v tid));

  forevery_singleton_intro'
    #(x:nat{tid <= x /\ x < tid + 1})
    (fun x -> tensor_pts_to_cell sa ((x <: natlt nth), ()) (seq![psum] @! (x - tid)))
    tid;
  fold array1_pts_to_slice sa tid (tid+1) seq![psum];

  (**)fold (array1_pts_to_slice_sum sa tid (tid + 1) vr_s);
  (**)assert pure (min (SZ.v tid + pow2 0) nth == SZ.v tid + 1);
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

  FStar.Math.Lemmas.modulo_lemma tid (pow2 it);
  rewrite
    (if_ (div_pow2 it tid) (array1_pts_to_slice_sum sa tid (min (tid + pow2 it) nth) vr_s))
  as
    (if_ (op_Equality #nat tid 0) (array1_pts_to_slice_sum sa 0 nth vr_s));

  log2_hreduce (v nth) it;
  rewrite (B.barrier_state it) as (B.barrier_state (hreduce_barrier_count nth));

  if (tid = 0sz) {
    if_elim_true' (op_Equality #nat tid 0) (array1_pts_to_slice_sum sa 0 nth vr_s);
    if_elim_true' (op_Equality #nat tid 0)
      (tensor_pts_to_cell output (abs_bij.gg (SZ.v bid <: natlt (SZ.v rows))) (acc1 sout (SZ.v bid)));
    unfold array1_pts_to_slice_sum sa 0 nth vr_s;
    (**)strided_sum_is_sum pre_map_r (reveal vr_chest) nth;
    (**)assert pure (Seq.equal (Seq.slice (reveal vr_s) 0 nth) (reveal vr_s));

    let res = array1_read_from_slice sa 0sz;
    tensor_write_cell output (bid, ()) res;

    with ss. assert array1_pts_to_slice sa 0 nth ss;
    unfold array1_pts_to_slice sa;
    let css : erased (chest1 et nth) = hide (seq_to_chest1 (reveal ss));
    (* Clean the index refinement [0<=k /\ k<nth] down to [k<nth] (= natlt nth),
       then reindex to the abstract tensor index and implode. *)
    forevery_refine_ext'
      #nat
      #(fun (k:nat) -> 0 <= k /\ k < nth)
      (fun (k:nat) -> k < nth)
      _;
    forevery_ext
      (fun (k:natlt nth) -> tensor_pts_to_cell sa ((k <: natlt nth), ()) (ss @! (k - 0)))
      (fun (k:natlt nth) -> tensor_pts_to_cell sa (abs_bij.gg k) (acc (reveal css) (abs_bij.gg k)));
    forevery_iso_back (abs_bij #nth)
      (fun (i : Kuiper.Shape.abs (nth @| INil)) -> tensor_pts_to_cell sa i (acc (reveal css) i));
    tensor_implode sa;
    rewrite each sa as from_array (l1_forward nth) shmem._1;
    if_intro_true' (op_Equality #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et).
        tensor_pts_to_cell output (abs_bij.gg (SZ.v bid <: natlt (SZ.v rows))) (v) **
        pure (v %~ chest1_rsum (chest_map pre_map_r (chest2_row vr (SZ.v bid))))
    );
    fold kpost_block pre_map pre_map_r rows cols nth x output sx vr sout shmem (SZ.v bid) (SZ.v tid);
  } else {
    if_elim_false' (op_Equality #nat tid 0) (array1_pts_to_slice_sum sa 0 nth vr_s);
    if_elim_false' (op_Equality #nat tid 0)
      (tensor_pts_to_cell output (abs_bij.gg (SZ.v bid <: natlt (SZ.v rows))) (acc1 sout (SZ.v bid)));
    if_intro_false' (op_Equality #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et).
        tensor_pts_to_cell output (abs_bij.gg (SZ.v bid <: natlt (SZ.v rows))) (v) **
        pure (v %~ chest1_rsum (chest_map pre_map_r (chest2_row vr (SZ.v bid))))
    );
    fold kpost_block pre_map pre_map_r rows cols nth x output sx vr sout shmem (SZ.v bid) (SZ.v tid);
    ()
  };
}
#pop-options

(* ── Block-level setup/teardown ────────────────────────────────────────── *)

ghost
fn block_setup_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   (SZ.v rows) (SZ.v cols))
  (vr   : chest2 real (SZ.v rows) (SZ.v cols) { sx %~ vr })
  (sout : chest1 et (SZ.v rows))
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt (SZ.v rows))
  ()
  norewrite
  requires
    live_c_shmems shmem **
    (x |-> Frac (1.0R /. SZ.v rows) sx **
     tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) ((acc1 sout bid)))
  ensures
    (forall+ (i : natlt nth). kpre_block pre_map pre_map_r rows cols nth x output sx vr sout shmem bid i) **
    emp
{
  unfold_live_c_shmems_cons shmem #_;
  unfold_live_c_shmems_nil shmem._2 #_;
  let gsa = shmem._1; rewrite each fst shmem as gsa;
  unfold live_c_shmem gsa;

  with vgsa. assert gsa |-> vgsa;
  gpu_pts_to_ref gsa;

  (* share input fractional permission across nth threads *)
  tensor_share_n x nth;

  (* tid 0 gets the output cell *)
  forevery_if_intro #(natlt nth) 0
    (fun _ -> tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) ((acc1 sout bid)));
  forevery_ext
    (fun tid -> if_ (op_Equality #(natlt nth) tid 0)
       (tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) ((acc1 sout bid))))
    (fun tid -> if_ (op_Equality #nat tid 0)
       (tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) ((acc1 sout bid))));

  forevery_zip (fun _ -> x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx) _;

  (* View shmem array as a tensor and explode it into per-cell ownership. *)
  tensor_abs' (l1_forward nth) gsa;
  tensor_explode (from_array (l1_forward nth) gsa);
  forevery_iso abs_bij _;

  forevery_zip #(natlt nth)
    (fun tid -> x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
                if_ (op_Equality #nat tid 0)
                  (tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) ((acc1 sout bid))))
    _;

  forevery_map
    #(natlt nth)
    (fun tid ->
      (x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
       if_ (op_Equality #nat tid 0)
         (tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) ((acc1 sout bid)))) **
      tensor_pts_to_cell (from_array (l1_forward nth) gsa) (abs_bij.gg (tid <: natlt nth))
        (acc (from_seq (l1_forward nth) vgsa) (abs_bij.gg (tid <: natlt nth)))
    )
    (fun (tid : natlt nth) -> kpre_block pre_map pre_map_r rows cols nth x output sx vr sout shmem bid tid)
    fn tid {
      rewrite each gsa as shmem._1;
      ();
    };
  ()
}

ghost
fn block_teardown_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   (SZ.v rows) (SZ.v cols))
  (vr   : chest2 real (SZ.v rows) (SZ.v cols) { sx %~ vr })
  (sout : chest1 et (SZ.v rows))
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt (SZ.v rows))
  ()
  norewrite
  requires
    (forall+ (i : natlt nth). kpost_block pre_map pre_map_r rows cols nth x output sx vr sout shmem bid i) **
    emp
  ensures
    live_c_shmems shmem **
    (x |-> Frac (1.0R /. SZ.v rows) sx **
     exists* (v : et).
       tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) (v) **
       pure (v %~ chest1_rsum (chest_map pre_map_r (chest2_row vr bid))))
{
  forevery_unzip _ _;

  tensor_gather_n x nth;

  forevery_ext #(natlt nth)
    (fun tid ->
      if_ (op_Equality #nat tid 0) (
        live (from_array (l1_forward nth) shmem._1) **
        exists* (v : et).
          tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) (v) **
          pure (v %~ chest1_rsum (chest_map pre_map_r (chest2_row vr bid)))))
    (fun tid ->
      if_ (op_Equality #(natlt nth) tid 0) (
        live (from_array (l1_forward nth) shmem._1) **
        exists* (v : et).
          tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) (v) **
          pure (v %~ chest1_rsum (chest_map pre_map_r (chest2_row vr bid)))));

  forevery_if_elim #(natlt nth) 0 (fun tid ->
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et).
        tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) (v) **
        pure (v %~ chest1_rsum (chest_map pre_map_r (chest2_row vr bid)))
  );

  tensor_concr (from_array (l1_forward nth) shmem._1);
  rewrite each core (from_array (l1_forward nth) shmem._1) as shmem._1;

  fold_live_c_shmems_nil shmem._2 #_;
  with vgsa. assert shmem._1 |-> vgsa;
  fold_live_c_shmem shmem._1;
  fold_live_c_shmems_cons shmem #_;
}

(* ── Outer setup/teardown: share x across blocks, explode output ─────── *)

ghost
fn setup_block_outer
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)             {| ctlayout lout |}
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   (SZ.v rows) (SZ.v cols))
  (vr   : chest2 real (SZ.v rows) (SZ.v cols) { sx %~ vr })
  (sout : chest1 et (SZ.v rows))
  ()
  norewrite
  requires
    x |-> sx ** output |-> sout
  ensures
    (forall+ (bid : natlt (SZ.v rows)).
       x |-> Frac (1.0R /. SZ.v rows) sx **
       tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) (acc1 sout bid)) **
    pure (SZ.fits (tlayout_ulen lout))
{
  tensor_pts_to_ref output;
  tensor_share_n x (SZ.v rows);
  tensor_explode output;
  forevery_iso abs_bij _;
  forevery_ext
    (fun (bid : natlt (SZ.v rows)) ->
       tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) (acc sout (abs_bij.gg (bid <: natlt (SZ.v rows)))))
    (fun (bid : natlt (SZ.v rows)) ->
       tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) (acc1 sout bid));

  forevery_zip #(natlt (SZ.v rows))
    (fun (_ : natlt (SZ.v rows)) -> x |-> Frac (1.0R /. SZ.v rows) sx)
    (fun (bid : natlt (SZ.v rows)) ->
       tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) (acc1 sout bid));
  ()
}

#push-options "--z3rlimit 40 --fuel 4 --ifuel 4"
ghost
fn teardown_block_outer
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)             {| ctlayout lout |}
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   (SZ.v rows) (SZ.v cols))
  (vr   : chest2 real (SZ.v rows) (SZ.v cols) { sx %~ vr })
  (sout : chest1 et (SZ.v rows))
  ()
  norewrite
  requires
    (forall+ (bid : natlt (SZ.v rows)).
       x |-> Frac (1.0R /. SZ.v rows) sx **
       exists* (v : et).
         tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) (v) **
         pure (v %~ chest1_rsum (chest_map pre_map_r (chest2_row vr bid)))) **
    pure (SZ.fits (tlayout_ulen lout))
  ensures
    exists* (sout' : chest1 et (SZ.v rows)).
      x |-> sx ** output |-> sout' **
      pure (forall (r : nat). r < SZ.v rows ==>
            (acc1 sout' r) %~ chest1_rsum (chest_map pre_map_r (chest2_row vr r)))
{
  forevery_unzip
    (fun (_ : natlt (SZ.v rows)) -> x |-> Frac (1.0R /. SZ.v rows) sx)
    (fun (bid : natlt (SZ.v rows)) ->
       exists* (v : et).
         tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) (v) **
         pure (v %~ chest1_rsum (chest_map pre_map_r (chest2_row vr bid))));

  tensor_gather_n x (SZ.v rows);

  (* Skolemize the existential: get a function bid -> et naming each cell value *)
  let f =
    forevery_exists
      (fun (bid : natlt (SZ.v rows)) (v : et) ->
         tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) (v) **
         pure (v %~ chest1_rsum (chest_map pre_map_r (chest2_row vr bid))));

  (* Build a concrete chest carrying the cell values. *)
  let sout' : erased (chest1 et (SZ.v rows)) =
    hide (mk1 (fun (bid : natlt (SZ.v rows)) -> f bid));

  (* Extract the per-row pure approximation fact across all bids. *)
  forevery_extract_pure
    (fun (bid : natlt (SZ.v rows)) ->
       tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) (f bid) **
       pure (f bid %~ chest1_rsum (chest_map pre_map_r (chest2_row vr bid))))
    (fun (bid : natlt (SZ.v rows)) ->
       (acc1 (reveal sout') bid) %~ chest1_rsum (chest_map pre_map_r (chest2_row vr bid)))
    fn _ {};

  (* Drop the per-cell pure now that we extracted the global fact. *)
  forevery_drop_pure
    (fun (bid : natlt (SZ.v rows)) ->
       tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) (f bid))
    (fun (bid : natlt (SZ.v rows)) ->
       f bid %~ chest1_rsum (chest_map pre_map_r (chest2_row vr bid)));

  forevery_ext
    (fun (bid : natlt (SZ.v rows)) ->
       tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) (f bid))
    (fun (bid : natlt (SZ.v rows)) ->
       tensor_pts_to_cell output (abs_bij.gg (bid <: natlt (SZ.v rows))) (acc (reveal sout') (abs_bij.gg (bid <: natlt (SZ.v rows)))));

  forevery_iso_back (abs_bij #(SZ.v rows))
    (fun (i : Kuiper.Shape.abs (rows @| INil)) -> tensor_pts_to_cell output i (acc (reveal sout') i));
  tensor_implode output;
  ()
}
#pop-options

(* ── Kernel descriptor ─────────────────────────────────────────────────── *)

inline_for_extraction noextract
let kdesc_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)             {| ctlayout lout |}
  (x      : array2 et lin  { is_global x      })
  (output : array1 et lout { is_global output })
  (sx   : chest2 et   (SZ.v rows) (SZ.v cols))
  (vr   : chest2 real (SZ.v rows) (SZ.v cols) { sx %~ vr })
  (sout : chest1 et (SZ.v rows))
  : kernel_desc
      (x |-> sx ** output |-> sout)
      (exists* (sout' : chest1 et (SZ.v rows)).
         x |-> sx ** output |-> sout' **
         pure (forall (r : nat). r < SZ.v rows ==>
               (acc1 sout' r) %~ chest1_rsum (chest_map pre_map_r (chest2_row vr r))))
  = {
    nblk = rows;
    nthr = nth;

    shmems_desc = [SHArray et nth];

    barrier_contract = (fun bid shmem ->
      mbarrier_contract (barrier_matrix #et nth (from_array _ shmem._1)
                          (vr_partial pre_map_r (chest2_row vr bid) nth)));
    barrier_count    = (fun _bid    -> hreduce_barrier_count nth);
    barrier_ok       = (fun bid shmem ->
      mbarrier_transform (barrier_matrix nth #(l1_forward nth)
                          (from_array _ shmem._1)
                          (vr_partial pre_map_r (chest2_row vr bid) nth)));

    f = kf_block pre_map pre_map_r rows cols nth x output sx vr sout;

    block_pre  = (fun bid ->
      x |-> Frac (1.0R /. SZ.v rows) sx **
      tensor_pts_to_cell (output <: array1 et lout) (abs_bij.gg (bid <: natlt (SZ.v rows))) ((acc1 sout bid)));
    block_post = (fun bid ->
      x |-> Frac (1.0R /. SZ.v rows) sx **
      exists* (v : et).
        tensor_pts_to_cell (output <: array1 et lout) (abs_bij.gg (bid <: natlt (SZ.v rows))) (v) **
        pure (v %~ chest1_rsum (chest_map pre_map_r (chest2_row vr bid))));

    setup    = setup_block_outer    pre_map pre_map_r rows cols nth x output sx vr sout;
    teardown = teardown_block_outer pre_map pre_map_r rows cols nth x output sx vr sout;

    block_frame    = (fun _shmem _bid -> emp);
    block_setup    = block_setup_block    pre_map pre_map_r rows cols nth x output sx vr sout;
    block_teardown = block_teardown_block pre_map pre_map_r rows cols nth x output sx vr sout;

    kpre  = kpre_block  pre_map pre_map_r rows cols nth x output sx vr sout;
    kpost = kpost_block pre_map pre_map_r rows cols nth x output sx vr sout;
    frame = pure (SZ.fits (tlayout_ulen lout));

    kpre_sendable       = magic();
    kpost_sendable      = magic();
    block_post_sendable = solve;
    block_pre_sendable  = solve;
  }

(* ── Entry point ──────────────────────────────────────────────────────── *)

inline_for_extraction noextract
fn row_reduce
  (#et : Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth  : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)      {| ctlayout lout |}
  (x      : array2 et lin  { is_global x      })
  (output : array1 et lout { is_global output })
  (#sx   : chest2 et   rows cols)
  (vr    : chest2 real rows cols)
  (#sout : chest1 et rows)
  preserves
    cpu **
    on gpu_loc (x |-> sx)
  requires
    on gpu_loc (output |-> sout) **
    pure (sx %~ vr)
  ensures
    exists* (sout' : chest1 et rows).
      on gpu_loc (output |-> sout') **
      pure (forall (r : nat). r < SZ.v rows ==>
            (acc1 sout' r) %~ chest1_rsum (chest_map pre_map_r (chest2_row vr r)))
{
  launch_sync (kdesc_block pre_map pre_map_r rows cols nth x output sx vr sout);
}
