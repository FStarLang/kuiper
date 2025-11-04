module Kuiper.Poly.HReduce

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
module RPM = Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Seq.Common

module SZ = Kuiper.SizeT
module U32 = FStar.UInt32

[@@CPrologue "__device__"]
noextract inline_for_extraction
let spow2 (s : sz{s < 32}) : r:sz{SZ.v r == pow2 s} =
  (* Computing 2^s by 1<<s *)
  SZ.uint32_to_sizet (U32.shift_left 1ul (sizet_to_u32 s))

[@@CPrologue "__device__"]
noextract inline_for_extraction
let sdiv_pow2 (i:sz{i < 32}) (tid: sz) : bool =
  // SZ.rem tid (spow2 i) = 0sz
  sizet_and tid SZ.(spow2 i -^ 1sz) = 0sz

let sdiv_pow2_ok (i:sz{i < 32}) (tid:sz) :
  Lemma (sdiv_pow2 i tid == div_pow2 i tid)
        [SMTPat (sdiv_pow2 i tid)]
= sizet_and_div_pow2 tid (spow2 i) i;
  calc (==) {
    SZ.v (SZ.rem tid (spow2 i));
    == {}
    SZ.v tid - ((SZ.v tid / SZ.v (spow2 i)) * SZ.v (spow2 i));
    == { FStar.Math.Lemmas.euclidean_division_definition (SZ.v tid) (SZ.v (spow2 i)) }
    SZ.v tid % SZ.v (spow2 i);
    == {}
    SZ.v tid % pow2 (SZ.v i);
}

[@@CPrologue "__device__"]
noextract inline_for_extraction
let smin (a b : sz): sz =
  let open FStar.SizeT in
  if a <^ b then a else b

(* Ownership of array r between i and j. The first value of that slice
is the reduction of all the values in the (original) slice v. *)
unfold
let gpu_pts_to_slice_sum_inner
  (#et:Type0) {| scalar et, real_like et |}
  (#sz:nat)
  (r : gpu_array et sz)
  (i j :nat)
  (v : seq et)
  (rr : seq Real.real { v %~ rr })
  (s : seq et)
: slprop
= gpu_pts_to_slice r i j s
  ** pure (i < j /\ j <= sz /\
           len v = sz /\
           len s = j - i /\
           squash ((s @! 0) `approximates` real_seq_sum (Seq.slice rr i j))) // SQUASH VERY IMPORTANT!!

(* Not easy to mark this unfold as it has a lambda (in the exists) *)
let gpu_pts_to_slice_sum
  (#et:Type0) {| scalar et, real_like et |}
  (#sz:nat)
  ([@@@mkey] r: gpu_array et sz)
  ([@@@mkey] i : nat)
  (j:nat)
  (v: seq et)
  (rr : seq Real.real { v %~ rr })
: slprop
= exists* s. gpu_pts_to_slice_sum_inner r i j v rr s

// Barrier

unfold
let barrier_matrix
  (#et:Type0) {| scalar et, real_like et |}
  (nth : nat) (r : gpu_array et nth)
  (v : seq et)
  (vr : seq Real.real { v %~ vr })
  (it from to : nat)
: slprop
=
  if_ (from = to + pow2 it)
      (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
           (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) v vr))

ghost
fn mk_barrier_pre
  (#et:Type0) {| scalar et, real_like et |}
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= max_threads })
  (r : gpu_array et nth)
  (vv : seq et)
  (vr : seq Real.real { vv %~ vr })
  (tid : sz{SZ.v tid < nth})
  (it: sz{it < 31})
  requires if_ (not (div_pow2 (it + 1) tid) && div_pow2 it tid)
      (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv vr)
  ensures forall+ (i:natlt nth). barrier_matrix nth r vv vr it tid i
{
  open FStar.SizeT;
  if (tid >=^ spow2 it) {
    forevery_if_intro #(natlt nth) (tid - pow2 it) (fun i ->
      if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid))
        (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv vr));
    forevery_ext
      (fun (i:natlt nth) ->
        if_ (op_Equality #(natlt (v nth)) i (tid - pow2 it))
          (if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid))
            (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv vr)))
      (fun (i:natlt nth) -> barrier_matrix nth r vv vr it tid i);
  } else {
    FStar.Math.Lemmas.modulo_lemma tid (spow2 it);
    FStar.Math.Lemmas.modulo_lemma 0 (spow2 (it +^ 1sz));
    // assert (pure ((tid <: nat) <> 0 ==> not (div_pow2 it tid)));
    // assert (pure ((tid <: nat) = 0 ==> (div_pow2 (it + 1) tid)));
    if_rewrite_bool (op_Negation (div_pow2 (SZ.v it + 1) (SZ.v tid)) && div_pow2 (SZ.v it) (SZ.v tid)) false _;
    if_elim_false _;

    forevery_emp_intro (natlt nth);
    forevery_ext
      (fun (i:natlt nth) -> emp)
      (fun (i:natlt nth) -> barrier_matrix nth r vv vr it tid i);
  }
}


unfold
let kpre
  (#et:Type0) {| scalar et, real_like et |}
  (lena : nat)
  (a : gpu_array et lena)
  (s : seq et)
  (vr : seq Real.real { s %~ vr })
  (#_: squash (len s == lena))
  (tid : natlt lena)
  : slprop =
    gpu_pts_to_slice a tid (tid +1) seq![s @! tid] **
    mbarrier_tok lena (barrier_matrix lena a s vr) 0 tid

unfold
let kpost
  (#et:Type0) {| scalar et, real_like et |}
  (lena : nat)
  (a : gpu_array et lena)
  (s : seq et)
  (vr : seq Real.real { s %~ vr })
  (#_: squash (len s == lena))
  (tid : natlt lena)
  : slprop =
    if_ (tid = 0) (gpu_pts_to_slice_sum a 0 lena s vr) **
    (exists* it. mbarrier_tok lena (barrier_matrix lena a s vr) it tid)

// #push-options "--print_implicits"
inline_for_extraction
fn iteration
  (#et:Type0) {| scalar et, real_like et |}
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= max_threads })
  (r : gpu_array et nth)
  (vv : erased (seq et))
  (vr : erased (seq Real.real) { reveal vv %~ reveal vr })
  (#_: squash (len vv == nth))
  (tid : sz{SZ.v tid < nth})
  (it: sz{it < 31})
  requires gpu
    ** mbarrier_tok nth (barrier_matrix nth r vv vr) it tid
    ** if_ (div_pow2 it tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv vr)
  ensures gpu
    ** mbarrier_tok nth (barrier_matrix nth r vv vr) (it+1) tid
    ** if_ (div_pow2 (it+1) tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 (it + 1)) nth) vv vr)
{
  assert (pure (len vv = nth));

  case_split (div_pow2 (it + 1) tid)
    (if_ (div_pow2 it tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv vr));
  if_flatten #(div_pow2 (it + 1) tid);
  if_flatten #(not (div_pow2 (it + 1) tid));

  div_pow2_lemma it (it + 1) tid;
  rewrite (if_ (div_pow2 (it + 1) tid && div_pow2 it tid)
            (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv vr))
      as (if_ (div_pow2 (it + 1) tid)
            (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv vr));

  mk_barrier_pre nth r vv vr tid it;
  fold RPM.row #nth (barrier_matrix nth r vv vr) it tid;
  mbarrier_wait ();
  unfold RPM.col #nth (barrier_matrix nth r vv vr) it tid;

  // combine (div_pow2 (it + 1) tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv) _;

  let nextid = FStar.SizeT.(tid +^ spow2 it);

  (* We do not use end_ in extracted code, so we can use a nat and erase it
  so there are no traces in the extracted C. *)
  let end_ : erased nat = hide (min (tid + 2 * pow2 it) nth);

  if (FStar.SizeT.(nextid <^ nth)) {
    forevery_ext
      (fun (from: natlt nth) ->
        if_ (op_Equality #int from (tid + pow2 it))
          (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
            (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) vv vr)))
      (fun (from: natlt nth) ->
        if_ (op_Equality #(natlt nth) from (tid + pow2 it))
          (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
            (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) vv vr)));
    forevery_if_elim #(natlt nth)
      (tid + pow2 it)
      (fun (from: nat) -> if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
         (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) vv vr));

    let b = sdiv_pow2 (FStar.SizeT.(it +^ 1sz)) tid;

    rewrite each (div_pow2 (SZ.v it + 1) (SZ.v tid)) as b;

    div_pow2_lemma_2 it tid;
    combine
      b
      (gpu_pts_to_slice_sum r nextid (min (tid + pow2 it + pow2 it) nth) vv vr)
      _;

    if b {
      assert (pure (div_pow2 (SZ.v it + 1) (SZ.v tid)));
      if_elim_true _;

      (**)unfold (gpu_pts_to_slice_sum r nextid end_ vv vr);
      (**)unfold (gpu_pts_to_slice_sum r tid nextid vv vr);
      (**)gpu_slice_concat #et #(SZ.v nth) r tid nextid end_;

      let s1 = gpu_array_read r tid;
      (**)assert (pure (s1 `approximates` real_seq_sum (Seq.slice vr tid nextid)));

      let s2 = gpu_array_read r nextid;
      (**)assert (pure (s2 `approximates` real_seq_sum (Seq.slice vr nextid end_)));

      let s = add s1 s2;
      (**)lem_append_slice vr tid nextid end_;
      (**)seq_approximates_append s1 s2 (Seq.slice vr tid nextid) (Seq.slice vr nextid end_);
      (**)assert (pure ((s1 `add` s2) `approximates` real_seq_sum (Seq.append (Seq.slice vr tid nextid) (Seq.slice vr nextid end_))));
      (**)real_seq_sum_append (Seq.slice vr tid nextid) (Seq.slice vr nextid end_);
      (**)assert (pure (s `approximates` real_seq_sum (Seq.slice vr tid end_)));

      gpu_array_write r tid s;

      (**)with seq. assert (gpu_pts_to_slice r tid end_ seq);
      (**)fold (gpu_pts_to_slice_sum r tid end_ vv vr);
      (**)if_intro_true (gpu_pts_to_slice_sum r tid end_ vv vr);
      // Step below optional right now, but good practice?
      (**)rewrite
      (**)  if_ true
      (**)      (gpu_pts_to_slice_sum r (SZ.v tid) (reveal end_) (reveal vv) vr)
      (**)as
      (**)  if_ (div_pow2 (SZ.v it + 1) (SZ.v tid))
      (**)      (gpu_pts_to_slice_sum r (SZ.v tid) (reveal end_) (reveal vv) vr);
    } else {
      (* no-op *)
      if_elim_false _;
      if_intro_false (gpu_pts_to_slice_sum r tid end_ vv vr);
    }
  } else {
    forevery_map
      (fun (from: natlt nth) ->
        if_ (op_Equality #int from (tid + pow2 it))
          (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
            (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) vv vr)))
      (fun from -> emp)
      fn from {
        if_rewrite_bool (from = tid + pow2 it) false _;
        if_elim_false _;
      };
    forevery_emp_elim _;
  }
}

inline_for_extraction noextract
fn kf
  (#et:Type0) {| scalar et, real_like et |}
  (nth : szp { nth <= max_threads })
  (a : gpu_array et nth)
  (#s : erased (seq et))
  (#vr : erased (seq real){ reveal s %~ reveal vr })
  (#_ : squash (Seq.length s == nth))
  (tid : szlt nth)
  ()
  requires
    gpu **
    kpre nth a s vr tid **
    thread_id nth tid
  ensures
    gpu **
    kpost nth a s vr tid **
    thread_id nth tid
{
  (* Reduction *)
  let mut n : szlt 32 = 0sz;

  (**)with ss. assert (gpu_pts_to_slice a tid (tid+1) ss);
  assert (pure (Seq.slice s tid (tid+1) `Seq.equal` seq![ss @! 0])); // sucks
  // (**)if_intro_true (exists* ss. gpu_pts_to_slice_sum_inner a tid (tid + 1) s ss);
  (**)fold (gpu_pts_to_slice_sum a tid (tid + 1) s vr);
  (**)if_intro_true (gpu_pts_to_slice_sum a tid (tid + 1) s vr);

  open FStar.SizeT;
  while ((spow2 !n <^ nth))
    invariant
      exists* (it : szlt 32).
        n |-> it **
        mbarrier_tok nth (barrier_matrix nth a s vr) it tid **
        if_ (div_pow2 it tid) (gpu_pts_to_slice_sum a tid (min (tid + pow2 it) nth) s vr)
  {
    iteration nth a s vr tid !n;
    n := !n +^ 1sz;
  };
}

ghost
fn factor_array
  (#et:Type0)
  (len : pos)
  (a : gpu_array et len)
  (d1 d2 : nat)
  (#va : seq et { Seq.length va == len /\ len == d1 * d2})
  requires
    a |-> va
  ensures
    forall+ (i1:natlt d1) (i2:natlt d2).
      gpu_pts_to_slice a (i1 * d2 + i2) (i1 * d2 + i2 + 1) seq![va @! (i1 * d2 + i2)]
{
  Kuiper.Array.gpu_array_slice_1 a;
  forevery_factor len d1 d2 _;
}

ghost
fn unfactor_array
  (#et:Type0)
  (len : pos)
  (a : gpu_array et len)
  (d1 d2 : nat)
  (#va : seq et { Seq.length va == len /\ len == d1 * d2})
  requires
    forall+ (i1:natlt d1) (i2:natlt d2).
      gpu_pts_to_slice a (i1 * d2 + i2) (i1 * d2 + i2 + 1) seq![va @! (i1 * d2 + i2)]
  ensures
    a |-> va
{
  forevery_unfactor len d1 d2 (fun i -> gpu_pts_to_slice a i (i+1) seq![va @! i]);
  Kuiper.Array.gpu_array_unslice_1 a;
}

ghost
fn block_setup
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp { lena < max_threads })
  (a : gpu_array et lena)
  (#va : seq et)
  (#vr : seq real { va %~ vr })
  (#_ : squash (Seq.length va == SZ.v lena))
  ()
  norewrite
  requires
    can_create_barrier lena **
    a |-> va
  ensures
    consumed_can_create_barrier **
    (forall+ (i : natlt lena). kpre lena a va vr i) **
    emp
{
  gpu_array_slice_1 a;
  mk_mbarrier lena (barrier_matrix lena a va vr);
  forevery_zip
      _
      (RPM.mbarrier_tok (sizet_to_nat lena) (barrier_matrix (sizet_to_nat lena) a va vr) 0)
    ;
}

ghost
fn block_teardown
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp { lena < max_threads })
  (a : gpu_array et lena)
  (#va : seq et)
  (#vr : seq real { va %~ vr })
  (#_ : squash (Seq.length va == SZ.v lena))
  ()
  norewrite
  requires
    (forall+ (i : natlt lena). kpost lena a va vr i) **
    emp
  ensures
    gpu_pts_to_slice_sum a 0 lena va vr
{
  forevery_map
    (fun (j:natlt lena) ->
      if_ (j = 0)
        (gpu_pts_to_slice_sum a 0 (SZ.v lena) va vr) **
      (exists* (it: nat).
          RPM.mbarrier_tok (SZ.v lena)
            (barrier_matrix (SZ.v lena) a va vr)
            it
            j))
    (fun (j:natlt lena) ->
      if_ (op_Equality #(natlt lena) j 0)
        (gpu_pts_to_slice_sum a 0 (SZ.v lena) va vr))
    fn j {
      with a b c d. assert (RPM.mbarrier_tok a b c d);
      drop_ (RPM.mbarrier_tok a b c d);
    };
  forevery_if_elim #(natlt lena) 0 (fun (x: natlt lena) ->
    gpu_pts_to_slice_sum a 0 (v lena) va vr);
}

inline_for_extraction noextract
let kernel
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp { lena < max_threads })
  (a : gpu_array et lena)
  (#va : erased (seq et) { Seq.length va == SZ.v lena })
  (#vr : erased (seq real) { reveal va %~ reveal vr })
: kernel_desc_1_n
    (a |-> va)
    (gpu_pts_to_slice_sum a 0 lena va vr)
= {
  nthr = lena;
  f = kf lena a #va #vr;

  block_setup = block_setup lena a #va;
  block_teardown = block_teardown lena a #va;
  kpost = kpost lena a va vr;
  kpre =  kpre lena a va vr;
  frame = emp;
}

inline_for_extraction noextract
fn reduce
  (#et:Type0) {| scalar et, real_like et |}
  (lena : szp { lena < max_threads })
  (a : gpu_array et lena)
  (#va : erased (seq et))
  (#vr : erased (seq real) { reveal va %~ reveal vr })
  requires
    cpu **
    (a |-> va)
  ensures
    cpu **
    (exists* (va' : seq et{Seq.length va' > 0}).
      gpu_pts_to_array a va' **
      pure ((va' @! 0) `approximates` seq_fold_left (+.) 0.0R vr))
{
  gpu_pts_to_ref a; (* recall length, automate *)
  launch_sync (kernel lena a #va #vr);
  unfold gpu_pts_to_slice_sum a 0 lena va vr;
  ()
}
