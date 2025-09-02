module Kuiper.Poly.HReduce

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
module RPM = Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Seq.Common
open Kuiper.IsReduction

module SZ = FStar.SizeT
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

// Barrier

let barrier_matrix
  (#et:Type0) {| scalar et |}
  (nth : nat) (r : gpu_array et nth)
  (v : seq et)
  (it from to : nat)
: slprop
=
  if_ (from = to + pow2 it)
      (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
           (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) v))

ghost
fn mk_barrier_pre
  (#et:Type0) {| scalar et |}
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= max_threads })
  (r : gpu_array et nth)
  (vv: erased (seq et))
  (#_: squash (len vv == nth))
  (tid : sz{SZ.v tid < nth})
  (it: sz{it < 31})
  requires if_ (not (div_pow2 (it + 1) tid) && div_pow2 it tid)
      (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv)
  ensures bigstar 0 nth (barrier_matrix nth r vv it tid)
{
  open FStar.SizeT;
  if (tid >=^ spow2 it) {
    bigstar_if_intro 0 nth (tid - pow2 it) (fun _ -> if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid)) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv));
  } else {
    FStar.Math.Lemmas.modulo_lemma tid (spow2 it);
    FStar.Math.Lemmas.modulo_lemma 0 (spow2 (it +^ 1sz));
    // assert (pure ((tid <: nat) <> 0 ==> not (div_pow2 it tid)));
    // assert (pure ((tid <: nat) = 0 ==> (div_pow2 (it + 1) tid)));
    if_rewrite_bool (op_Negation (div_pow2 (SZ.v it + 1) (SZ.v tid)) && div_pow2 (SZ.v it) (SZ.v tid)) false _;
    if_elim_false _;

    bigstar_emp_intro 0 nth;
  }
}


unfold
let kpre
  (#et:Type0) {| scalar et |}
  (lena : nat)
  (a : gpu_array et lena)
  (s : erased (seq et))
  (#_: squash (len s == lena))
  (tid : natlt lena)
  : slprop =
    gpu_pts_to_slice a tid (tid +1) seq![s @! tid] **
    mbarrier_tok lena (barrier_matrix lena a s) 0 tid

unfold
let kpost
  (#et:Type0) {| scalar et |}
  (lena : nat)
  (a : gpu_array et lena)
  (s : erased (seq et))
  (#_: squash (len s == lena))
  (tid : natlt lena)
  : slprop =
    if_ (tid = 0) (gpu_pts_to_slice_sum a 0 lena s) **
    (exists* it. mbarrier_tok lena (barrier_matrix lena a s) it tid)

inline_for_extraction
fn iteration
  (#et:Type0) {| scalar et |}
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= max_threads })
  (r : gpu_array et nth)
  (vv: erased (seq et))
  (#_: squash (len vv == nth))
  (tid : sz{SZ.v tid < nth})
  (it: sz{it < 31})
  requires gpu
    ** mbarrier_tok nth (barrier_matrix nth r vv) it tid
    ** if_ (div_pow2 it tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv)
  ensures gpu
    ** mbarrier_tok nth (barrier_matrix nth r vv) (it+1) tid
    ** if_ (div_pow2 (it+1) tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 (it + 1)) nth) vv)
{
  assert (pure (len vv = nth));

  case_split (div_pow2 (it + 1) tid) (if_ (div_pow2 it tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv));
  if_flatten #(div_pow2 (it + 1) tid);
  if_flatten #(not (div_pow2 (it + 1) tid));

  div_pow2_lemma it (it + 1) tid;
  rewrite (if_ (div_pow2 (it + 1) tid && div_pow2 it tid)
            (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv))
      as (if_ (div_pow2 (it + 1) tid)
            (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv));

  mk_barrier_pre nth r vv tid it;
  fold RPM.row #nth (barrier_matrix nth r vv) it tid;
  mbarrier_wait ();
  unfold RPM.col #nth (barrier_matrix nth r vv) it tid;

  // combine (div_pow2 (it + 1) tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv) _;

  let nextid = FStar.SizeT.(tid +^ spow2 it);

  (* We do not use end_ in extracted code, so we can use a nat and erase it
  so there are no traces in the extracted C. *)
  let end_   : erased nat = hide (min (tid + 2 * pow2 it) nth);

  if (FStar.SizeT.(nextid <^ nth)) {
    bigstar_if_elim #_ #0
      #nth (tid + pow2 it)
      (fun (from: nat) -> if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from)) (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) vv));

    let b = sdiv_pow2 (FStar.SizeT.(it +^ 1sz)) tid;

    rewrite each (div_pow2 (SZ.v it + 1) (SZ.v tid)) as b;

    div_pow2_lemma_2 it tid;
    combine
      b
      (gpu_pts_to_slice_sum r nextid (min (tid + pow2 it + pow2 it) nth) vv)
      _;

    if b {
      assert (pure (div_pow2 (SZ.v it + 1) (SZ.v tid)));
      if_elim_true _;

      (**)unfold (gpu_pts_to_slice_sum #et r tid nextid vv);
      (**)unfold (gpu_pts_to_slice_sum r nextid end_ vv);
      (**)gpu_slice_concat #et #(SZ.v nth) r tid nextid end_;

      let s1 = gpu_array_read r tid;
      (**)assert (pure (squash (is_reduction zero add (Seq.slice vv tid nextid) s1)));

      let s2 = gpu_array_read r nextid;
      (**)assert (pure (squash (is_reduction zero add (Seq.slice vv nextid end_) s2)));

      let s = add s1 s2;
      (**)lem_append_slice vv tid nextid end_;
      (**)assert (pure (squash (is_reduction zero add (Seq.slice vv tid end_) s)));

      gpu_array_write r tid s;

      (**)with seq. assert (gpu_pts_to_slice r tid end_ seq);
      (**)fold (gpu_pts_to_slice_sum r tid end_ vv);
      (**)if_intro_true (gpu_pts_to_slice_sum r tid end_ vv);
      // Step below optional right now, but good practice?
      (**)rewrite
      (**)  if_ true
      (**)      (gpu_pts_to_slice_sum r (SZ.v tid) (reveal end_) (reveal vv))
      (**)as
      (**)  if_ (div_pow2 (SZ.v it + 1) (SZ.v tid))
      (**)      (gpu_pts_to_slice_sum r (SZ.v tid) (reveal end_) (reveal vv));
    } else {
      (* no-op *)
      if_elim_false _;
      if_intro_false (gpu_pts_to_slice_sum r tid end_ vv);
    }
  } else {
    bigstar_map #_ #_ #0 #nth #(fun (from:nat { 0 <= from /\ from < nth }) -> _ from)
      (fun (from: nat{0 <= from /\ from < nth}) ->
        if_rewrite_bool (from = tid + pow2 it) false _);
    bigstar_map #_ #_ #0 #nth #(fun (from:nat { 0 <= from /\ from < nth }) -> _ from)
      (fun (from: nat{0 <= from /\ from < nth}) ->
        if_elim_false (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from)) (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) vv)));
    bigstar_emp_elim #_;
  }
}

inline_for_extraction noextract
fn kf
  (#et:Type0) {| scalar et |}
  (nth : szp { nth <= max_threads })
  (a : gpu_array et nth)
  (#s : erased (seq et))
  (#_ : squash (Seq.length s == nth))
  (tid : szlt nth)
  ()
  requires
    gpu **
    kpre nth a s tid **
    thread_id nth tid
  ensures
    gpu **
    kpost nth a s tid **
    thread_id nth tid
{
  (* Reduction *)
  let mut n : szlt 32 = 0sz;

  (**)with ss. assert (gpu_pts_to_slice a tid (tid+1) ss);
  assert (pure (Seq.slice s tid (tid+1) `Seq.equal` seq![ss @! 0])); // sucks
  // (**)if_intro_true (exists* ss. gpu_pts_to_slice_sum_inner a tid (tid + 1) s ss);
  (**)fold (gpu_pts_to_slice_sum a tid (tid + 1) s);
  (**)if_intro_true (gpu_pts_to_slice_sum a tid (tid + 1) s);

  open FStar.SizeT;
  while ((spow2 !n <^ nth))
    invariant
      exists* (it : szlt 32).
        n |-> it **
        mbarrier_tok nth (barrier_matrix nth a s) it tid **
        if_ (div_pow2 it tid) (gpu_pts_to_slice_sum a tid (min (tid + pow2 it) nth) s)
  {
    iteration nth a s tid !n;
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
  open Kuiper.Enumerable;
  Kuiper.Array.gpu_array_slice_1 a;
  rewrite each len as cardinal (natlt len) #_;
  forevery_fromstar #(natlt len) (fun i -> gpu_pts_to_slice a i (i+1) seq![va @! i]);
  forevery_factor len d1 d2 _;
  ();
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
  open Kuiper.Enumerable;
  forevery_unfactor len d1 d2 (fun i -> gpu_pts_to_slice a i (i+1) seq![va @! i]);
  forevery_tostar #(natlt len) _;
  rewrite each (cardinal (natlt len) #_) as len;
  Kuiper.Array.gpu_array_unslice_1 a;
}

ghost
fn block_setup
  (#et:Type0) {| scalar et |}
  (lena : szp { lena < max_threads })
  (a : gpu_array et lena)
  (#va : erased (seq et) { Seq.length va == SZ.v lena })
  ()
  norewrite
  requires
    block_setup_tok lena **
    a |-> va
  ensures
    block_setup_tok lena **
    (forall+ (i : natlt lena). kpre lena a va i) **
    emp
{
  open Kuiper.Enumerable;
  gpu_array_slice_1 a;
  mk_mbarrier lena (barrier_matrix lena a va);
  bigstar_zip 0 lena
      _
      (RPM.mbarrier_tok (sizet_to_nat lena) (barrier_matrix (sizet_to_nat lena) a va) 0)
    ;
  rewrite each (SZ.v lena) as cardinal (natlt lena) #_;
  forevery_fromstar #(natlt lena) (fun i -> kpre lena a va i);
}

ghost
fn block_teardown
  (#et:Type0) {| scalar et |}
  (lena : szp { lena < max_threads })
  (a : gpu_array et lena)
  (#va : erased (seq et) { Seq.length va == SZ.v lena })
  ()
  norewrite
  requires
    (forall+ (i : natlt lena). kpost lena a va i) **
    emp
  ensures
    gpu_pts_to_slice_sum a 0 lena va
{
  open Kuiper.Enumerable;
  forevery_tostar #(natlt lena) _;
  rewrite each
    (Kuiper.Enumerable.cardinal (natlt (SZ.v lena))
            #(Kuiper.Enumerable.enumerable_natlt (SZ.v lena)))
    as lena;
  ghost
  fn mapper (j: nat{b2t (0 <= j) /\ b2t (j < SZ.v lena)})
    norewrite
    requires
      if_ (op_Equality #int (Kuiper.Enumerable.of_nat #(natlt lena) j) 0)
        (gpu_pts_to_slice_sum
            a
            0
            (SZ.v lena)
            (reveal #(seq et) va)) **
      (exists* (it: nat).
          RPM.mbarrier_tok (SZ.v lena)
            (barrier_matrix (SZ.v lena) a (reveal #(seq et) va))
            it
            (Kuiper.Enumerable.of_nat #(natlt lena) j))
    ensures
      if_ (op_Equality #int j 0)
        (gpu_pts_to_slice_sum
            a
            0
            (SZ.v lena)
            (reveal #(seq et) va))
  {
    with a b c d. assert (RPM.mbarrier_tok a b c d);
    drop_ (RPM.mbarrier_tok a b c d);
    ();
  };
  bigstar_map #_ #_ #0 #lena mapper;
  bigstar_if_elim #0 #0 #(SZ.v lena) 0 _;
}

inline_for_extraction noextract
let kernel
  (#et:Type0) {| scalar et |}
  (lena : szp { lena < max_threads })
  (a : gpu_array et lena)
  (#va : erased (seq et) { Seq.length va == SZ.v lena })
: kernel_desc_1_n
    (a |-> va)
    (gpu_pts_to_slice_sum a 0 lena va)
= {
  nthr = lena;
  f = kf lena a #va;

  block_setup = block_setup lena a #va;
  block_teardown = block_teardown lena a #va;
  kpost = kpost lena a va;
  kpre =  kpre lena a va;
  frame = emp;
}

inline_for_extraction noextract
fn reduce
  (#et:Type0) {| scalar et |}
  (lena : szp { lena < max_threads })
  (a : gpu_array et lena)
  requires
    cpu **
    (a |-> 'va)
  ensures
    cpu **
    gpu_pts_to_slice_sum a 0 lena 'va
{
  gpu_pts_to_ref a; (* recall length, automate *)
  launch_sync (kernel lena a);
}
