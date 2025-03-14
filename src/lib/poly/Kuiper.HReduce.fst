module Kuiper.HReduce

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
let spow2 (s : sz{s < 32}) : r:sz{SZ.v r == pow2 (SZ.v s)} =
  (* Computing 2^s by 1<<s *)
  SZ.uint32_to_sizet (U32.shift_left 1ul (sizet_to_u32 s))

[@@CPrologue "__device__"]
noextract inline_for_extraction
let sdiv_pow2 (i:sz{i < 32}) (tid: sz) : bool =
  // SZ.rem tid (spow2 i) = 0sz
  sizet_and tid SZ.(spow2 i -^ 1sz) = 0sz

let sdiv_pow2_ok (i:sz{i < 32}) (tid:sz) :
  Lemma (sdiv_pow2 i tid <==> div_pow2 (SZ.v i) (SZ.v tid))
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
  (#et:Type0) {| scalar et |}
  (#sz:nat)
  (r : gpu_array et sz)
  (i j :nat)
  (v : seq et)
  (s : seq et)
: slprop
= gpu_pts_to_slice r i j s
  ** pure (i < j /\ j <= sz /\
           len v = sz /\
           len s = j - i /\
           squash (is_reduction zero add (Seq.slice v i j) (s @! 0))) // SQUASH VERY IMPORTANT!!

(* Not easy to mark this unfold as it has a lambda (in the exists) *)
let gpu_pts_to_slice_sum
  (#et:Type0) {| scalar et |}
  (#sz:nat)
  ([@@@mkey] r: gpu_array et sz)
  ([@@@mkey] i : nat)
  (j:nat)
  (v: seq et)
: slprop
= if_ (i < j && j <= sz) (exists* s. gpu_pts_to_slice_sum_inner r i j v s)

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
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
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
    gpu_pts_to_slice a tid (tid +1) seq![Seq.index s tid] **
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

// KrmlPrivate is essentially a "noextract". F* usually adds it
// automatically to any definition that does not appear in the fsti,
// but we have disable that since it interoperates poorly with pulse
// (due to splicing).
[@@CPrologue "__device__"; "KrmlPrivate"]
inline_for_extraction
fn iteration
  (#et:Type0) {| scalar et |}
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
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
      (**)if_elim_true (exists* s. gpu_pts_to_slice_sum_inner r tid nextid vv s);
      let s1 = gpu_array_read #et #_ #tid #nextid r tid;
      (**)assert (pure (squash (is_reduction #et Scalars.zero Scalars.add (Seq.slice vv tid nextid) s1)));

      (**)unfold (gpu_pts_to_slice_sum r nextid end_ vv);
      (**)if_elim_true (exists* s. gpu_pts_to_slice_sum_inner r nextid end_ vv s);
      let s2 = gpu_array_read #_ #_ #nextid #end_ r nextid;
      (**)assert (pure (squash (is_reduction zero add (Seq.slice vv nextid end_) s2)));

      let s = add s1 s2;
      (**)lem_append_slice vv tid nextid end_;
      (**)assert (pure (squash (is_reduction zero add (Seq.slice vv tid end_) s)));

      gpu_array_write #et #(SZ.v nth) #(SZ.v tid) #(SZ.v nextid) r tid s;

      (**)gpu_slice_concat #et #(SZ.v nth) r tid nextid end_;
      (**)with seq. assert (gpu_pts_to_slice r tid end_ seq);
      (**)if_intro_true (exists* s. gpu_pts_to_slice_sum_inner r tid end_ vv s);
      (**)fold (gpu_pts_to_slice_sum r tid end_ vv);
      (**)if_intro_true (gpu_pts_to_slice_sum r tid end_ vv);
      (**)rewrite
      (**)  if_ true (gpu_pts_to_slice_sum r (SZ.v tid) (reveal end_) (reveal vv))
      (**)as
      (**)  if_ (div_pow2 (SZ.v it + 1) (SZ.v tid))
      (**)    (gpu_pts_to_slice_sum r
      (**)        (SZ.v tid)
      (**)        (min (SZ.v tid + pow2 (SZ.v it + 1)) (SZ.v nth))
      (**)        (reveal vv));
      ();
    } else {
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
fn d_reduce
  (#et:Type0) {| scalar et |}
  (nth : szp { nth <= 1024 })
  (a : gpu_array et nth)
  (#s :  erased (seq et))
  (#_ : squash (Seq.length s == nth))
  (etid : enatlt nth)
  ()
  requires
    gpu **
    kpre nth a s etid **
    thread_id nth etid
  ensures
    gpu **
    kpost nth a s etid **
    thread_id nth etid
{
  let tid = get_tid (); rewrite each etid as SZ.v tid;

  (* Reduction *)
  let mut n = 0sz;

  (**)with ss. assert (gpu_pts_to_slice a tid (tid+1) ss);
  assert (pure (Seq.slice s tid (tid+1) `Seq.equal` seq![ss @! 0])); // sucks
  (**)if_intro_true (exists* ss. gpu_pts_to_slice_sum_inner a tid (tid + pow2 0) s ss);
  (**)fold (gpu_pts_to_slice_sum a tid (tid + pow2 0) s);
  (**)if_intro_true (gpu_pts_to_slice_sum a tid (tid + pow2 0) s);

  open FStar.SizeT;
  while (let it = !n; (spow2 it <^ nth))
    invariant c.
      exists* (it:sz).
        gpu **
        pts_to n it **
        mbarrier_tok nth (barrier_matrix nth a s) it tid **
        if_ (div_pow2 (SZ.v it) (SZ.v tid)) (gpu_pts_to_slice_sum a tid (min (tid + pow2 it) nth) s) **
        pure (c == (pow2 it < nth) /\ SZ.v it < 32)
  {
    let it = !n;
    iteration nth a s tid it;
    n := it +^ 1sz;
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
    gpu_pts_to_array a va
  ensures
    forall+ (i1:natlt d1) (i2:natlt d2).
      gpu_pts_to_slice a (i1 * d2 + i2) (i1 * d2 + i2 + 1) seq![va @! (i1 * d2 + i2)]
{
  open Kuiper.Enumerable;
  Array.gpu_array_slice_1 a;
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
    gpu_pts_to_array a va
{
  (* everything above is clearly reversible *)
  admit();
}

inline_for_extraction noextract
let kernel
  (#et:Type0) {| scalar et |}
  (lena : szp { lena < max_threads })
  (a : gpu_array et lena)
  (#va : erased (seq et) { Seq.length va == SZ.v lena })
: kernel_desc_1_n
    (gpu_pts_to_array a va)
    (exists* va'. gpu_pts_to_array a #1.0R va')
= {
  nthr = lena;
  f = d_reduce lena a #va;

  block_teardown = magic();
  block_setup = magic();
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
    gpu_pts_to_array a 'va
  ensures
    cpu **
    (exists* va'. gpu_pts_to_array a va') (* underspec *)
{
  gpu_pts_to_ref a; (* recall length, automate *)
  launch_sync (kernel lena a);

  // factor_array lena a 1 lena;

  // launch_kernel_n_m_barrier 1sz lena
  //   #(kpre  1 lena a 'va)
  //   #(kpost 1 lena a 'va)
  //   (fun etid -> kk lena a etid);

  // forevery_singleton_elim #(natlt 1) (fun bid -> forall+ (tid:natlt lena). kpost 1 lena a 'va bid tid);
  // forevery_tostar #(natlt lena) _;

  // bigstar_extract 0 lena (kpost 1 lena a 'va 0) 0;
  // if_elim_true _;

  // unfold (gpu_pts_to_slice_sum a 0 lena 'va);
  // // if_elim_true _;
  // (* ^ Bad unification from matching, instead of trying to prove that the condition
  //    of if_ p f is true (to match it with if_ true ?u) it picks ?u = if_ p f. *)
  // with pp ff. assert (if_ pp ff);
  // rewrite each pp as true;
  // if_elim_true _;
  // bigstar_emp_elim #_ #0 #0;
  // bigstar_emp_elim' #_ #1 #lena _;

  // ()
}
