module Kuiper.HReduceF32Plus
#set-options "--z3rlimit 10"

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Seq.Common

module SZ = FStar.SizeT
module U32 = FStar.UInt32

[@@ CPrologue "__device__"]
noextract inline_for_extraction
let spow2 (s : sz{s < 32}) : r:sz{SZ.v r == pow2 (SZ.v s)} =
  (* Computing 2^s by 1<<s *)
  SZ.uint32_to_sizet (U32.shift_left 1ul (sizet_to_u32 s))

[@@ CPrologue "__device__"]
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

[@@ CPrologue "__device__"]
noextract inline_for_extraction
let smin (a b : sz): sz =
  let open FStar.SizeT in
  if a <^ b then a else b

// Barrier

let barrier_matrix (nth: nat) (r : gpu_array ety nth) (v: seq ety) (it from to: nat): slprop =
  if_ (from = to + pow2 it)
      (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
           (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) v))

ghost
fn fold_barrier_matrix_true
  (nth : nat)
  (r: gpu_array ety nth)
  (v: seq ety { len v == nth })
  (it: nat)
  (tid: nat { tid <= nth /\ tid >= pow2 it })
  (to: nat)
  requires if_ (to = tid - pow2 it)
             (if_ (not (div_pow2 (it + 1) tid) && div_pow2 it tid)
               (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) v))
  ensures  barrier_matrix nth r v it tid to
{
  fold (barrier_matrix nth r v it tid to);
}

ghost
fn fold_barrier_matrix_false
  (nth : nat)
  (r: gpu_array ety nth)
  (v: seq ety { len v == nth })
  (it: nat)
  (tid: nat { tid <= nth /\ tid < pow2 it })
  (to: nat { to <= nth })
  requires emp
  ensures  barrier_matrix nth r v it tid to
{
  assert (pure (tid < to + pow2 it /\ not (tid = to + pow2 it)));
  if_intro_false (if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid)) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) v));
  // (tid = to + pow2 it)
  fold (barrier_matrix nth r v it tid to);
}

// #push-options "--print_implicits --print_bound_var_types"

ghost
fn mk_barrier_pre
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
  (r : gpu_array ety nth)
  (vv: erased (seq ety))
  (#_: squash (len vv == nth))
  (tid : sz{SZ.v tid < nth})
  (it: sz{it < 31})
  requires if_ (not (div_pow2 (it + 1) tid) && div_pow2 it tid)
      (gpu_pts_to_slice_sum #nth r tid (min (tid + pow2 it) nth) vv)
  ensures bigstar 0 nth (barrier_matrix nth r vv it tid)
{
  open FStar.SizeT;
  if (tid >=^ spow2 it) {
    bigstar_if_intro 0 nth (tid - pow2 it) (fun _ -> if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid)) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv));
    bigstar_map #_ #_ #0 #nth #(fun (i:nat { 0 <= i /\ i < nth }) -> _ i) (fold_barrier_matrix_true nth r vv it tid);
  } else {
    FStar.Math.Lemmas.modulo_lemma tid (spow2 it);
    FStar.Math.Lemmas.modulo_lemma 0 (spow2 (it +^ 1sz));
    // assert (pure ((tid <: nat) <> 0 ==> not (div_pow2 it tid)));
    // assert (pure ((tid <: nat) = 0 ==> (div_pow2 (it + 1) tid)));
    if_rewrite_bool (op_Negation (div_pow2 (SZ.v it + 1) (SZ.v tid)) && div_pow2 (SZ.v it) (SZ.v tid)) false _;
    if_elim_false _;

    bigstar_emp_intro 0 nth;
    bigstar_map #_ #_ #0 #nth #(fun (i:nat { 0 <= i /\ i < nth }) -> emp) (fold_barrier_matrix_false nth r vv it tid);
  }
}

// KrmlPrivate is essentially a "noextract". F* usually adds it
// automatically to any definition that does not appear in the fsti,
// but we have disable that since it interoperates poorly with pulse
// (due to splicing).
[@@ CPrologue "__device__"; "KrmlPrivate"]
inline_for_extraction
fn iteration
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
  (r : gpu_array ety nth)
  (vv: erased (seq ety))
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
  open FStar.SizeT;
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
  mbarrier_wait #(SZ.v nth) #(barrier_matrix nth r vv) #(SZ.v it) #(SZ.v tid);

  ghost fn aux (from : nat)
    requires barrier_matrix nth r vv it from tid
    ensures  if_ (from = tid + pow2 it) (
               if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from)) (
                 gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) vv
             ))
  {
    unfold barrier_matrix;
  };
  bigstar_map #_ #_ #0 #nth (fun (from: nat { 0 <= from /\ from < nth }) -> aux from );

  // combine (div_pow2 (it + 1) tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv) _;

  let nextid = tid +^ spow2 it;

  (* We do not use end_ in extracted code, so we can use a nat and erase it
  so there are no traces in the extracted C. *)
  let end_   : erased nat = hide (min (tid + 2 * pow2 it) nth);

  if (nextid <^ nth) {
    bigstar_if_elim #_ #0
      #nth (tid + pow2 it)
      (fun (from: nat) -> if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from)) (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) vv));

    let b = sdiv_pow2 (it +^ 1sz) tid;

    rewrite each (div_pow2 (SZ.v it + 1) (SZ.v tid)) as b;

    div_pow2_lemma_2 it tid;
    combine
      b
      (gpu_pts_to_slice_sum r nextid (min (tid + pow2 it + pow2 it) nth) vv)
      _;

    if b {
      assert (pure (div_pow2 (SZ.v it + 1) (SZ.v tid)));
      if_elim_true _;

      (**)unfold (gpu_pts_to_slice_sum r tid nextid vv);
      (**)if_elim_true (exists* s. gpu_pts_to_slice_sum_inner r tid nextid vv s);
      let s1 = gpu_array_read #_ #_ #tid #nextid r tid;
      (**)assert (pure (squash (is_reduction neu op (Seq.slice vv tid nextid) s1)));

      (**)unfold (gpu_pts_to_slice_sum r nextid end_ vv);
      (**)if_elim_true (exists* s. gpu_pts_to_slice_sum_inner r nextid end_ vv s);
      // (**)unfold gpu_pts_to_slice_sum_inner;
      let s2 = gpu_array_read #_ #_ #nextid #end_ r nextid;
      (**)assert (pure (squash (is_reduction neu op (Seq.slice vv nextid end_) s2)));

      let s = op s1 s2;
      (**)lem_append_slice vv tid nextid end_;
      (**)assert (pure (squash (is_reduction neu op (Seq.slice vv tid end_) s)));

      gpu_array_write #ety #(SZ.v nth) #(SZ.v tid) #(SZ.v nextid) r tid s;

      (**)gpu_slice_concat #ety #(SZ.v nth) r tid nextid end_;
      (**)with seq. assert (gpu_pts_to_slice r tid end_ seq);
      // (**)fold (gpu_pts_to_slice_sum_inner #nth r tid end_ vv seq);
      (**)if_intro_true (exists* s. gpu_pts_to_slice_sum_inner #nth r tid end_ vv s);
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

[@@ CPrologue "__device__"; "KrmlPrivate"]
inline_for_extraction
fn reduce
  (nth : szp { nth <= 1024 })
  (a : gpu_array ety nth)
  (#s :  erased (seq ety))
  (#_ : squash (len s == nth))
  (etid : erased tid_t { gdim_x etid == 1ul /\ bdim_x etid == nth })
  preserves
    gpu ** thread_id etid
  requires
    mbarrier_tok nth (barrier_matrix nth a s) 0 (tidx_x etid) **
    kpre nth a s (thread_index etid)
  ensures
    exists* it.
      mbarrier_tok nth (barrier_matrix nth a s) it (tidx_x etid) **
      kpost nth a s (thread_index etid)
{
  let tid = thread_idx_x ();
  let tid : sz = SZ.uint32_to_sizet tid;
  rewrite each thread_index etid as tid;

  (* Reduction *)
  let mut n = 0sz;

  (**)with ss. assert (gpu_pts_to_slice a tid (tid+1) ss);
  assert (pure (Seq.slice s tid (tid+1) `Seq.equal` seq![ss @! 0])); // sucks
  (**)if_intro_true (exists* ss. gpu_pts_to_slice_sum_inner #nth a tid (tid + pow2 0) s ss);
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

[@@ CPrologue "__global__"]
fn k_reduce
  (nth : szp { nth <= 1024 })
  (a : gpu_array ety nth)
  (#s :  erased (seq ety))
  (#_: squash (len s == nth))
  (etid : erased tid_t { (gdim_x etid <: nat) == 1ul /\ (bdim_x etid <: nat) == SZ.sizet_to_uint32 nth })
  requires
    gpu **
    thread_id etid **
    mbarrier_tok nth (barrier_matrix nth a s) 0 (tidx_x etid) **
    kpre nth a s (thread_index etid)
  ensures
    gpu **
    thread_id etid **
    (exists* it. mbarrier_tok nth (barrier_matrix nth a s) it (tidx_x etid)) **
    kpost nth a s (thread_index etid)
{
  reduce nth a #s etid;
}
