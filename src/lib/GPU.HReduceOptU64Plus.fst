module GPU.HReduceOptU64Plus
#set-options "--z3rlimit 10"

(* This module is specialized to U64 and addition.

The only admits are a boring fact about associativity of add_mod (unsure why
it's not already trivial in F* ) and lack of overflow of the iteration counter.
This last thing should fall out from the fact that any the size of an array must
fit in a sizet, and the log of that size even more so. *)

#lang-pulse

open GPU
open GPU.Barrier.RPM
open GPU.Math
open GPU.Seq.Common
open GPU.Kernel

module A = Pulse.Lib.Array
module SZ = FStar.SizeT
module U32 = FStar.UInt32
module U64 = FStar.UInt64

let op_assoc () : Lemma (is_associative op) = admit() // prove
let op_neu () : Lemma (is_neutral_for neu op) = ()
let op_monoid () : Lemma (is_monoid neu op) = op_assoc (); op_neu ()

(* same, also the op_monoid does not (cannot?) have a pattern. *)
let sum_lemma (s1 s2 : seq ety) : Lemma (sum (s1 `Seq.append` s2) == op (sum s1) (sum s2)) =
  op_monoid();
  lemma_seq_fold_left_sum neu op s1 s2

[@@ CPrologue "__device__"]
noextract inline_for_extraction
let spow2 (s : sz{s < 32}) : r:sz{SZ.v r == pow2 (SZ.v s)} =
  (* Computing 2^s by 1<<s *)
  SZ.uint32_to_sizet (U32.shift_left 1ul (sizet_to_u32 s))

[@@ CPrologue "__device__"]
noextract inline_for_extraction
let sdiv_pow2 (i:sz{i < 32}) (tid: sz) : bool =
  SZ.rem tid (spow2 i) = 0sz

let sdiv_pow2_ok (i:sz{i < 32}) (tid:sz) :
  Lemma (sdiv_pow2 i tid <==> div_pow2 (SZ.v i) (SZ.v tid))
        [SMTPat (sdiv_pow2 i tid)]
= calc (==) {
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

let barrier_matrix (nth: nat) (sr gr : gpu_array ety nth) (v: seq ety { Seq.length v == nth }) (it: nat) (from to : (i: nat { 0 <= i /\ i < nth })): slprop =
  if pow2 it < nth then if_ (from = to + pow2 it)
      (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
           (gpu_pts_to_slice_sum sr from (min (from + pow2 it) nth) v))
  else if_ (to = 0) (kpre nth gr v from)

val lemma_div_exact: a:int -> p:pos -> Lemma
  (a % p = 0 <==> a = p * (a / p))
let lemma_div_exact a p = ()

let div_pow2_lemma_2 (it tid: nat):
  Lemma (
    (not (div_pow2 (it + 1) (tid + pow2 it)) && div_pow2 it (tid + pow2 it))
    <==>
    div_pow2 (it + 1) tid
  ) =
    calc (<==>) {
      (not (div_pow2 (it + 1) (tid + pow2 it))) && div_pow2 it (tid + pow2 it) <: prop;
      <==> {}
      (tid + pow2 it)                       % (2 * pow2 it) <> 0 && (tid + pow2 it) % pow2 it = 0 <: prop;
      <==> { FStar.Math.Lemmas.lemma_div_mod_plus tid 1 (pow2 it) }
      (tid + pow2 it)                       % (2 * pow2 it) <> 0 && tid % pow2 it = 0 <: prop;
      <==> { lemma_div_exact tid (pow2 it) }
      (pow2 it * (tid / pow2 it) + pow2 it) % (2 * pow2 it) <> 0 && tid % pow2 it = 0 <: prop;
      <==> { FStar.Math.Lemmas.distributivity_add_right (pow2 it) (tid / pow2 it) 1 }
      (pow2 it * (tid / pow2 it + 1))       % (2 * pow2 it) <> 0 && tid % pow2 it = 0 <: prop;
      <==> { FStar.Math.Lemmas.modulo_scale_lemma (tid / pow2 it + 1) (pow2 it) 2 }
      pow2 it * ((tid / pow2 it + 1) % 2)                   <> 0 && tid % pow2 it = 0 <: prop;
      <==> {}
      pow2 it * ((tid / pow2 it) % 2)                        = 0 && tid % pow2 it = 0 <: prop;
      <==> { FStar.Math.Lemmas.modulo_scale_lemma (tid / pow2 it) (pow2 it) 2 }
      (pow2 it * (tid / pow2 it)) % (2 * pow2 it)            = 0 && tid % pow2 it = 0 <: prop;
      <==> { lemma_div_exact tid (pow2 it) }
      tid % (2 * pow2 it)                                    = 0 && tid % pow2 it = 0 <: prop;
      <==> { div_pow2_lemma it (it + 1) tid }
      div_pow2 (it + 1) tid;
    }

ghost fn if_else_intro_true (b : bool) (p1 p2 : slprop) (#_: squash b)
  requires p1
  ensures  (if b then p1 else p2)
{ () }

ghost fn if_else_elim_true (b : bool) (p1 p2 : slprop) (#_: squash b)
  requires (if b then p1 else p2)
  ensures  p1
{ () }

ghost fn fold_barrier_matrix_true
  (nth : nat)
  (sr: gpu_array ety nth)
  (gr: erased (gpu_array ety nth))
  (v: seq ety { Seq.length v == nth })
  (it: nat { pow2 it < nth })
  (tid: nat { tid < nth /\ tid >= pow2 it })
  (to: nat { to < nth })
  requires if_ (to = tid - pow2 it)
             (if_ (not (div_pow2 (it + 1) tid) && div_pow2 it tid)
               (gpu_pts_to_slice_sum sr tid (min (tid + pow2 it) nth) v))
  ensures  barrier_matrix nth sr gr v it tid to
{
  if_else_intro_true (pow2 it < nth) _ (if_ (to = 0) (kpre nth gr v tid)) #_;
  fold (barrier_matrix nth sr gr v it tid to);
}

ghost fn fold_barrier_matrix_false
  (nth : nat)
  (sr: gpu_array ety nth)
  (gr: erased (gpu_array ety nth))
  (v: seq ety { Seq.length v == nth })
  (it: nat { pow2 it < nth })
  (tid: nat { tid < nth /\ tid < pow2 it })
  (to: nat { to < nth })
  requires emp
  ensures  barrier_matrix nth sr gr v it tid to
{
  assert (pure (tid < to + pow2 it /\ not (tid = to + pow2 it)));
  if_intro_false (if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid)) (gpu_pts_to_slice_sum sr tid (min (tid + pow2 it) nth) v));
  // (tid = to + pow2 it)
  if_else_intro_true (pow2 it < nth) _ (if_ (to = 0) (kpre nth gr v tid)) #_;
  fold (barrier_matrix nth sr gr v it tid to);
}

// #push-options "--print_implicits --print_bound_var_types"

ghost
fn mk_barrier_pre
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
  (sr: gpu_array ety nth)
  (gr: erased (gpu_array ety nth))
  (vv: erased (seq ety))
  (#_: squash (Seq.length vv == nth))
  (tid : sz{SZ.v tid < nth})
  (it: sz{pow2 it < nth /\ it < 31})
  requires if_ (not (div_pow2 (it + 1) tid) && div_pow2 it tid)
      (gpu_pts_to_slice_sum #nth sr tid (min (tid + pow2 it) nth) vv)
  ensures bigstar 0 nth (barrier_matrix nth sr gr vv it tid)
{
  open FStar.SizeT;
  if (tid >=^ spow2 it) {
    bigstar_if_intro 0 nth (tid - pow2 it) (fun _ -> if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid)) (gpu_pts_to_slice_sum sr tid (min (tid + pow2 it) nth) vv));
    bigstar_map #_ #_ #0 #nth #(fun (i:nat { 0 <= i /\ i < nth }) -> _ i) (fold_barrier_matrix_true nth sr gr vv it tid);
  } else {
    FStar.Math.Lemmas.modulo_lemma tid (spow2 it);
    FStar.Math.Lemmas.modulo_lemma 0 (spow2 (it +^ 1sz));
    // assert (pure ((tid <: nat) <> 0 ==> not (div_pow2 it tid)));
    // assert (pure ((tid <: nat) = 0 ==> (div_pow2 (it + 1) tid)));
    if_rewrite_bool (op_Negation (div_pow2 (SZ.v it + 1) (SZ.v tid)) && div_pow2 (SZ.v it) (SZ.v tid)) false _;
    if_elim_false _;

    bigstar_emp_intro 0 nth;
    bigstar_map #_ #_ #0 #nth #(fun (i:nat { 0 <= i /\ i < nth }) -> emp) (fold_barrier_matrix_false nth sr gr vv it tid);
  }
}

[@@ CPrologue "__device__"]
inline_for_extraction
fn iteration
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
  (sr: gpu_array ety nth)
  (gr: erased (gpu_array ety nth))
  (vv: erased (seq ety))
  (#_: squash (Seq.length vv == nth))
  (tid : sz{SZ.v tid < nth})
  (it: sz{pow2 it < nth /\ it < 31})
  requires gpu
    ** mbarrier_tok nth (barrier_matrix nth sr gr vv) it tid
    ** if_ (div_pow2 it tid) (gpu_pts_to_slice_sum sr tid (min (tid + pow2 it) nth) vv)
  ensures gpu
    ** mbarrier_tok nth (barrier_matrix nth sr gr vv) (it+1) tid
    ** if_ (div_pow2 (it+1) tid) (gpu_pts_to_slice_sum sr tid (min (tid + pow2 (it + 1)) nth) vv)
{
  open FStar.SizeT;
  assert (pure (Seq.length vv = nth));

  case_split (div_pow2 (it + 1) tid) (if_ (div_pow2 it tid) (gpu_pts_to_slice_sum sr tid (min (tid + pow2 it) nth) vv));
  if_flatten #(div_pow2 (it + 1) tid);
  if_flatten #(not (div_pow2 (it + 1) tid));
  
  div_pow2_lemma it (it + 1) tid;
  rewrite (if_ (div_pow2 (it + 1) tid && div_pow2 it tid)
            (gpu_pts_to_slice_sum sr tid (min (tid + pow2 it) nth) vv))
      as (if_ (div_pow2 (it + 1) tid)
            (gpu_pts_to_slice_sum sr tid (min (tid + pow2 it) nth) vv));

  mk_barrier_pre nth sr gr vv tid it;
  mbarrier_wait #(SZ.v nth) #(barrier_matrix nth sr gr vv) #(SZ.v it) #(SZ.v tid);

  ghost fn aux (from : nat { 0 <= from /\ from < nth })
    requires barrier_matrix nth sr gr vv it from tid
    ensures  if_ (from = tid + pow2 it) (
               if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from)) (
                 gpu_pts_to_slice_sum sr from (min (from + pow2 it) nth) vv
             ))
  {
    unfold barrier_matrix;
    if_else_elim_true (pow2 it < nth) _ _ #_;
  };
  bigstar_map #_ #_ #0 #nth (fun (from: nat { 0 <= from /\ from < nth }) -> aux from );

  // combine (div_pow2 (it + 1) tid) (gpu_pts_to_slice_sum sr tid (min (tid + pow2 it) nth) vv) _;

  let middle : sz = smin (tid +^ spow2 it) nth;
  (* We do not use end_ in extracted code, so we can use a nat and erase it
  so there are no traces in the extracted C. *)
  let end_   : erased nat = hide (min (tid + 2 * pow2 it) nth);

  if (tid +^ spow2 it <^ nth) {
    bigstar_if_elim #_ #0
      #nth (tid + pow2 it)
      (fun (from: nat) -> if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from)) (gpu_pts_to_slice_sum sr from (min (from + pow2 it) nth) vv));

    let b = sdiv_pow2 (it +^ 1sz) tid;
    
    rewrite each (div_pow2 (SZ.v it + 1) (SZ.v tid)) as b;

    div_pow2_lemma_2 it tid;
    combine
      b
      (gpu_pts_to_slice_sum sr (tid + pow2 it) (min (tid + pow2 it + pow2 it) nth) vv)
      _;
      
    if b {
      assert (pure (div_pow2 (SZ.v it + 1) (SZ.v tid)));
      if_elim_true _;

      unfold (gpu_pts_to_slice_sum sr tid middle vv);
      if_elim_true (exists* s. gpu_pts_to_slice_sum_inner sr tid middle vv s);
      with s. assert (gpu_pts_to_slice_sum_inner sr tid middle vv s);
      ();
      unfold (gpu_pts_to_slice_sum_inner sr tid middle vv s);
      unfold (gpu_pts_to_slice_sum sr middle end_ vv);
      if_elim_true (exists* s. gpu_pts_to_slice_sum_inner sr middle end_ vv s);
      unfold gpu_pts_to_slice_sum_inner;

      let s1 = gpu_array_read #ety #(SZ.v nth) #(SZ.v tid) #(SZ.v middle) sr tid;
      let s2 = gpu_array_read #ety #(SZ.v nth) #(SZ.v middle) #end_ sr middle;
      let s = op s1 s2;
      // sum_seq_lemma vv tid middle end_;
      
      // lemma_seq_fold_left_sum neu op s1 s2;
      assert (pure (s1 == sum (Seq.slice vv tid middle)));
      assert (pure (s2 == sum (Seq.slice vv middle end_)));
      lem_append_slice vv tid middle end_;
      sum_lemma (Seq.slice vv tid middle) (Seq.slice vv middle end_);
      assert (pure (s == sum (Seq.slice vv tid end_)));
      
      // assert (pure ( s == sum_seq vv tid end_ ));
      gpu_array_write #ety #(SZ.v nth) #(SZ.v tid) #(SZ.v middle) sr tid s;

      gpu_slice_concat #ety #(SZ.v nth) sr tid middle end_;
      with seq. assert (gpu_pts_to_array_slice sr tid end_ seq);
      // assert (pure (Seq.index seq 0 == s));
      fold (gpu_pts_to_slice_sum_inner #nth sr tid end_ vv seq);
      if_intro_true (exists* s. gpu_pts_to_slice_sum_inner #nth sr tid end_ vv s);
      fold (gpu_pts_to_slice_sum sr tid end_ vv);
      if_intro_true (gpu_pts_to_slice_sum sr tid end_ vv);
      rewrite
        if_ true (gpu_pts_to_slice_sum sr (SZ.v tid) (reveal end_) (reveal vv))
      as
        if_ (div_pow2 (SZ.v it + 1) (SZ.v tid))
          (gpu_pts_to_slice_sum sr
              (SZ.v tid)
              (min (SZ.v tid + pow2 (SZ.v it + 1)) (SZ.v nth))
              (reveal vv));
      ();
    } else {
      if_elim_false _;
      if_intro_false (gpu_pts_to_slice_sum sr tid end_ vv);
    }
  } else {
    bigstar_map #_ #_ #0 #nth #(fun (from:nat { 0 <= from /\ from < nth }) -> _ from)
      (fun (from: nat{0 <= from /\ from < nth}) ->
        if_rewrite_bool (from = tid + pow2 it) false _);
    bigstar_map #_ #_ #0 #nth #(fun (from:nat { 0 <= from /\ from < nth }) -> _ from)
      (fun (from: nat{0 <= from /\ from < nth}) ->
        if_elim_false (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from)) (gpu_pts_to_slice_sum sr from (min (from + pow2 it) nth) vv)));
    bigstar_emp_elim #_;
  }
}

[@@ CPrologue "__device__"]
inline_for_extraction
fn reduce
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
  (sr: gpu_array ety nth)
  (gr: erased (gpu_array ety nth))
  (#s :  erased (seq ety))
  (#_: squash (Seq.length s == nth))
  (etid : erased tid_t { (gdim_x etid <: nat) == 1ul /\ (bdim_x etid <: nat) == SZ.sizet_to_uint32 nth })
  requires
    gpu **
    thread_id etid **
    mbarrier_tok nth (barrier_matrix nth sr gr s) 0 (tidx_x etid) **
    kpre nth sr s (thread_index etid)
  ensures 
    gpu **
    thread_id etid **
    (exists* it. mbarrier_tok nth (barrier_matrix nth sr gr s) it (tidx_x etid) ** pure (pow2 it >= nth /\ (it > 0 ==> pow2 (it - 1) < nth))) **
    kpost nth sr s (thread_index etid)
{
  let tid = thread_idx_x ();
  let tid : sz = SZ.uint32_to_sizet tid;

  (* Reduction *)
  let mut n = 0sz;

  (**)with ss. assert (gpu_pts_to_array_slice sr tid (tid+1) ss);
  (**) gpu_pts_to_slice_ref sr tid (tid+1);
  (**)let v0 : erased ety = Ghost.hide (Seq.index ss 0);
  (**)fold (gpu_pts_to_slice_sum_inner #nth sr tid (tid+1) s ss);
  (**)if_intro_true (exists* ss. gpu_pts_to_slice_sum_inner #nth sr tid (tid + pow2 0) s ss);
  (**)fold (gpu_pts_to_slice_sum sr tid (tid + pow2 0) s);
  (**)if_intro_true (gpu_pts_to_slice_sum sr tid (tid + pow2 0) s);

  open FStar.SizeT;
  while (let it = !n; (spow2 it <^ nth))
    invariant c.
    exists* (it:sz).
      gpu **
      pts_to n it **
      mbarrier_tok nth (barrier_matrix nth sr gr s) it tid **
      if_ (div_pow2 (SZ.v it) (SZ.v tid)) (gpu_pts_to_slice_sum sr tid (min (tid + pow2 it) nth) s) **
      pure (c == (pow2 it < nth) /\ SZ.v it < 31 /\ (SZ.v it > 0 ==> pow2 (SZ.v it - 1) < SZ.v nth))
  {
    let it = !n <: nat;
    iteration nth sr gr s tid it;
    n := it +^ 1sz;
  };

  (**)let it = !n;
  (**)FStar.Math.Lemmas.modulo_lemma tid (pow2 it);

  if (tid = 0sz) {
    (**)if_elim_true (gpu_pts_to_slice_sum sr tid (min (tid + pow2 it) nth) s);
    (**)if_intro_true (gpu_pts_to_slice_sum sr 0 nth s);
  }
}

ghost fn if_else_intro_false (b : bool) (p1 p2 : slprop) (#_: squash (not b))
  requires p2
  ensures  (if b then p1 else p2)
{ () }

ghost fn if_else_elim_false (b : bool) (p1 p2 : slprop) (#_: squash (not b))
  requires (if b then p1 else p2)
  ensures  p2
{ () }

ghost fn fold_barrier_matrix_last
  (nth : nat)
  (sr: gpu_array ety nth)
  (gr: erased (gpu_array ety nth))
  (v: seq ety { Seq.length v == nth })
  (it: nat { pow2 it >= nth })
  (tid: nat { tid < nth })
  (to: nat { to < nth })
  requires if_ (to = 0) (kpre nth gr v tid)
  ensures  barrier_matrix nth sr gr v it tid to
{
  if_else_intro_false (pow2 it < nth) (
    if_ (tid = to + pow2 it) (
      if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid)) (
        gpu_pts_to_slice_sum sr tid (min (tid + pow2 it) nth) v
      )
    )
  ) (if_ (to = 0) (kpre nth gr v tid)) #_;
  fold (barrier_matrix nth sr gr v it tid to);
}

ghost fn unfold_barrier_matrix_last
  (nth : nat)
  (sr: gpu_array ety nth)
  (gr: erased (gpu_array ety nth))
  (v: seq ety { Seq.length v == nth })
  (it: nat { pow2 it >= nth })
  (tid: nat { tid < nth })
  (from: nat { from < nth })
  requires barrier_matrix nth sr gr v it from tid
  ensures  if_ (tid = 0) (kpre nth gr v from)
{
  unfold barrier_matrix;
  if_else_elim_false (pow2 it < nth) _ _ #_;
}

[@@ CPrologue "__device__"]
noextract inline_for_extraction
fn copy_first
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
  (a : gpu_array ety nth)
  (#s :  erased (seq ety))
  (#_: squash (Seq.length s == nth))
  (ar: gpu_array ety nth)
  (tid : sz { SZ.v tid < nth })
  requires
    gpu **
    gpu_pts_to_array a s **
    gpu_pts_to_slice_sum ar 0 nth s
  ensures
    gpu **
    gpu_pts_to_slice_sum a 0 nth s **
    gpu_pts_to_slice_sum ar 0 nth s
{
  unfold gpu_pts_to_slice_sum;
  if_elim_true (exists* v. gpu_pts_to_slice_sum_inner ar 0 (SZ.v nth) s v);
  unfold gpu_pts_to_slice_sum_inner;

  let vv = gpu_array_read #ety #nth #0 #nth ar 0sz;
  unfold gpu_pts_to_array a s;
  gpu_array_write #ety #nth #0 #nth a 0sz vv;
  
  with v1. assert (gpu_pts_to_array_slice ar 0 (SZ.v nth) v1);
  fold gpu_pts_to_slice_sum_inner #nth ar 0 nth s v1;
  if_intro_true (exists* v. gpu_pts_to_slice_sum_inner #nth ar 0 nth s v);
  fold gpu_pts_to_slice_sum ar 0 nth s;

  with v2. assert (gpu_pts_to_array_slice a 0 (SZ.v nth) v2);
  fold gpu_pts_to_slice_sum_inner #nth a 0 nth s v2;
  if_intro_true (exists* v. gpu_pts_to_slice_sum_inner #nth a 0 nth s v);
  fold gpu_pts_to_slice_sum a 0 nth s;
  ()
}

// #push-options "--print_implicits --print_bound_var_types"
[@@ CPrologue "__device__"]
noextract inline_for_extraction
fn copy_first_all
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
  (a : gpu_array ety nth)
  (#s :  erased (seq ety))
  (#_: squash (Seq.length s == nth))
  (ar: gpu_array ety nth)
  (tid : sz { SZ.v tid < nth })
  requires
    gpu **
    kpost nth ar s tid **
    bigstar 0 nth (fun from -> if_ (SZ.v tid = 0) (kpre nth a s from))
  ensures
    gpu **
    kpost nth ar s tid **
    kpost nth a s tid
{
  if (tid = 0sz) {
    bigstar_map #_ #_ #0 #nth #(fun (from:nat { 0 <= from /\ from < nth }) -> _ from)
      (fun (from: nat{0 <= from /\ from < nth}) ->
        if_rewrite_bool (SZ.v tid = 0) true _);
    bigstar_map #_ #_ #0 #nth
      #(fun (from:nat { 0 <= from /\ from < nth }) -> _ from)
      #(fun (from:nat { 0 <= from /\ from < nth }) -> _ from)
      (fun (from: nat { 0 <= from /\ from < nth }) -> if_elim_true (kpre nth a s from));
    gpu_array_unslice_1 a;
    if_elim_true (gpu_pts_to_slice_sum ar 0 nth s);
    copy_first nth a #s ar tid;
    if_intro_true (gpu_pts_to_slice_sum a 0 nth s);
    if_intro_true (gpu_pts_to_slice_sum ar 0 nth s);
  } else {
    bigstar_map #_ #_ #0 #nth #(fun (from:nat { 0 <= from /\ from < nth }) -> _ from)
      (fun (from: nat{0 <= from /\ from < nth}) ->
        if_rewrite_bool (SZ.v tid = 0) false _);
    bigstar_map #_ #_ #0 #nth
      #(fun (from:nat { 0 <= from /\ from < nth }) -> _ from)
      #(fun _ -> emp)
      (fun (from: nat { 0 <= from /\ from < nth }) -> if_elim_false (kpre nth a s from));
    bigstar_emp_elim #_;
    if_intro_false (gpu_pts_to_slice_sum a 0 nth s);
    ()
  }
}

[@@ CPrologue "__device__"]
noextract inline_for_extraction
fn fixup
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
  (a : gpu_array ety nth)
  (#s :  erased (seq ety))
  (#_: squash (Seq.length s == nth))
  (ar: gpu_array ety nth)
  (tid : sz { SZ.v tid < nth })
  requires gpu **
    (exists* s. gpu_pts_to_array_slice a (SZ.v tid) (SZ.v tid + 1) s) **
    (exists* it. mbarrier_tok nth (barrier_matrix nth ar a s) it tid ** pure (pow2 it >= nth /\ (it > 0 ==> pow2 (it - 1) < nth))) **
    kpost nth ar s tid
  ensures
    gpu **
    shared_post nth ar a s tid **
    kpost nth a s tid
{
  with it. assert (mbarrier_tok nth (barrier_matrix nth ar a s) it tid);
  bigstar_if_intro 0 nth 0 (fun _ -> kpre nth a s tid);
  bigstar_map #_ #_ #0 #nth #(fun (i:nat { 0 <= i /\ i < nth }) -> _ i) (fold_barrier_matrix_last nth ar a s it tid);
  mbarrier_wait #(SZ.v nth) #(barrier_matrix nth ar a s) #it #(SZ.v tid);
  bigstar_map #_ #_ #0 #nth #(fun (i:nat { 0 <= i /\ i < nth }) -> _ i) (unfold_barrier_matrix_last nth ar a s it tid);
  copy_first_all nth a #s ar tid;
  ()
}

[@@ CPrologue "__global__"]
inline_for_extraction
fn k_reduce
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
  (a : gpu_array ety nth)
  (#s :  erased (seq ety))
  (#_: squash (Seq.length s == nth))
  (ear: erased (gpu_array ety nth))
  (etid : erased tid_t { (gdim_x etid <: nat) == 1ul /\ (bdim_x etid <: nat) == SZ.sizet_to_uint32 nth })
  requires
    gpu **
    thread_id etid **
    shmem_tok ear **
    shared_pre nth ear a s 0 (thread_index etid) **
    kpre nth a s (thread_index etid)
  ensures
    gpu **
    thread_id etid **
    shared_post nth ear a s (thread_index etid) **
    kpost nth a s (thread_index etid)
{
  let ar = obtain_shmem ear;
  let tid = thread_idx_x ();
  let tid : sz = SZ.uint32_to_sizet tid;
  assert (pure ((thread_index etid) == tid));
  let v = gpu_array_read #ety #nth #(SZ.v tid) #(SZ.v tid + 1) a tid;
  gpu_array_write #ety #nth #(SZ.v tid) #(SZ.v tid + 1) ar tid v;
  with ss. assert (gpu_pts_to_array_slice ar tid (tid+1) ss);
  FStar.Seq.lemma_eq_intro ss seq![Seq.index s tid];

  reduce nth ar a #s etid;

  fixup nth a #s ar tid;
  ()
}
