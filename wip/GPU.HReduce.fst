module GPU.HReduce

#set-options "--ext pulse:trace"

#lang-pulse

open GPU
open GPU.Barrier.RPM
open GPU.Math
open GPU.Seq.Common

module A = Pulse.Lib.Array
module SZ = FStar.SizeT
module U32 = FStar.UInt32
module U64 = FStar.UInt64

#set-options "--z3rlimit 20"

let size : sz = 1024sz

(* no polymorphism, but at least keep the definitions here *)
let ety = u64
let op = U64.add_mod
let neu = 0uL

let op_assoc () : Lemma (is_associative op) = admit() // prove
let op_neu () : Lemma (is_neutral_for neu op) = ()
let op_monoid () : Lemma (is_monoid neu op) [SMTPat (is_monoid neu op)] = op_assoc (); op_neu ()

[@@ CPrologue "__device__"]
let spow2 (s : sz{s < 32}) : r:sz{SZ.v r == pow2 (SZ.v s)} =
  (* Computing 2^s by 1<<s *)
  let r = SZ.uint32_to_sizet (U32.shift_left 1ul (sizet_to_u32 s)) in
  assume (SZ.v r == pow2 (SZ.v s)); // prove this from UInt library
  r

[@@ CPrologue "__device__"]
let sdiv_pow2 (i:sz{i < 32}) (tid: sz): bool =
  SZ.rem tid (spow2 i) = 0sz

[@@ CPrologue "__device__"]
let smin (a b : sz): sz =
  let open FStar.SizeT in
  if a <^ b then a else b

(* Ownership of array r between i and j. The first value of that slice
is the reduction of all the values in the (original) slice v. *)
let gpu_pts_to_slice_sum_inner
  (#sz:nat)
  (r : gpu_array u64 sz)
  (i j :nat)
  (v : seq u64)
  (s : seq u64)
: slprop
= gpu_pts_to_array_slice r i j s
  ** pure (i < j /\ j <= sz /\
           Seq.length v = sz /\
           Seq.length s = j - i /\
           Seq.index s 0 = GPU.Seq.Common.seq_fold_left op neu (Seq.slice v i j))

let gpu_pts_to_slice_sum
  (#sz:nat)
  ([@@@equate_strict] r: gpu_array u64 sz)
  (i j:nat)
  (v: seq u64)
: slprop
= if_ (i < j && j <= sz) (exists* s. gpu_pts_to_slice_sum_inner #sz r i j v s)

// Barrier

let barrier_matrix (nth: nat) (r : gpu_array u64 nth) (v: seq u64) (it from to: nat): slprop =
  if_ (from = to + pow2 it)
      (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
           (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) v))

ghost fn unfold_barrier_matrix (nth: nat) (r : gpu_array u64 nth) (v: erased (seq u64))
 (it from to: nat)
  requires barrier_matrix nth r v it from to
  ensures  if_ (from = to + pow2 it) (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from)) (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) v))
{
  unfold (barrier_matrix nth r v it from to)
}

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

ghost fn fold_barrier_matrix_true
  (nth : nat)
  (r: gpu_array u64 nth)
  (v: seq u64 { Seq.length v == nth })
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

ghost fn fold_barrier_matrix_false
  (nth : nat)
  (r: gpu_array u64 nth)
  (v: seq u64 { Seq.length v == nth })
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
  (r : gpu_array u64 nth)
  (vv: erased (seq u64))
  (#_: squash (Seq.length vv == nth))
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

[@@ CPrologue "__device__"]
fn iteration
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
  (r : gpu_array u64 nth)
  (vv: erased (seq u64))
  (#_: squash (Seq.length vv == nth))
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
  assume (pure (forall (x:nat). FStar.SizeT.fits x)); // CHEATING overflow
  assert (pure (Seq.length vv = nth));

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

  bigstar_map #_ #_ #0 #nth (fun (from: nat { 0 <= from /\ from < nth }) -> unfold_barrier_matrix nth r vv it from tid);

  // combine (div_pow2 (it + 1) tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv) _;

  let middle : sz = smin (tid +^ spow2 it) nth;
  (* We do not use end_ in extracted code, so we can use a nat and erase it
  so there are no traces in the extracted C. *)
  let end_   : erased nat = hide (min (tid + 2 * pow2 it) nth);

  if (tid +^ spow2 it <^ nth) {
    bigstar_if_elim #_ #0
      #nth (tid + pow2 it)
      (fun (from: nat) -> if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from)) (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) vv));

    let b = sdiv_pow2 (it +^ 1sz) tid;
    
    assume (pure (b <==> (div_pow2 (SZ.v it + 1) (SZ.v tid))));
    rewrite each (div_pow2 (SZ.v it + 1) (SZ.v tid)) as b;

    div_pow2_lemma_2 it tid;
    combine
      b
      (gpu_pts_to_slice_sum r (tid + pow2 it) (min (tid + pow2 it + pow2 it) nth) vv)
      _;
      
    if b {
      assert (pure (div_pow2 (SZ.v it + 1) (SZ.v tid)));
      if_elim_true _;

      unfold (gpu_pts_to_slice_sum r tid middle vv);
      assume (pure False); // FIXME
      if_elim_true (exists* s. gpu_pts_to_slice_sum_inner r tid middle vv s);
      with s. assert (gpu_pts_to_slice_sum_inner r tid middle vv s);
      ();
      unfold (gpu_pts_to_slice_sum_inner r tid middle vv s);
      unfold (gpu_pts_to_slice_sum r middle end_ vv);
      if_elim_true (exists* s. gpu_pts_to_slice_sum_inner r middle end_ vv s);
      unfold gpu_pts_to_slice_sum_inner;

      let s1 = gpu_array_read #u64 #(SZ.v nth) #(SZ.v tid) #(SZ.v middle) r tid;
      let s2 = gpu_array_read #u64 #(SZ.v nth) #(SZ.v middle) #end_ r middle;
      let s = op s1 s2;
      // sum_seq_lemma vv tid middle end_;
      
      // admit();
      // lemma_seq_fold_left_sum neu op s1 s2;
      assume (pure (s == seq_fold_left op neu (Seq.slice vv tid end_)));
      
      // assert (pure ( s == sum_seq vv tid end_ ));
      gpu_array_write #u64 #(SZ.v nth) #(SZ.v tid) #(SZ.v middle) r tid s;

      gpu_slice_concat #u64 #(SZ.v nth) r tid middle end_;
      with seq. assert (gpu_pts_to_array_slice r tid end_ seq);
      // assert (pure (Seq.index seq 0 == s));
      fold (gpu_pts_to_slice_sum_inner #nth r tid end_ vv seq);
      if_intro_true (exists* s. gpu_pts_to_slice_sum_inner #nth r tid end_ vv s);
      fold (gpu_pts_to_slice_sum r tid end_ vv);
      if_intro_true (gpu_pts_to_slice_sum r tid end_ vv);
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

let kpre (nth: nat) (a r : gpu_array u64 nth) (s : erased (seq u64))
  (#_: squash (Seq.length s == nth)) (tid:nat{tid < nth})
  : slprop =
    gpu_pts_to_array #u64 #nth a #(1.0R /. Real.of_int nth) s **
    gpu_pts_to_array1 r tid

let kpost (nth: nat) (a r : gpu_array u64 nth) (s : erased (seq u64))
  (#_: squash (Seq.length s == nth)) (tid:nat{tid < nth})
  : slprop =
    gpu_pts_to_array #u64 #nth a #(1.0R /. Real.of_int nth) s **
    if_ (tid = 0) (gpu_pts_to_slice_sum r 0 nth s)

#set-options "--debug SMTFail --split_queries always"

[@@ CPrologue "__global__"]
fn kernel
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
  (a r : gpu_array u64 nth)
  (#s :  erased (seq u64))
  (#_: squash (Seq.length s == nth))
  (etid : erased tid_t { (gdim_x etid <: nat) == 1ul /\ (bdim_x etid <: nat) == SZ.sizet_to_uint32 nth })
  requires
    gpu **
    thread_id etid **
    mbarrier_tok nth (barrier_matrix nth r s) 0 (tidx_x etid) **
    kpre nth a r s (thread_index etid)
  ensures 
    gpu **
    thread_id etid **
    (exists* it. mbarrier_tok nth (barrier_matrix nth r s) it (tidx_x etid)) **
    kpost nth a r s (thread_index etid)
{
  let s1 = ();
  let s2 = ();
  let ga1 = ();
  let ga2 = ();
  let tid = thread_idx_x ();
  let tid : sz = SZ.uint32_to_sizet tid;
  (**)unfold kpre ;

  (* Reduction *)
  let mut n = 0sz;

  (**)unfold (gpu_pts_to_array1 r tid);
  (**)with ss. assert (gpu_pts_to_array_slice r tid (tid+1) ss);
  gpu_pts_to_slice_ref r tid (tid+1);
  assert (pure (Seq.length ss == 1));
  (**)let v0 : erased ety = Ghost.hide (Seq.index ss 0);
  let r : erased ety = (seq_fold_left #u64 op neu ss);
  // assume (pure (reveal v0 == seq_fold_left op neu ss));
  admit();
  // (**)fold (gpu_pts_to_slice_sum_inner #nth r tid (tid+1) s ss);
  // let dot_v = v0;
  // (**)if_intro_true (exists* s. gpu_pts_to_slice_sum_inner #nth r tid (tid + pow2 0) dot_v s);
  // (**)fold (gpu_pts_to_slice_sum r tid (tid + pow2 0) dot_v);
  // (**)if_intro_true (gpu_pts_to_slice_sum r tid (tid + pow2 0) dot_v);

  // open FStar.SizeT;
  // while (let it = !n; (spow2 it <^ nth))
  //   invariant c.
  //   exists* (it:sz).
  //     gpu **
  //     pts_to n it **
  //     mbarrier_tok nth (barrier_matrix nth r dot_v) it tid **
  //     if_ (div_pow2 (SZ.v it) (SZ.v tid)) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) dot_v) **
  //     pure (c == (pow2 it < nth) /\ SZ.v it < 31)
  // {
  //   let it = !n <: nat;
  //   iteration nth r dot_v tid it;
  //   assume (pure (SZ.v it < 30)); // FIXME: overflow
  //   n := it +^ 1sz;
  // };
  
  // (**)let it = !n;
  // (**)FStar.Math.Lemmas.modulo_lemma tid (pow2 it);
  // // assert (pure (pow2 it >= nth /\ tid < nth /\ (div_pow2 it tid) == (tid = 0)));

  // // rewrite (if_ (div_pow2 (reveal it) tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 (reveal it)) nth) dot_v))
  // //     as  (if_ (tid = 0) (gpu_pts_to_slice_sum r tid nth dot_v));

  // if (tid = 0sz) {
  //   (**)if_elim_true (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) dot_v);
  //   (**)if_intro_true (gpu_pts_to_slice_sum r 0 nth dot_v);
  //   (**)fold (kpost nth ga1 ga2 r #s1 #s2 tid);
  // } else {
  //   (**)fold (kpost nth ga1 ga2 r #s1 #s2 tid);
  // };
}
