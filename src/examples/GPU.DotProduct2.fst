module GPU.DotProduct2

#lang-pulse

open FStar.Mul
open Pulse.Lib.Array
open Pulse.Lib.Pervasives
module A = Pulse.Lib.Array
module SZ = FStar.SizeT
module U32 = FStar.UInt32
module U64 = FStar.UInt64
open Pulse.Lib.BigStar
open GPU
open GPU.Barrier.RPM
open FStar.SizeT

#set-options "--z3rlimit 20"

let size : SZ.t = 1024sz

let mul (s1 s2: seq U64.t)
  : Ghost (seq U64.t)
          (requires Seq.length s1 == Seq.length s2)
          (ensures fun _ -> True)
  = Seq.init_ghost (Seq.length s1)
      (fun i -> U64.mul_mod (Seq.index s1 i) (Seq.index s2 i))

let rec log2 (n: nat{n <> 0}): GTot (r:nat{r < n}) =
  if n = 1 then 0 else 1 + log2 (n / 2)
let rec pow_log_lemma (n: nat) : Lemma (log2 (pow2 n) = n) =
  if n = 0 then () else pow_log_lemma (n - 1)

[@@ CPrologue "__device__"]
let spow2 (s : SZ.t{s < 32}) : r:SZ.t{SZ.v r == pow2 (SZ.v s)} =
  let r = SZ.uint32_to_sizet (U32.shift_left 1ul (sizet_to_u32 s)) in
  assume (SZ.v r == pow2 (SZ.v s)); // prove this from UInt library
  r

let div_pow2 (i tid: nat) : GTot bool =
  (=) #int (tid % pow2 i) 0

[@@ CPrologue "__device__"]
let sdiv_pow2 (i:SZ.t{i < 32}) (tid: SZ.t): bool =
  SZ.rem tid (spow2 i) = 0sz

let rec div_pow2_lemma (i j tid: nat):
  Lemma
    (requires i < j)
    (ensures (div_pow2 j tid) ==> (div_pow2 i tid))
  = if not (div_pow2 j tid) then () else (
      if i = j - 1 then () else div_pow2_lemma i (j - 1) tid;
      FStar.Math.Lemmas.mod_mult_exact tid (pow2 (j - 1)) 2
  )

let min (a b: nat) : GTot nat =
  if a < b then a else b

[@@ CPrologue "__device__"]
let smin (a b: SZ.t ): SZ.t = if a <^ b then a else b

// Pure SUM

let rec sum_seq (s: seq U64.t) (i j : nat)
  : Ghost U64.t (requires i < j /\ j <= Seq.length s)
                (ensures fun _ -> True)
                (decreases j - i)
=
  if i = j - 1 then Seq.index s i else U64.add_mod (Seq.index s i) (sum_seq s (i + 1) j)

let add_mod_assoc (a b c: U64.t): Lemma (U64.add_mod (U64.add_mod a b) c = U64.add_mod a (U64.add_mod b c)) = admit()

let rec sum_seq_lemma (s: seq U64.t) (i j k:nat):
  Lemma (requires i < j && j < k && k <= Seq.length s)
        (ensures sum_seq s i k = U64.add_mod (sum_seq s i j) (sum_seq s j k)) (decreases j - i) =
    if i = j - 1 then () else (sum_seq_lemma s (i + 1) j k; add_mod_assoc (Seq.index s i) (sum_seq s (i + 1) j) (sum_seq s j k))

// Impure SUM

let gpu_pts_to_slice_sum_inner
  (#sz:nat)
  (r: gpu_array U64.t sz)
  (i j:nat)
  (v: seq U64.t)
  (s: seq U64.t)
: slprop = gpu_pts_to_array_slice r i j s ** pure (i < j /\ j <= sz /\ Seq.length v = sz /\ Seq.length s = j - i /\ Seq.index s 0 = sum_seq v i j)

let gpu_pts_to_slice_sum
  (#sz:nat)
  (r: gpu_array U64.t sz)
  (i j:nat)
  (v: seq U64.t)
: slprop = if_ (i < j && j <= sz) (exists* s. gpu_pts_to_slice_sum_inner #sz r i j v s)

// Barrier

let barrier_matrix (nth: nat) (r : gpu_array U64.t nth) (v: seq U64.t) (it from to: nat): slprop =
  if_ (from = to + pow2 it)
      (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
           (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) v))

// #push-options "--debug SMTFail --split_queries always"
// #push-options "--print_implicits"


ghost fn unfold_barrier_matrix (nth: nat) (r : gpu_array U64.t nth) (v: erased (seq U64.t))
 (it from to: nat)
  requires barrier_matrix nth r v it from to
  ensures  if_ (op_Equality #int from (to + pow2 it)) (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from)) (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) v))
{
  unfold (barrier_matrix nth r v it from to)
}

// let even n : prop = n % 2 == 0
// let odd  n : prop = ~ (n % 2 == 0)

// let div_helper (n : nat) :
//   Lemma ((~(even (n+1))) <==>  even n) = ()

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
  (r: gpu_array U64.t nth)
  (v: seq U64.t { Seq.length v == nth })
  (it: nat)
  (tid: nat { tid <= nth /\ tid >= pow2 it })
  (to: nat)
  requires if_ (op_Equality #int to (tid - pow2 it))
             (if_ (not (div_pow2 (it + 1) tid) && div_pow2 it tid)
               (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) v))
  ensures  barrier_matrix nth r v it tid to
{
  fold (barrier_matrix nth r v it tid to);
}

ghost fn fold_barrier_matrix_false
  (nth : nat)
  (r: gpu_array U64.t nth)
  (v: seq U64.t { Seq.length v == nth })
  (it: nat)
  (tid: nat { tid <= nth /\ tid < pow2 it })
  (to: nat { to <= nth })
  requires emp
  ensures  barrier_matrix nth r v it tid to
{
  assert (pure (tid < to + pow2 it /\ not (op_Equality #int tid (to + pow2 it))));
  if_intro_false (if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid)) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) v));
  // (op_Equality #int tid (to + pow2 it))
  fold (barrier_matrix nth r v it tid to);
}

// #push-options "--print_implicits --print_bound_var_types"

ghost
fn mk_barrier_pre
  (nth : SZ.t { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
  (r : gpu_array U64.t nth)
  (vv: erased (seq U64.t))
  (#_: squash (Seq.length vv == nth))
  (tid : SZ.t{SZ.v tid < nth})
  (it: SZ.t{it < 31})
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

#set-options "--print_implicits"

[@@ CPrologue "__device__"]
fn iteration
  (nth : SZ.t { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
  (r : gpu_array U64.t nth)
  (vv: erased (seq U64.t))
  (#_: squash (Seq.length vv == nth))
  (tid : SZ.t{SZ.v tid < nth})
  (it: SZ.t{it < 31})
  requires gpu
    ** mbarrier_tok nth (barrier_matrix nth r vv) it tid
    ** if_ (div_pow2 it tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv)
  ensures gpu
    ** mbarrier_tok nth (barrier_matrix nth r vv) (it+1) tid
    ** if_ (div_pow2 (it+1) tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 (it + 1)) nth) vv)
{
  open FStar.SizeT;
  assume_ (pure (forall (x:nat). FStar.SizeT.fits x)); // CHEATING overflow
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

  let middle : SZ.t = smin (tid +^ spow2 it) nth;
  (* We do not use end_ in extracted code, so we can use a nat and erase it
  so there are no traces in the extracted C. *)
  let end_   : erased nat = hide (min (tid + 2 * pow2 it) nth);

  if (tid +^ spow2 it <^ nth) {
    bigstar_if_elim #_ #0
      #nth (tid + pow2 it)
      (fun (from: nat) -> if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from)) (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) vv));

    let b = sdiv_pow2 (it +^ 1sz) tid;
    
    assume_ (pure (b <==> (div_pow2 (SZ.v it + 1) (SZ.v tid))));
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
      if_elim_true (exists* s. gpu_pts_to_slice_sum_inner r tid middle vv s);
      unfold gpu_pts_to_slice_sum_inner;
      unfold (gpu_pts_to_slice_sum r middle end_ vv);
      if_elim_true (exists* s. gpu_pts_to_slice_sum_inner r middle end_ vv s);
      unfold gpu_pts_to_slice_sum_inner;

      let s1 = gpu_array_read #U64.t #(SZ.v nth) #(SZ.v tid) #(SZ.v middle) r tid;
      // assert (pure (s1 == sum_seq vv tid middle));
      let s2 = gpu_array_read #U64.t #(SZ.v nth) #(SZ.v middle) #end_ r middle;
      // assert (pure (s2 == sum_seq vv middle end_));
      let s = U64.add_mod s1 s2;
      sum_seq_lemma vv tid middle end_;
      // assert (pure ( s == sum_seq vv tid end_ ));
      gpu_array_write #U64.t #(SZ.v nth) #(SZ.v tid) #(SZ.v middle) r tid s;

      gpu_slice_concat #U64.t #(SZ.v nth) r tid middle end_;
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
        if_rewrite_bool (op_Equality #int from (tid + pow2 it)) false _);
    bigstar_map #_ #_ #0 #nth #(fun (from:nat { 0 <= from /\ from < nth }) -> _ from)
      (fun (from: nat{0 <= from /\ from < nth}) ->
        if_elim_false (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from)) (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) vv)));
    bigstar_emp_elim #_;
  }
}

let kpre (nth: nat) (ga1 ga2 r : gpu_array U64.t nth) (#s1 #s2: erased (seq U64.t))
  (#_: squash ( Seq.length s1 == nth /\ Seq.length s2 == nth )) (tid:nat{tid < nth})
  : slprop =
    (gpu_pts_to_array #U64.t #nth ga1 #(1.0R /. Real.of_int nth) s1 **
    gpu_pts_to_array #U64.t #nth ga2 #(1.0R /. Real.of_int nth) s2) **
    gpu_pts_to_array1 r tid

let kpost (nth: nat) (ga1 ga2 r : gpu_array U64.t nth) (#s1 #s2: erased (seq U64.t))
  (#_: squash ( Seq.length s1 == nth /\ Seq.length s2 == nth )) (tid:nat{tid < nth})
  : slprop =
    ((gpu_pts_to_array #U64.t #nth ga1 #(1.0R /. Real.of_int nth) s1 **
    gpu_pts_to_array #U64.t #nth ga2 #(1.0R /. Real.of_int nth) s2) **
    if_ (tid = 0) (gpu_pts_to_slice_sum r 0 nth (mul s1 s2)))

// #set-options "--ext pulse:env_on_err=1"

[@@ CPrologue "__global__"]
fn kernel
  (nth : SZ.t { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
  (ga1 ga2 : gpu_array U64.t nth)
  (r : gpu_array U64.t nth)
  (#s1 #s2: erased (seq U64.t))
  (#_: squash ( Seq.length s1 == nth /\ Seq.length s2 == nth ))
  (etid : erased tid_t { (gdim_x etid <: nat) == 1ul /\ (bdim_x etid <: nat) == SZ.sizet_to_uint32 nth })
  requires gpu ** thread_id etid ** mbarrier_tok nth (barrier_matrix nth r (mul s1 s2)) 0 (tidx_x etid) ** kpre  nth ga1 ga2 r #s1 #s2 (thread_index etid)
  ensures  gpu ** thread_id etid ** (exists* it. mbarrier_tok nth (barrier_matrix nth r (mul s1 s2)) it (tidx_x etid)) ** kpost nth ga1 ga2 r #s1 #s2 (thread_index etid)
{
  let tid : U32.t = thread_idx_x ();
  let tid : SZ.t = SZ.uint32_to_sizet tid;
  (**)unfold (kpre nth ga1 ga2 r #s1 #s2 tid);

  (**)unfold (gpu_pts_to_array #U64.t #(SZ.v nth) ga1 #(1.0R /. Real.of_int nth) s1);
  let v1 = gpu_array_read #U64.t #(SZ.v nth) #0 #(SZ.v nth) ga1 tid #s1;
  (**)fold (gpu_pts_to_array #U64.t #nth ga1 #(1.0R /. Real.of_int nth) s1);

  (**)unfold (gpu_pts_to_array #U64.t #nth ga2 #(1.0R /. Real.of_int nth) s2);
  let v2 = gpu_array_read #U64.t #(SZ.v nth) #0 #(SZ.v nth) ga2 tid #s2;
  (**)fold (gpu_pts_to_array #U64.t #nth ga2 #(1.0R /. Real.of_int nth) s2);
  
  let vm = U64.mul_mod v1 v2;
  let dot_v = hide (mul s1 s2);
  
  (**)unfold (gpu_pts_to_array1 r tid);
  gpu_array_write #U64.t #(SZ.v nth) #(SZ.v tid) #(hide (SZ.v tid+1)) r tid vm;
  
  (* Reduction *)
  let mut n = 0sz;

  (**)with s. assert (gpu_pts_to_array_slice r tid (tid+1) s);
  (**)fold (gpu_pts_to_slice_sum_inner #nth r tid (tid+1) dot_v s);
  (**)if_intro_true (exists* s. gpu_pts_to_slice_sum_inner #nth r tid (tid + pow2 0) dot_v s);
  (**)fold (gpu_pts_to_slice_sum r tid (tid + pow2 0) dot_v);
  (**)if_intro_true (gpu_pts_to_slice_sum r tid (tid + pow2 0) dot_v);

  while (let it = !n; (spow2 it <^ nth))
    invariant c.
    exists* (it:SZ.t).
      gpu **
      pts_to n it **
      mbarrier_tok nth (barrier_matrix nth r dot_v) it tid **
      if_ (div_pow2 (SZ.v it) (SZ.v tid)) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) dot_v) **
      pure (c == (pow2 it < nth) /\ SZ.v it < 31)
  {
    let it = !n <: nat;
    iteration nth r dot_v tid it;
    assume_ (pure (SZ.v it < 30)); // FIXME: overflow
    n := it +^ 1sz;
  };
  
  (**)let it = !n;
  (**)FStar.Math.Lemmas.modulo_lemma tid (pow2 it);
  // assert (pure (pow2 it >= nth /\ tid < nth /\ (div_pow2 it tid) == (tid = 0)));

  // rewrite (if_ (div_pow2 (reveal it) tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 (reveal it)) nth) dot_v))
  //     as  (if_ (tid = 0) (gpu_pts_to_slice_sum r tid nth dot_v));

  if (tid = 0sz) {
    (**)if_elim_true (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) dot_v);
    (**)if_intro_true (gpu_pts_to_slice_sum r 0 nth dot_v);
    (**)fold (kpost nth ga1 ga2 r #s1 #s2 tid);
  } else {
    (**)fold (kpost nth ga1 ga2 r #s1 #s2 tid);
  };
}

let shared_array (#nth : nat { nth <> 0 }) (ga : gpu_array U64.t nth) (#v: seq U64.t { Seq.length v == nth }) (_: nat): slprop =
  gpu_pts_to_array ga #(1.0R /. Real.of_int nth) v

ghost
fn share_array
  (#nth : nat { nth <> 0 })
  (ga : gpu_array U64.t nth)
  (#v: erased (seq U64.t) { reveal (Seq.length v) == nth })
  requires gpu_pts_to_array ga #1.0R v
  ensures  bigstar 0 nth (shared_array #nth ga #v)
{
  rewrite gpu_pts_to_array ga #1.0R v
    as gpu_pts_to_array ga #(1.0R /. of_int 1) v;
  admit();
}

ghost
fn gather_array
  (#nth : nat { nth <> 0 })
  (ga : gpu_array U64.t nth)
  (#v: erased (seq U64.t) { reveal (Seq.length v) == nth })
  requires bigstar 0 nth (shared_array #nth ga #v)
  ensures  gpu_pts_to_array ga #1.0R v
{
  admit();
}

fn main
  (a1 a2: array U64.t)
  (v1 v2: erased (seq U64.t))
  (#_: squash (Seq.length v1 = size /\ Seq.length v2 = size))
  requires cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2
  returns  dp: U64.t
  ensures  cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2 ** pure (dp == sum_seq (mul v1 v2) 0 size)
{
  let ar = A.alloc #U64.t 0UL size;

  let ga1 = gpu_array_alloc #U64.t size;
  let ga2 = gpu_array_alloc #U64.t size;

  GPU.Array.gpu_memcpy_host_to_device a1 ga1 size;
  GPU.Array.gpu_memcpy_host_to_device a2 ga2 size;
  
  let gr = gpu_array_alloc #U64.t size;

  // Slicing the arrays
  (**)share_array ga1;
  (**)share_array ga2;
  (**)gpu_array_slice_1_underspec gr;

  // Boring combination of resources
  (**)bigstar_zip 0 size (shared_array ga1) (shared_array ga2);
  (**)bigstar_zip 0 size _ (gpu_pts_to_array1 gr);
  (**)rewrite
    (bigstar 0 size
      (fun i -> ((shared_array #size ga1 #v1 i **
                 shared_array #size ga2 #v2 i) **
                 gpu_pts_to_array1 gr i)))
  as
    (bigstar 0 size (fun i -> kpre size ga1 ga2 gr #v1 #v2 i))
    by tadmit ();
  (**)bigstar_uneta ();

  rewrite
    bigstar 0 size
      (kpre size ga1 ga2 gr #v1 #v2)
  as
    bigstar 0 (1 * SZ.v size)
      (kpre size ga1 ga2 gr #v1 #v2);

  launch_kernel_n_m_barrier #0 1sz size
    #(kpre size ga1 ga2 gr #v1 #v2)
    #(kpost size ga1 ga2 gr #v1 #v2)
    #(barrier_matrix size gr (mul v1 v2))
    (fun etid -> kernel size ga1 ga2 gr #v1 #v2 etid);

  (**)bigstar_eta ();
  // TODO:
  (**)drop_
        (bigstar 0 (1 * SZ.v size) (fun i -> kpost size ga1 ga2 gr #v1 #v2 i));
  (**)assume_
        (bigstar 0 size
          (fun i -> ((gpu_pts_to_array #U64.t #size ga1 #(1.0R /. Real.of_int size) v1 **
                    gpu_pts_to_array #U64.t #size ga2 #(1.0R /. Real.of_int size) v2) **
                    if_ (op_Equality #int i 0) (gpu_pts_to_slice_sum gr 0 size (mul v1 v2)))
        ));
  
  (**)bigstar_unzip 0 size _ _;
  (**)bigstar_unzip 0 size _ _;
  
  (**)bigstar_uneta () #0 #0 #size #(shared_array #size ga1 #v1);
  gather_array ga1;
  (**)bigstar_uneta () #0 #0 #size #(shared_array #size ga2 #v2);
  gather_array ga2;

  bigstar_if_elim #_ #0 #size 0 (fun _ -> gpu_pts_to_slice_sum #size gr 0 size (mul v1 v2));

  unfold gpu_pts_to_slice_sum;
  if_elim_true _;
  unfold gpu_pts_to_slice_sum_inner;
  with res. assert (gpu_pts_to_array_slice gr 0 size res);
  fold (gpu_pts_to_array #U64.t #size gr #1.0R res);

  // TODO: don't copy whole array
  GPU.Array.gpu_memcpy_device_to_host ar gr size;

  gpu_array_free ga1;
  gpu_array_free ga2;
  gpu_array_free gr;

  let dp = ar.(0sz);
  A.free ar;
  dp
}
