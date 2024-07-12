module GPU.DotProduct2

open FStar.Mul
open Pulse.Lib.Array
open Pulse.Lib.Pervasives
module A = Pulse.Lib.Array
module SZ = FStar.SizeT
module U64 = FStar.UInt64
open Pulse.Lib.BigStar
open GPU

let size : nat = 1024

let mul (s1 s2: erased (FStar.Seq.seq U64.t)) (#_: squash (FStar.Seq.length s1 == FStar.Seq.length s2)): erased (FStar.Seq.seq U64.t)
  = FStar.Seq.init_ghost (FStar.Seq.length s1) (fun i -> U64.mul_mod (FStar.Seq.index s1 i) (FStar.Seq.index s2 i))

let rec log2 (n: nat{n <> 0}): (r:nat{r < n}) = if n = 1 then 0 else 1 + log2 (n / 2)
let rec pow_log_lemma (n: nat): Lemma (log2 (pow2 n) = n) = if n = 0 then () else pow_log_lemma (n - 1)

let div_pow2 (i tid: nat): bool = tid % pow2 i = 0

let rec div_pow2_lemma (i j tid: nat):
  Lemma
    (requires i < j)
    (ensures (div_pow2 j tid) ==> (div_pow2 i tid))
  = if not (div_pow2 j tid) then () else (
      if i = j - 1 then () else div_pow2_lemma i (j - 1) tid;
      FStar.Math.Lemmas.mod_mult_exact tid (pow2 (j - 1)) 2
  )

let min (a b: nat): nat = if a < b then a else b

// Pure SUM

let rec sum_seq (s: FStar.Seq.seq U64.t) (i j:nat) (#_: squash (i < j && j <= FStar.Seq.length s)): Tot U64.t (decreases j - i) =
  if i = j - 1 then FStar.Seq.index s i else U64.add_mod (FStar.Seq.index s i) (sum_seq s (i + 1) j)

let add_mod_assoc (a b c: U64.t): Lemma (U64.add_mod (U64.add_mod a b) c = U64.add_mod a (U64.add_mod b c)) = admit()

let rec sum_seq_lemma (s: FStar.Seq.seq U64.t) (i j k:nat):
  Lemma (requires i < j && j < k && k <= FStar.Seq.length s) (ensures sum_seq s i k = U64.add_mod (sum_seq s i j) (sum_seq s j k)) (decreases j - i) =
    if i = j - 1 then () else (sum_seq_lemma s (i + 1) j k; add_mod_assoc (FStar.Seq.index s i) (sum_seq s (i + 1) j) (sum_seq s j k))

// Impure SUM

let gpu_pts_to_slice_sum_inner
  (#sz:nat)
  (r: gpu_array U64.t sz)
  (i j:nat)
  (v: FStar.Seq.seq U64.t)
  (s: FStar.Seq.seq U64.t)
: slprop = gpu_pts_to_array_slice r i j s ** pure (i < j /\ j <= sz /\ FStar.Seq.length v = sz /\ FStar.Seq.length s = j - i /\ FStar.Seq.index s 0 = sum_seq v i j)

let gpu_pts_to_slice_sum
  (#sz:nat)
  (r: gpu_array U64.t sz)
  (i j:nat)
  (v: FStar.Seq.seq U64.t)
: slprop = if_ (i < j && j <= sz) (exists* s. gpu_pts_to_slice_sum_inner #sz r i j v s)

// Barrier

let barrier_matrix (nth: nat) (r : gpu_array U64.t nth) (v: erased (FStar.Seq.seq U64.t)) (it from to: nat): slprop =
  if_ (from = to + pow2 it) (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from)) (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) v))

// #push-options "--debug SMTFail --split_queries always"
// #push-options "--print_implicits"


```pulse
ghost fn unfold_barrier_matrix (nth: nat) (r : gpu_array U64.t nth) (v: erased (FStar.Seq.seq U64.t)) (it from to: nat)
  requires barrier_matrix nth r v it from to
  ensures  if_ (from = to + pow2 it) (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from)) (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) v))
{ unfold (barrier_matrix nth r v it from to) }
```

let even n = n % 2 == 0
let odd  n = ~ (n % 2 == 0)


let div_helper (n : nat) :
  Lemma ((~(even (n+1))) <==>  even n) = ()

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

```pulse
ghost fn fold_barrier_matrix_true
  (nth : nat)
  (r: gpu_array U64.t nth)
  (v: FStar.Seq.seq U64.t { FStar.Seq.length v == nth })
  (it: nat)
  (tid: nat { tid <= nth /\ tid >= pow2 it })
  (to: nat)
  requires if_ (to = tid - pow2 it) (if_ (op_Negation (div_pow2 (it + 1) tid) && div_pow2 it tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) v))
  ensures  barrier_matrix nth r v it tid to
{
  fold (barrier_matrix nth r v it tid to);
}
```

```pulse
ghost fn fold_barrier_matrix_false
  (nth : nat)
  (r: gpu_array U64.t nth)
  (v: FStar.Seq.seq U64.t { FStar.Seq.length v == nth })
  (it: nat)
  (tid: nat { tid <= nth /\ tid < pow2 it })
  (to: nat { to <= nth })
  requires emp
  ensures  barrier_matrix nth r v it tid to
{
  assert (pure (tid < to + pow2 it /\ not (op_Equality #nat tid (to + pow2 it))));
  if_intro_false (op_Equality #nat tid (to + pow2 it)) (if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid)) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) v));
  fold (barrier_matrix nth r v it tid to);
}
```

// #push-options "--print_implicits --print_bound_var_types"

```pulse
fn mk_barrier_pre
  (nth : nat)
  (r : gpu_array U64.t nth)
  (b: erased (barrier nth))
  (v: erased (FStar.Seq.seq U64.t))
  (#_: squash (FStar.Seq.length v == nth))
  (tid : nat{tid < nth})
  (it: nat)
  requires if_ (not (div_pow2 (it + 1) tid) && div_pow2 it tid)
      (gpu_pts_to_slice_sum #nth r tid (min (tid + pow2 it) nth) v)
  ensures bigstar 0 nth (barrier_matrix nth r v it tid)
{
  if (tid >= pow2 it) {
    bigstar_if_intro 0 0 nth (tid - pow2 it) (fun _ -> if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid)) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) v));
    bigstar_map #_ #_ #0 #nth #(fun (i:nat { 0 <= i /\ i < nth }) -> _ i) (fold_barrier_matrix_true nth r v it tid);
  } else {
    FStar.Math.Lemmas.modulo_lemma tid (pow2 it);
    FStar.Math.Lemmas.modulo_lemma 0 (pow2 (it + 1));
    // assert (pure (tid <> 0 ==> not (div_pow2 it tid)));
    // assert (pure (tid = 0 ==> (div_pow2 (it + 1) tid)));
    if_elim_false _ _;

    bigstar_emp_intro 0 0 nth;
    bigstar_map #_ #_ #0 #nth #(fun (i:nat { 0 <= i /\ i < nth }) -> emp) (fold_barrier_matrix_false nth r v it tid);
  }
}
```

```pulse
fn iteration
  (nth : nat)
  (r : gpu_array U64.t nth)
  (b: erased (barrier nth))
  (v: erased (FStar.Seq.seq U64.t))
  (#_: squash (FStar.Seq.length v == nth))
  (tid : nat{tid < nth})
  (it: nat)
  requires gpu
    ** mbarrier_tok (barrier_matrix nth r v) b it tid
    ** if_ (div_pow2 it tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) v)
  ensures gpu
    ** mbarrier_tok (barrier_matrix nth r v) b (it+1) tid
    ** if_ (div_pow2 (it+1) tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 (it + 1)) nth) v)
{
  assert (pure (FStar.Seq.length v = nth));

  case_split (div_pow2 (it + 1) tid) (if_ (div_pow2 it tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) v));
  if_flatten #(div_pow2 (it + 1) tid);
  if_flatten #(not (div_pow2 (it + 1) tid));

  div_pow2_lemma it (it + 1) tid;
  rewrite (if_ (div_pow2 (it + 1) tid && div_pow2 it tid)
            (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) v))
      as (if_ (div_pow2 (it + 1) tid)
            (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) v));

  mk_barrier_pre nth r b v tid it;
  mbarrier_wait #nth #(barrier_matrix nth r v) b #it #tid;

  bigstar_map #_ #_ #0 #nth (fun (from: nat { 0 <= from /\ from < nth }) -> unfold_barrier_matrix nth r v it from tid);

  // combine (div_pow2 (it + 1) tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) v) _;

  let middle = min (tid + pow2 it) nth;
  let end_ = min (tid + 2 * pow2 it) nth;

  if (tid + pow2 it < nth) {
    bigstar_if_elim #_ #0 #nth (tid + pow2 it) (fun (from: nat) -> if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from)) (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) v));

    div_pow2_lemma_2 it tid;
    combine (div_pow2 (it + 1) tid) (gpu_pts_to_slice_sum r (tid + pow2 it) (min (tid + pow2 it + pow2 it) nth) v) _;

    if (div_pow2 (it + 1) tid) {
      if_elim_true _ _;
    
      unfold (gpu_pts_to_slice_sum r tid middle v);
      if_elim_true _ _;
      unfold gpu_pts_to_slice_sum_inner;
      unfold (gpu_pts_to_slice_sum r middle end_ v);
      if_elim_true _ _;
      unfold gpu_pts_to_slice_sum_inner;

      let s1 = gpu_array_read #U64.t #nth #tid #middle r tid;
      // assert (pure (s1 == sum_seq v tid middle));
      let s2 = gpu_array_read #U64.t #nth #middle #end_ r middle;
      // assert (pure (s2 == sum_seq v middle end_));
      let s = U64.add_mod s1 s2;
      sum_seq_lemma v tid middle end_;
      // assert (pure ( s == sum_seq v tid end_ ));
      gpu_array_write #U64.t #nth #tid #middle r tid s;

      gpu_slice_concat #U64.t #nth r 1.0R tid middle end_;
      with seq. assert (gpu_pts_to_array_slice r tid end_ seq);
      // assert (pure (FStar.Seq.index seq 0 == s));
      fold (gpu_pts_to_slice_sum_inner #nth r tid end_ v seq);
      if_intro_true (tid < end_ && end_ <= nth) (exists* s. gpu_pts_to_slice_sum_inner #nth r tid end_ v s);
      fold (gpu_pts_to_slice_sum r tid end_ v);
      if_intro_true (div_pow2 (it + 1) tid) (gpu_pts_to_slice_sum r tid end_ v);
    } else {
      if_elim_false _ _;
      if_intro_false (div_pow2 (it + 1) tid) (gpu_pts_to_slice_sum r tid end_ v);
    }
  } else {
    bigstar_map #_ #_ #0 #nth #(fun (from:nat { 0 <= from /\ from < nth }) -> _ from)
      (fun (from: nat{0 <= from /\ from < nth}) ->
        if_elim_false (op_Equality #nat from (tid + pow2 it)) (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from)) (gpu_pts_to_slice_sum r from (min (from + pow2 it) nth) v)));
    bigstar_emp_elim #_;
  }
}
```

let kpre (nth: nat) (ga1 ga2 r : gpu_array U64.t nth) (#s1 #s2: erased (FStar.Seq.seq U64.t))
  (#_: squash ( FStar.Seq.length s1 == nth /\ FStar.Seq.length s2 == nth ))
  (b: erased (barrier nth)) (tid:nat{tid < nth})
  : slprop =
    (gpu_pts_to_array #U64.t #nth ga1 #(1.0R /. Real.of_int nth) s1 **
    gpu_pts_to_array #U64.t #nth ga2 #(1.0R /. Real.of_int nth) s2) **
    gpu_pts_to_array1 r tid **
    mbarrier_tok (barrier_matrix nth r (mul s1 s2)) b 0 tid

let kpost (nth: nat) (ga1 ga2 r : gpu_array U64.t nth) (#s1 #s2: erased (FStar.Seq.seq U64.t))
  (#_: squash ( FStar.Seq.length s1 == nth /\ FStar.Seq.length s2 == nth ))
  (b: erased (barrier nth)) (tid:nat{tid < nth})
  : slprop =
    ((gpu_pts_to_array #U64.t #nth ga1 #(1.0R /. Real.of_int nth) s1 **
    gpu_pts_to_array #U64.t #nth ga2 #(1.0R /. Real.of_int nth) s2) **
    if_ (tid = 0) (gpu_pts_to_slice_sum r 0 nth (mul s1 s2))) **
    (exists* it. mbarrier_tok (barrier_matrix nth r (mul s1 s2)) b it tid)

// #set-options "--ext pulse:env_on_err=1"

```pulse
fn kernel
  (nth : nat)
  (ga1 ga2 : gpu_array U64.t nth)
  (r : gpu_array U64.t nth)
  (#s1 #s2: erased (FStar.Seq.seq U64.t))
  (#_: squash ( FStar.Seq.length s1 == nth /\ FStar.Seq.length s2 == nth ))
  (b: erased (barrier nth))
  (tid : nat{tid < nth})
  requires gpu ** kpre  nth ga1 ga2 r #s1 #s2 b tid
  ensures  gpu ** kpost nth ga1 ga2 r #s1 #s2 b tid
{
  (**)unfold (kpre nth ga1 ga2 r #s1 #s2 b tid);

  (**)unfold (gpu_pts_to_array #U64.t #nth ga1 #(1.0R /. Real.of_int nth) s1);
  let v1 = gpu_array_read #U64.t #nth #0 #nth ga1 tid #s1;
  (**)fold (gpu_pts_to_array #U64.t #nth ga1 #(1.0R /. Real.of_int nth) s1);

  (**)unfold (gpu_pts_to_array #U64.t #nth ga2 #(1.0R /. Real.of_int nth) s2);
  let v2 = gpu_array_read #U64.t #nth #0 #nth ga2 tid #s2;
  (**)fold (gpu_pts_to_array #U64.t #nth ga2 #(1.0R /. Real.of_int nth) s2);
  
  let vm = U64.mul_mod v1 v2;
  let dot_v = mul s1 s2;
  
  (**)unfold (gpu_pts_to_array1 r tid);
  gpu_array_write #U64.t #nth #tid #(tid+1) r tid vm;
  
  (* Reduction *)
  let mut n = 0 <: nat;

  (**)with s. assert (gpu_pts_to_array_slice r tid (tid+1) s);
  (**)fold (gpu_pts_to_slice_sum_inner #nth r tid (tid+1) dot_v s);
  (**)if_intro_true (tid < tid + pow2 0 && tid + pow2 0 <= nth) (exists* s. gpu_pts_to_slice_sum_inner #nth r tid (tid + pow2 0) dot_v s);
  (**)fold (gpu_pts_to_slice_sum r tid (tid + pow2 0) dot_v);
  (**)if_intro_true (div_pow2 0 tid) (gpu_pts_to_slice_sum r tid (tid + pow2 0) dot_v);

  while (let it = !n; (pow2 it < nth))
    invariant c.
    exists* (it:nat).
      gpu **
      pts_to n it **
      mbarrier_tok #nth (barrier_matrix nth r dot_v) b it tid **
      if_ (div_pow2 it tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) dot_v) **
      pure (c == (pow2 it < nth))
  {
    let it = !n <: nat;
    iteration nth r b dot_v tid it;
    n := it + 1;
  };
  
  (**)let it = !n;
  (**)FStar.Math.Lemmas.modulo_lemma tid (pow2 it);
  // assert (pure (pow2 it >= nth /\ tid < nth /\ (div_pow2 it tid) == (tid = 0)));

  // rewrite (if_ (div_pow2 (reveal it) tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 (reveal it)) nth) dot_v))
  //     as  (if_ (tid = 0) (gpu_pts_to_slice_sum r tid nth dot_v));

  if (tid = 0) {
    (**)if_elim_true _ _;
    (**)if_intro_true (tid = 0) (gpu_pts_to_slice_sum r 0 nth dot_v);
    (**)fold (kpost nth ga1 ga2 r #s1 #s2 b tid);
  } else {
    (**)fold (kpost nth ga1 ga2 r #s1 #s2 b tid);
  };
}
```

let shared_array (#nth : nat { nth <> 0 }) (ga : gpu_array U64.t nth) (#v: FStar.Seq.seq U64.t { FStar.Seq.length v == nth }) (_: nat): slprop =
  gpu_pts_to_array ga #(1.0R /. Real.of_int nth) v

```pulse
fn share_array
  (#nth : nat { nth <> 0 })
  (ga : gpu_array U64.t nth)
  (#v: erased (FStar.Seq.seq U64.t) { reveal (FStar.Seq.length v) == nth })
  requires gpu_pts_to_array ga #1.0R v
  ensures  bigstar 0 nth (shared_array #nth ga #v)
{
  rewrite gpu_pts_to_array ga #1.0R v
    as gpu_pts_to_array ga #(1.0R /. of_int 1) v;
  admit();
}
```

```pulse
fn gather_array
  (#nth : nat { nth <> 0 })
  (ga : gpu_array U64.t nth)
  (#v: erased (FStar.Seq.seq U64.t) { reveal (FStar.Seq.length v) == nth })
  requires bigstar 0 nth (shared_array #nth ga #v)
  ensures  gpu_pts_to_array ga #1.0R v
{
  admit();
}
```

```pulse
fn main
  (a1 a2: array U64.t)
  (v1 v2: erased (Seq.Base.seq U64.t))
  (#_: squash (Seq.length v1 = size /\ Seq.length v2 = size))
  requires cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2
  returns dp: U64.t
  ensures  cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2 ** pure (dp == sum_seq (mul v1 v2) 0 size)
{
  let ar = A.alloc #U64.t 0UL (SZ.uint_to_t size);

  let mut i = 0sz;

  let ga1 = gpu_array_alloc #U64.t size;
  let ga2 = gpu_array_alloc #U64.t size;

  GPU.Array.gpu_memcpy_host_to_device a1 ga1;
  GPU.Array.gpu_memcpy_host_to_device a2 ga2;
  
  let gr = gpu_array_alloc #U64.t size;

  // Slicing the arrays
  (**)share_array ga1;
  (**)share_array ga2;
  (**)gpu_array_slice_1_underspec gr;

  let b = mk_mbarrier size (barrier_matrix size gr (mul v1 v2));

  // Boring combination of resources
  (**)bigstar_zip 0 size (shared_array ga1) (shared_array ga2);
  (**)bigstar_zip 0 size _ (gpu_pts_to_array1 gr);
  (**)bigstar_zip 0 size _ (mbarrier_tok (barrier_matrix size gr (mul v1 v2)) b 0);
  (**)rewrite
    (bigstar 0 size
      (fun i -> ((shared_array #size ga1 #v1 i **
                 shared_array #size ga2 #v2 i) **
                 gpu_pts_to_array1 gr i) **
                 mbarrier_tok (barrier_matrix size gr (mul v1 v2)) b 0 i))
  as
    (bigstar 0 size (fun i -> kpre size ga1 ga2 gr #v1 #v2 b i));
  (**)bigstar_uneta ();

  launch_kernel_n size #(kpre size ga1 ga2 gr #v1 #v2 b) #(kpost size ga1 ga2 gr #v1 #v2 b) (kernel size ga1 ga2 gr #v1 #v2 b);

  (**)bigstar_eta ();
  // TODO:
  (**)drop_
        (bigstar 0 size (fun i -> kpost size ga1 ga2 gr #v1 #v2 b i));
  let it = 10;
  (**)assume_
        (bigstar 0 size
          (fun i -> ((gpu_pts_to_array #U64.t #size ga1 #(1.0R /. Real.of_int size) v1 **
                    gpu_pts_to_array #U64.t #size ga2 #(1.0R /. Real.of_int size) v2) **
                    if_ (op_Equality #nat i 0) (gpu_pts_to_slice_sum gr 0 size (mul v1 v2))) **
                    mbarrier_tok (barrier_matrix size gr (mul v1 v2)) b it i
        ));
  
  (**)bigstar_unzip 0 size _ _;
  (**)bigstar_unzip 0 size _ _;
  (**)bigstar_unzip 0 size _ _;
  
  (**)bigstar_uneta () #0 #size #(shared_array #size ga1 #v1);
  gather_array ga1;
  (**)bigstar_uneta () #0 #size #(shared_array #size ga2 #v2);
  gather_array ga2;

  drop_mbarrier #size #(barrier_matrix size gr (mul v1 v2)) #b #it;

  bigstar_if_elim #_ #0 #size 0 (fun _ -> gpu_pts_to_slice_sum #size gr 0 size (mul v1 v2));

  unfold gpu_pts_to_slice_sum;
  if_elim_true _ _;
  unfold gpu_pts_to_slice_sum_inner;
  with res. assert (gpu_pts_to_array_slice gr 0 size res);
  fold (gpu_pts_to_array #U64.t #size gr #1.0R res);

  // TODO: don't copy whole array
  GPU.Array.gpu_memcpy_device_to_host ar gr;

  gpu_array_free ga1;
  gpu_array_free ga2;
  gpu_array_free gr;

  let dp = ar.(0sz);
  A.free ar;
  dp
}
```
