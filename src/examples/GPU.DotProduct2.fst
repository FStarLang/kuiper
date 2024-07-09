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

let dot (s1 s2: erased (FStar.Seq.seq U64.t)) (#_: squash (FStar.Seq.length s1 == FStar.Seq.length s2)): erased (FStar.Seq.seq U64.t)
  = FStar.Seq.init_ghost (FStar.Seq.length s1) (fun i -> U64.mul_mod (FStar.Seq.index s1 i) (FStar.Seq.index s2 i))

let rec log2 (n: nat{n <> 0}): (r:nat{r < n}) = if n = 1 then 0 else 1 + log2 (n / 2)
let rec pow_log_lemma (n: nat): Lemma (log2 (pow2 n) = n) = if n = 0 then () else pow_log_lemma (n - 1)

let div_pow_2 (i tid: nat): bool = tid % pow2 i = 0

let mod_lemma (a b: nat):
  Lemma (requires a < b) (ensures (a % b == a)) = ()

let rec div_pow_2_lemma (i j tid: nat):
  Lemma
    (requires i < j)
    (ensures (div_pow_2 j tid) ==> (div_pow_2 i tid))
  = if not (div_pow_2 j tid) then () else (
      if i = j - 1 then () else div_pow_2_lemma i (j - 1) tid;
      // TODO:
      assert (tid % (2 * pow2 (j - 1)) = 0);
      assume (tid % pow2 (j - 1) = 0))

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

// ```pulse
// fn slice_sum_read
//   (#sz: nat)
//   (#r: gpu_array U64.t sz)
//   (#i #j:nat)
//   (#v: FStar.Seq.seq U64.t)
//   (#_: squash (i < j /\ j <= sz /\ FStar.Seq.length v = sz))
//   requires gpu ** gpu_pts_to_slice_sum #sz r i j v
//   returns  result: U64.t
//   ensures  gpu ** gpu_pts_to_slice_sum #sz r i j v ** pure (result = sum_seq v i j)
// {
//   unfold gpu_pts_to_slice_sum;
//   if_elim_true _ _;
//   with s. assert (gpu_pts_to_slice_sum_inner #sz r i j v s);
//   unfold gpu_pts_to_slice_sum_inner;

//   let ret = gpu_array_read #U64.t #sz #i #j r i;

//   fold (gpu_pts_to_slice_sum_inner #sz r i j v s);
//   if_intro_true (i < j && j <= sz) (exists* s. gpu_pts_to_slice_sum_inner #sz r i j v s);
//   fold (gpu_pts_to_slice_sum r i j v);
//   ret
// }
// ```

// Barrier

let barrier_pre (nth: nat) (r : gpu_array U64.t nth) (v: erased (FStar.Seq.seq U64.t)) (it tid: nat): slprop =
  if_ (not (div_pow_2 (it + 1) tid) && (div_pow_2 it tid)) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) v)

let barrier_post (nth: nat) (r : gpu_array U64.t nth) (v: erased (FStar.Seq.seq U64.t)) (it tid: nat): slprop =
  if_ (div_pow_2 (it + 1) tid) (gpu_pts_to_slice_sum r (min (tid + pow2 it) nth) (min (tid + 2 * pow2 it) nth) v)

// #push-options "--debug SMTFail --split_queries always"
// #push-options "--print_implicits"

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
    ** barrier_tok (barrier_pre nth r v) (barrier_post nth r v) b it tid
    ** if_ (div_pow_2 it tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) v)
  ensures gpu
    ** barrier_tok (barrier_pre nth r v) (barrier_post nth r v) b (it+1) tid
    ** if_ (div_pow_2 (it+1) tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 (it + 1)) nth) v)
{
  assert (pure (FStar.Seq.length v = nth));

  case_split (div_pow_2 (it + 1) tid) (if_ (div_pow_2 it tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) v));
  if_flatten #(div_pow_2 (it + 1) tid);
  if_flatten #(not (div_pow_2 (it + 1) tid));
  fold (barrier_pre nth r v it tid);
  barrier_wait #nth #(barrier_pre nth r v) #(barrier_post nth r v) b #it #tid;
  unfold barrier_post nth r v it tid;

  div_pow_2_lemma it (it + 1) tid;

  rewrite (if_ (div_pow_2 (it + 1) tid && div_pow_2 it tid)
            (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) v))
      as (if_ (div_pow_2 (it + 1) tid)
            (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) v));

  combine (div_pow_2 (it + 1) tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) v) _;

  if (tid % pow2 (it + 1) = 0) {
    // if_map (fun (_: unit) -> gpu_slice_concat r (1.0R) tid (tid + pow2 it) (tid + 2 * pow2 it));
    if_elim_true _ _;

    let middle = min (tid + pow2 it) nth;
    let end_ = min (tid + 2 * pow2 it) nth;


    if (tid + pow2 it < nth) {
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
      if_intro_true (div_pow_2 (it + 1) tid) (gpu_pts_to_slice_sum r tid end_ v);
    } else {
      unfold (gpu_pts_to_slice_sum r middle end_ v);
      if_elim_false _ _;
      // assert (pure (middle == end_));
      if_intro_true (div_pow_2 (it + 1) tid) (gpu_pts_to_slice_sum r tid end_ v);
    }
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
    barrier_tok (barrier_pre nth r (dot s1 s2)) (barrier_post nth r (dot s1 s2)) b 0 tid

let kpost (nth: nat) (ga1 ga2 r : gpu_array U64.t nth) (#s1 #s2: erased (FStar.Seq.seq U64.t))
  (#_: squash ( FStar.Seq.length s1 == nth /\ FStar.Seq.length s2 == nth ))
  (b: erased (barrier nth)) (tid:nat{tid < nth})
  : slprop =
    ((gpu_pts_to_array #U64.t #nth ga1 #(1.0R /. Real.of_int nth) s1 **
    gpu_pts_to_array #U64.t #nth ga2 #(1.0R /. Real.of_int nth) s2) **
    if_ (tid = 0) (gpu_pts_to_slice_sum r 0 nth (dot s1 s2))) **
    (exists* it. barrier_tok (barrier_pre nth r (dot s1 s2)) (barrier_post nth r (dot s1 s2)) b it tid)

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
  let dot_v = dot s1 s2;
  
  (**)unfold (gpu_pts_to_array1 r tid);
  gpu_array_write #U64.t #nth #tid #(tid+1) r tid vm;
  
  (* Reduction *)
  let mut n = 0 <: nat;

  (**)with s. assert (gpu_pts_to_array_slice r tid (tid+1) s);
  (**)fold (gpu_pts_to_slice_sum_inner #nth r tid (tid+1) dot_v s);
  (**)if_intro_true (tid < tid + pow2 0 && tid + pow2 0 <= nth) (exists* s. gpu_pts_to_slice_sum_inner #nth r tid (tid + pow2 0) dot_v s);
  (**)fold (gpu_pts_to_slice_sum r tid (tid + pow2 0) dot_v);
  (**)if_intro_true (div_pow_2 0 tid) (gpu_pts_to_slice_sum r tid (tid + pow2 0) dot_v);

  while (let it = !n; (pow2 it < nth))
    invariant c.
    exists* (it:nat).
      gpu **
      pts_to n it **
      barrier_tok #nth (barrier_pre nth r dot_v) (barrier_post nth r dot_v) b it tid **
      if_ (div_pow_2 it tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 it) nth) dot_v) **
      pure (c == (pow2 it < nth))
  {
    let it = !n <: nat;
    iteration nth r b dot_v tid it;
    n := it + 1;
  };
  
  (**)let it = !n;
  (**)mod_lemma tid (pow2 it);
  // assert (pure (pow2 it >= nth /\ tid < nth /\ (div_pow_2 it tid) == (tid = 0)));

  // rewrite (if_ (div_pow_2 (reveal it) tid) (gpu_pts_to_slice_sum r tid (min (tid + pow2 (reveal it)) nth) dot_v))
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
ghost fn barrier_proof
  (nth : nat)
  (r: gpu_array U64.t nth)
  (v: FStar.Seq.seq U64.t { FStar.Seq.length v == nth })
  (it: nat)
  requires bigstar 0 nth ((barrier_pre nth r v) it)
  ensures  bigstar 0 nth ((barrier_post nth r v) it)
{
  // TODO: proof
  admit()
}
```

// #push-options "--print_implicits"


```pulse
ghost fn bigstar_if
  (n m x: nat)
  (p: slprop)
  (#_: squash (n <= x /\ x < m))
  requires bigstar n m (fun i -> if_ (i = x) p)
  ensures  p
{
  admit()
}
```

```pulse
fn main
  (a1 a2: array U64.t)
  (v1 v2: erased (Seq.Base.seq U64.t))
  (#_: squash (Seq.length v1 = size /\ Seq.length v2 = size))
  requires cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2
  returns dp: U64.t
  ensures  cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2 ** pure (dp == sum_seq (dot v1 v2) 0 size)
{
  let ar = A.alloc #U64.t 0UL (SZ.uint_to_t size);

  let mut i = 0sz;

  // while (let v = !i; (SZ.v v < size))
  //    invariant b.
  //      exists* v. pts_to i v **
  //      (exists* s. A.pts_to a1 s ** pure (Seq.length s == size)) **
  //      (exists* s. A.pts_to a2 s ** pure (Seq.length s == size)) **
  //      pure (b == (SZ.v v < size))
  // {
  //   let v = !i;
  //   a1.(v) <- SZ.v v;
  //   a2.(v) <- SZ.v v;
  //   i := SZ.add v 1sz;
  //   ()
  // };

  let ga1 = gpu_array_alloc #U64.t size;
  let ga2 = gpu_array_alloc #U64.t size;

  GPU.Array.gpu_memcpy_host_to_device a1 ga1;
  GPU.Array.gpu_memcpy_host_to_device a2 ga2;
  
  let gr = gpu_array_alloc #U64.t size;

  // Slicing the arrays
  (**)share_array ga1;
  (**)share_array ga2;
  (**)gpu_array_slice_1_underspec gr;

  let b = mk_barrier size (barrier_pre size gr (dot v1 v2)) (barrier_post size gr (dot v1 v2)) (barrier_proof size gr (dot v1 v2));

  // Boring combination of resources
  (**)bigstar_zip 0 size (shared_array ga1) (shared_array ga2);
  (**)bigstar_zip 0 size _ (gpu_pts_to_array1 gr);
  (**)bigstar_zip 0 size _ (barrier_tok (barrier_pre size gr (dot v1 v2)) (barrier_post size gr (dot v1 v2)) b 0);
  (**)rewrite
    (bigstar 0 size
      (fun i -> ((shared_array #size ga1 #v1 i **
                 shared_array #size ga2 #v2 i) **
                 gpu_pts_to_array1 gr i) **
                 barrier_tok (barrier_pre size gr (dot v1 v2)) (barrier_post size gr (dot v1 v2)) b 0 i))
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
                    if_ (i = 0) (gpu_pts_to_slice_sum gr 0 size (dot v1 v2))) **
                    barrier_tok (barrier_pre size gr (dot v1 v2)) (barrier_post size gr (dot v1 v2)) b it i
        ));
  
  (**)bigstar_unzip 0 size _ _;
  (**)bigstar_unzip 0 size _ _;
  (**)bigstar_unzip 0 size _ _;
  
  (**)bigstar_uneta () #0 #size #(shared_array #size ga1 #v1);
  gather_array ga1;
  (**)bigstar_uneta () #0 #size #(shared_array #size ga2 #v2);
  gather_array ga2;

  drop_barrier #size #(barrier_pre size gr (dot v1 v2)) #(barrier_post size gr (dot v1 v2)) #b #it;

  bigstar_if 0 size 0 (gpu_pts_to_slice_sum #size gr 0 size (dot v1 v2)) #();
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
