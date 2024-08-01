module GPU.MatMulOpt.Kernel

#push-options "--fuel 1 --ifuel 1"

open FStar.Mul
open Pulse.Lib.Pervasives
open Pulse.Lib.BigStar
open GPU
open GPU.SizeT
open GPU.MatMulOpt.Barrier
open GPU.MatMulOpt.Layout
module Impure = GPU.MatMulOpt.Impure
module Pure = GPU.MatMulOpt.Pure
module SZ = FStar.SizeT
module U32 = FStar.UInt32
module U64 = FStar.UInt64

let blocksize : SZ.t = 32sz
// Thread Per Block
let tpb : SZ.t = SZ.(blocksize *^ blocksize)

// TODO: un-hardcode
let rows : (i: SZ.t { SZ.v i % SZ.v blocksize == 0 }) = 256sz // rows of ga1/r
// assume val rows : nat
let shared : SZ.t = 1024sz // columns of ga1, rows of ga2
let columns : (i: SZ.t { SZ.v i % SZ.v blocksize == 0 }) = 256sz // columns of ga2/r

let mapping (blocksize: pos) (columns rows: (i: pos { i % blocksize == 0 })) = titi_permutation blocksize blocksize (columns / blocksize) (rows / blocksize)
let mapping_lemma (blocksize: pos) (columns rows: (i: pos { i % blocksize == 0 })) (tid: nat { tid < rows * columns }):
  Lemma ((mapping blocksize columns rows).f tid < rows * columns) = ()

let mapping_inv_lemma (blocksize: pos) (columns rows: (i: pos { i % blocksize == 0 })) (tid: nat { tid < rows * columns }):
  Lemma (tid < blocksize * blocksize * (columns / blocksize) * (rows / blocksize)) = ()
let mapping_fixed = hide (mapping blocksize columns rows)

let singleton #a (elem: a) : Seq.Base.seq a = Seq.Base.cons elem Seq.Base.empty

let kpre_pair (rows shared columns: nat)
  (ga1: gpu_array U64.t (rows * shared))
  (ga2: gpu_array U64.t (shared * columns))
  (#s1: erased (Seq.Base.seq U64.t))
  (#s2: erased (Seq.Base.seq U64.t))
  (size: erased pos)
  : slprop
  =
  Impure.gpu_pts_to_matrix rows shared ga1 size s1
  ** Impure.gpu_pts_to_matrix shared columns ga2 size s2

let kpre (shared: nat)
  (rows columns: (i: nat { i % SZ.v blocksize == 0 }))
  (ga1: gpu_array U64.t (rows * shared))
  (ga2: gpu_array U64.t (shared * columns))
  (r: gpu_array U64.t (rows * columns))
  (#s1: erased (Seq.Base.seq U64.t) )
  (#s2: erased (Seq.Base.seq U64.t))
  (size: erased pos { reveal size == rows * columns })
  (idx : nat)
  : slprop
  =
  kpre_pair rows shared columns ga1 ga2 #s1 #s2 (hide (reveal size))
  ** (exists* sr. gpu_pts_to_array_slice r idx (idx+1) sr)

let lemma_div_lt (a b: nat) (c: pos): Lemma (requires a < b * c) (ensures 0 <= a / c /\ a / c < b) = ()
let lemma_mod_lt (a: nat) (c: pos): Lemma (0 <= a % c /\ a % c < c) = ()

let kpost (shared: nat)
  (rows columns: (i: nat { i % SZ.v blocksize == 0 }))
  (ga1: gpu_array U64.t (rows * shared))
  (ga2: gpu_array U64.t (shared * columns))
  (r: gpu_array U64.t (rows * columns))
  (#s1: erased (Seq.Base.seq U64.t) {Seq.Base.length s1 == rows * shared})
  (#s2: erased (Seq.Base.seq U64.t) {Seq.Base.length s2 == shared * columns})
  (size: erased pos { reveal size == rows * columns })
  (idx : nat { idx < rows * columns })
  : slprop
  =
  Impure.gpu_pts_to_matrix rows shared ga1 size s1
  ** Impure.gpu_pts_to_matrix shared columns ga2 size s2
  ** (lemma_div_lt idx rows columns;
      gpu_pts_to_array_slice r idx (idx+1) (singleton (Pure.matmul_single rows shared columns s1 s2 (idx / columns) (idx % columns) shared)))
  // ** (exists* s. gpu_pts_to_array_slice r tid (tid+1) s)

let shared_pre (nthr : SZ.t { 0 < nthr /\ nthr <= max_threads }) (it: nat) (ar: gpu_array U64.t SZ.(2sz *^ nthr)) (i: nat { 0 <= i /\ i < nthr }): slprop =
  gpu_pts_to_array1 ar i ** gpu_pts_to_array1 ar (i + nthr) ** mbarrier_tok nthr (barrier_mm nthr) it i

// #push-options "--print_implicits --print_bound_var_types"

```pulse
ghost
fn block_setup_ghost
  (nblk : SZ.t { 0 < reveal nblk /\ reveal nblk <= max_blocks })
  (nthr : SZ.t { 0 < nthr /\ nthr <= max_threads })
  (smem_sz : SZ.t { smem_sz == SZ.(2sz *^ nthr) })
  (ar: gpu_array U64.t smem_sz)
  (bid: SZ.t { 0 <= bid /\ SZ.v bid < SZ.v nblk })
  requires block_setup nthr ** (exists* v. gpu_pts_to_array #U64.t #smem_sz ar #1.0R v)
  ensures block_setup nthr ** bigstar 0 nthr (shared_pre nthr 0 ar)
{
  with v. assert (gpu_pts_to_array #U64.t #smem_sz ar #1.0R v);
  unfold gpu_pts_to_array ar v;
  gpu_slice_slice_1_underspec #1 ar #1.0R 0 smem_sz nthr;
  drop_   (bigstar #1 0 (SZ.v nthr - 0) (fun x -> gpu_pts_to_array1 ar (x + 0)));
  assume_ (bigstar #1 0 nthr            (fun x -> gpu_pts_to_array1 ar x));

  gpu_slice_slice_1_underspec #2 ar #1.0R nthr smem_sz smem_sz;
  drop_   (bigstar #2 0 (smem_sz - nthr) (fun x -> gpu_pts_to_array1 ar (x + nthr)));
  assume_ (bigstar #2 0 nthr             (fun x -> gpu_pts_to_array1 ar (x + nthr)));

  bigstar_zip #1 #2 #1 0 nthr _ _;

  mk_mbarrier nthr (barrier_mm nthr);
  bigstar_zip #1 #0 #0 0 nthr _ _;

  // FOLD:
  drop_   (bigstar #0 0 nthr (fun x -> gpu_pts_to_array1 ar x ** gpu_pts_to_array1 ar (x + nthr) ** mbarrier_tok nthr (barrier_mm nthr) 0 x));
  assume_ (bigstar #0 0 nthr (fun x -> shared_pre nthr 0 ar x));

  bigstar_uneta();
  gpu_slice_empty_elim ar smem_sz;
}
```

let shared_post (nthr : SZ.t { 0 < nthr /\ nthr <= max_threads }) (ar: gpu_array U64.t SZ.(2sz *^ nthr)) (i: nat { 0 <= i /\ i < nthr }): slprop =
  exists* it. shared_pre nthr it ar i

```pulse
val fn thread_id_to_idx_2_sz (tid: SZ.t { SZ.v tid < rows * columns })
  requires emp
  returns  idx: (i: SZ.t { SZ.v i < rows * columns /\ SZ.v i == (mapping blocksize columns rows).f (SZ.v tid) })
  ensures  emp
```

#push-options "--print_implicits --print_bound_var_types"

```pulse
fn kernel
  // (rows: nat) (shared: nat { shared < pow2 16 }) (columns: nat)
  (ga1 : gpu_array U64.t (rows * shared)) (ga2 : gpu_array U64.t (shared * columns)) (r : gpu_array U64.t (rows * columns))
  (#s1: erased (Seq.Base.seq U64.t) {Seq.Base.length s1 == rows * shared})
  (#s2: erased (Seq.Base.seq U64.t) {Seq.Base.length s2 == shared * columns})
  (nblk : erased SZ.t { 0 < reveal nblk /\ reveal nblk <= max_blocks })
  (nthr : erased SZ.t { 0 < reveal nthr /\ reveal nthr <= max_threads /\ reveal nthr = tpb })
  (size : erased SZ.t { SZ.v size == SZ.v SZ.(rows *^ columns) /\ reveal size == SZ.(nblk *^ nthr) })
  (ar: gpu_array U64.t SZ.(2sz *^ nthr))
  (etid : erased tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr })
  requires gpu
    ** thread_id etid
    ** shared_pre nthr 0 ar (SZ.v (tidx_x etid))
    ** (assert (thread_index etid < rows * columns); mapping_inv_lemma blocksize columns rows (thread_index etid);
      kpre shared rows columns ga1 ga2 r #s1 #s2 (SZ.v size) (mapping_fixed.f (thread_index etid)))
  ensures  gpu
    ** thread_id etid
    ** shared_post nthr ar (SZ.v (tidx_x etid))
    ** (mapping_lemma blocksize columns rows (thread_index etid); assert (mapping_fixed.f (thread_index etid) < rows * columns);
      kpost shared rows columns ga1 ga2 r #s1 #s2 (SZ.v size) (mapping_fixed.f (thread_index etid)))
{
  open FStar.SizeT;

  assert (pure (thread_index etid < rows * columns));
  mapping_inv_lemma blocksize columns rows (thread_index etid);
  mapping_lemma blocksize columns rows (thread_index etid);
  let tid : SZ.t = thread_idx_all ();
  mapping_inv_lemma blocksize columns rows tid;
  mapping_lemma blocksize columns rows tid;
  let idx = (mapping_fixed.f (SZ.v tid));
  let idx_sz = thread_id_to_idx_2_sz tid;

  unfold kpre shared rows columns ga1 ga2 r #s1 #s2 (SZ.v size) idx;
  unfold kpre_pair rows shared columns ga1 ga2 #s1 #s2 (SZ.v size);

  (* r[tid] = TODO *)
  let trow = SZ.div idx_sz columns;
  let tcol = SZ.rem idx_sz columns;
  lemma_div_lt idx_sz rows columns;
  assume_ (pure (tcol < columns));
  assert (pure ( idx_sz < rows * columns /\ trow < rows /\ tcol < columns ));
  // assert (pure (0 <= trow /\ trow < rows /\ 0 <= tcol /\ tcol < columns));

  let mut i = 0sz;
  let mut sum = 0UL;

  while (let v = !i; (v <^ shared))
     invariant b.
       exists* v.
       pure (0 <= shared /\ b == (SZ.v v < shared) /\ SZ.v v <= shared /\ SZ.v v >= 0) **
       pts_to i v **
       gpu **
       pts_to sum (Pure.matmul_single rows shared columns s1 s2 trow tcol (SZ.v v))
       ** Impure.gpu_pts_to_matrix #U64.t rows shared ga1 (SZ.v size) s1
       ** Impure.gpu_pts_to_matrix #U64.t shared columns ga2 (SZ.v size) s2
       ** shared_pre nthr (2 * v) ar (SZ.v (tidx_x etid))
  {
    let v = !i;
    let s = !sum;
    let v1 = Impure.gpu_matrix_read #U64.t #rows #shared ga1 #(SZ.v size) #s1 trow v;
    let v2 = Impure.gpu_matrix_read #U64.t #shared #columns ga2 #(SZ.v size) #s2 v tcol;

    i := SZ.add v 1sz;
    sum := U64.add_mod (U64.mul_mod v1 v2) s;

    (**)Pure.matmul_single_lemma rows shared columns s1 s2 trow tcol (SZ.v (SZ.add v 1sz));
    drop_   (shared_pre nthr (2 * v)       ar (SZ.v (tidx_x etid)));
    assume_ (shared_pre nthr (2 * (v + 1)) ar (SZ.v (tidx_x etid)));
    ()
  };

  let s = !sum;
  gpu_array_write #U64.t #(rows * columns) #idx #(idx + 1) r idx_sz s;

  with #v. assert (gpu_pts_to_array_slice r idx (idx + 1) v);
  (**)Seq.Base.lemma_eq_intro v (singleton s);
  (**)rewrite gpu_pts_to_array_slice r idx (idx + 1) v
    as gpu_pts_to_array_slice r idx (idx + 1) (singleton s);

  fold kpost shared rows columns ga1 ga2 r #s1 #s2 (SZ.v size) idx;
  fold shared_post nthr ar (SZ.v (tidx_x etid));
  ()
}
```


```pulse
ghost fn fold_pre_pair
  (rows shared columns: nat)
  (ga1: gpu_array U64.t (rows * shared))
  (ga2: gpu_array U64.t (shared * columns))
  (#s1: erased (Seq.Base.seq U64.t) {Seq.Base.length s1 == rows * shared})
  (#s2: erased (Seq.Base.seq U64.t) {Seq.Base.length s2 == shared * columns})
  (size: erased nat { size > 0 })
  (tid: nat)
  requires Impure.gpu_pts_to_matrix rows shared ga1 size s1
        ** Impure.gpu_pts_to_matrix shared columns ga2 size s2
  ensures  kpre_pair rows shared columns ga1 ga2 #s1 #s2 size
{
  fold kpre_pair rows shared columns ga1 ga2 #s1 #s2 size;
  ()
}
```

```pulse
ghost fn fold_pre
  (shared: nat)
  (rows columns: (i: nat { i % SZ.v blocksize == 0 }))
  (ga1: gpu_array U64.t (rows * shared))
  (ga2: gpu_array U64.t (shared * columns))
  (gr: gpu_array U64.t (rows * columns))
  (#s1: erased (Seq.Base.seq U64.t) {Seq.Base.length s1 == rows * shared})
  (#s2: erased (Seq.Base.seq U64.t) {Seq.Base.length s2 == shared * columns})
  (#sr: (Seq.Base.seq U64.t) {Seq.Base.length sr == 1})
  (size: erased nat { size == (rows * columns) })
  (tid: nat { tid < size /\ tid < rows * columns })
  requires kpre_pair rows shared columns ga1 ga2 #s1 #s2 size
        ** gpu_pts_to_array_slice #U64.t #size gr tid (tid+1) sr
  ensures  kpre shared rows columns ga1 ga2 gr #s1 #s2 size tid
{
  fold kpre shared rows columns ga1 ga2 gr #s1 #s2 size tid;
  ()
}
```


// #push-options "--print_implicits --print_bound_var_types"

// ```pulse
// ghost fn unfold_post
//   (shared: nat)
//   (rows columns: (i: nat { i % SZ.v blocksize == 0 }))
//   (ga1: gpu_array U64.t (rows * shared))
//   (ga2: gpu_array U64.t (shared * columns))
//   (gr: gpu_array U64.t (rows * columns))
//   (#s1: erased (Seq.Base.seq U64.t) {Seq.Base.length s1 == rows * shared})
//   (#s2: erased (Seq.Base.seq U64.t) {Seq.Base.length s2 == shared * columns})
//   (size: erased nat { reveal size == rows * columns })
//   (idx: nat { idx < rows * columns })
//   requires kpost shared rows columns ga1 ga2 gr #s1 #s2 size idx
//   ensures  Impure.gpu_pts_to_matrix rows shared ga1 size s1
//         ** Impure.gpu_pts_to_matrix shared columns ga2 size s2
//         ** (assert (idx < rows * columns /\ idx / columns < rows);
//             gpu_pts_to_array_slice gr idx (idx+1) (singleton (Pure.matmul_single rows shared columns s1 s2 (idx / columns) (idx % columns) shared)))
// {
//   unfold kpost shared rows columns ga1 ga2 gr #s1 #s2 size idx;
//   ()
// }
// ```

// #pop-options
