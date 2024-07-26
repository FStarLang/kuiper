module GPU.MatMulOpt.Kernel

#push-options "--fuel 1 --ifuel 1"

open FStar.Mul
open Pulse.Lib.Pervasives
open Pulse.Lib.BigStar
open GPU
open GPU.SizeT
open GPU.MatMulOpt.Barrier
module Impure = GPU.MatMulOpt.Impure
module Pure = GPU.MatMulOpt.Pure
module SZ = FStar.SizeT
module U32 = FStar.UInt32
module U64 = FStar.UInt64

let blocksize : SZ.t = 32sz
// Thread Per Block
let tpb : SZ.t = SZ.(blocksize *^ blocksize)

// TODO: un-hardcode
let rows : SZ.t = 256sz // rows of ga1/r
// assume val rows : nat
let shared : SZ.t = 1024sz // columns of ga1, rows of ga2
let columns : SZ.t = 256sz // columns of ga2/r

let singleton #a (elem: a) : Seq.Base.seq a = Seq.Base.cons elem Seq.Base.empty

let kpre_pair (rows shared columns: nat)
  (ga1: gpu_array U64.t (rows * shared))
  (ga2: gpu_array U64.t (shared * columns))
  (#s1: erased (Seq.Base.seq U64.t))
  (#s2: erased (Seq.Base.seq U64.t))
  (size: erased nat { size > 0 })
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
  (size: erased nat { reveal size == rows * columns })
  (tid : nat{ tid < rows * columns})
  : slprop
  =
  kpre_pair rows shared columns ga1 ga2 #s1 #s2 size
  ** (exists* sr. gpu_pts_to_array_slice r tid (tid+1) sr)

let kpost (shared: nat)
  (rows columns: (i: nat { i % SZ.v blocksize == 0 }))
  (ga1: gpu_array U64.t (rows * shared))
  (ga2: gpu_array U64.t (shared * columns))
  (r: gpu_array U64.t (rows * columns))
  (#s1: erased (Seq.Base.seq U64.t) {Seq.Base.length s1 == rows * shared})
  (#s2: erased (Seq.Base.seq U64.t) {Seq.Base.length s2 == shared * columns})
  (size: erased nat { reveal size == rows * columns })
  (tid : nat { tid < rows * columns })
  : slprop
  =
  Impure.gpu_pts_to_matrix rows shared ga1 size s1
  ** Impure.gpu_pts_to_matrix shared columns ga2 size s2
  ** gpu_pts_to_array_slice r tid (tid+1) (singleton (Pure.matmul_single rows shared columns s1 s2 (tid / columns) (tid % columns) shared))
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
    ** kpre shared rows columns ga1 ga2 r #s1 #s2 (SZ.v size) (thread_index etid)
  ensures  gpu
    ** thread_id etid
    ** shared_post nthr ar (SZ.v (tidx_x etid))
    ** kpost shared rows columns ga1 ga2 r #s1 #s2 (SZ.v size) (thread_index etid)
{
  open FStar.SizeT;

  let tid : SZ.t = thread_idx_all ();

  unfold kpre shared rows columns ga1 ga2 r #s1 #s2 (SZ.v size) (SZ.v tid);
  unfold kpre_pair rows shared columns ga1 ga2 #s1 #s2 (SZ.v size);

  (* r[tid] = TODO *)
  let trow = SZ.div tid columns;
  let tcol = SZ.rem tid columns;
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
  gpu_array_write #U64.t #(rows * columns) #((SZ.v tid)) #((SZ.v tid + 1)) r tid s;

  with #v. assert (gpu_pts_to_array_slice r tid (tid + 1) v);
  (**)Seq.Base.lemma_eq_intro v (singleton s);
  (**)rewrite gpu_pts_to_array_slice r tid (tid + 1) v
    as gpu_pts_to_array_slice r tid (tid + 1) (singleton s);

  fold kpost shared rows columns ga1 ga2 r #s1 #s2 (SZ.v size) tid;
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


#push-options "--print_implicits --print_bound_var_types"

```pulse
ghost fn unfold_post
  (shared: nat)
  (rows columns: (i: nat { i % SZ.v blocksize == 0 }))
  (ga1: gpu_array U64.t (rows * shared))
  (ga2: gpu_array U64.t (shared * columns))
  (gr: gpu_array U64.t (rows * columns))
  (#s1: erased (Seq.Base.seq U64.t) {Seq.Base.length s1 == rows * shared})
  (#s2: erased (Seq.Base.seq U64.t) {Seq.Base.length s2 == shared * columns})
  (size: erased nat { reveal size == rows * columns })
  (tid: nat {  tid < rows * columns })
  requires kpost shared rows columns ga1 ga2 gr #s1 #s2 size tid
  ensures  Impure.gpu_pts_to_matrix rows shared ga1 size s1
        ** Impure.gpu_pts_to_matrix shared columns ga2 size s2
        ** gpu_pts_to_array_slice gr tid (tid+1) (singleton (Pure.matmul_single rows shared columns s1 s2 (tid / columns) (tid % columns) shared))
{
  unfold kpost shared rows columns ga1 ga2 gr #s1 #s2 size tid;
  ()
}
```

#pop-options
