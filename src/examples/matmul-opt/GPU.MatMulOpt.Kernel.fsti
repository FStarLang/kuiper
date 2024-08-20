module GPU.MatMulOpt.Kernel

#lang-pulse

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

let blocksize : sz = 32sz
// Thread Per Block
let tpb : sz = SZ.(blocksize *^ blocksize)

// TODO: un-hardcode
let rows : (i: sz { SZ.v i % SZ.v blocksize == 0 }) = 256sz // rows of ga1/r
// assume val rows : nat
let shared : sz = 1024sz // columns of ga1, rows of ga2
let columns : (i: sz { SZ.v i % SZ.v blocksize == 0 }) = 256sz // columns of ga2/r

let mapping (blocksize: pos) (columns rows: (i: pos { i % blocksize == 0 })) = titi_permutation blocksize blocksize (columns / blocksize) (rows / blocksize)
let mapping_lemma (blocksize: pos) (columns rows: (i: pos { i % blocksize == 0 })) (tid: nat { tid < rows * columns }):
  Lemma ((mapping blocksize columns rows).f tid < rows * columns) = ()

let mapping_inv_lemma (blocksize: pos) (columns rows: (i: pos { i % blocksize == 0 })) (tid: nat { tid < rows * columns }):
  Lemma (tid < blocksize * blocksize * (columns / blocksize) * (rows / blocksize)) = ()
let mapping_fixed = hide (mapping blocksize columns rows)
let mapping_fixed_lemma (tid: nat { tid < rows * columns }): Lemma (mapping_fixed.f tid < rows * columns) = ()

let kpre_pair (rows shared columns: nat)
  (ga1: gpu_array u64 (rows * shared))
  (ga2: gpu_array u64 (shared * columns))
  (#s1: erased (seq u64))
  (#s2: erased (seq u64))
  (size: erased pos)
  : slprop
  =
  Impure.gpu_pts_to_matrix rows shared ga1 size s1
  ** Impure.gpu_pts_to_matrix shared columns ga2 size s2

let kpre (shared: nat)
  (rows columns: (i: nat { i % SZ.v blocksize == 0 }))
  (ga1: gpu_array u64 (rows * shared))
  (ga2: gpu_array u64 (shared * columns))
  (r: gpu_array u64 (rows * columns))
  (#s1: erased (seq u64) )
  (#s2: erased (seq u64))
  (size: erased pos { reveal size == rows * columns })
  (idx : nat)
  : slprop
  =
  kpre_pair rows shared columns ga1 ga2 #s1 #s2 (hide (reveal size))
  ** (exists* sr. gpu_pts_to_array_slice r idx (idx+1) sr)

let lemma_div_lt (a b: nat) (c: pos): Lemma (requires a < b * c) (ensures 0 <= a / c /\ a / c < b) = ()
let lemma_mod_lt (a: nat) (c: pos): Lemma (0 <= a % c /\ a % c < c) = ()

let kpost (shared: nat)
  (rows columns: (i: pos { i % SZ.v blocksize == 0 }))
  (ga1: gpu_array u64 (rows * shared))
  (ga2: gpu_array u64 (shared * columns))
  (r: gpu_array u64 (rows * columns))
  (#s1: erased (seq u64) {Seq.length s1 == rows * shared})
  (#s2: erased (seq u64) {Seq.length s2 == shared * columns})
  (size: erased pos { (reveal size <: nat) == rows * columns })
  (idx : nat { idx < rows * columns })
  : slprop
  = lemma_div_lt idx rows columns;
  Impure.gpu_pts_to_matrix rows shared ga1 size s1
  ** Impure.gpu_pts_to_matrix shared columns ga2 size s2
  ** gpu_pts_to_array_slice r idx (idx+1) (Pure.singleton (Pure.matmul_single rows shared columns s1 s2 (idx / columns) (idx % columns) shared))
  // ** (exists* s. gpu_pts_to_array_slice r tid (tid+1) s)

// #push-options "--print_implicits --print_bound_var_types"

fn thread_id_to_idx_2_sz (tid: sz { SZ.v tid < rows * columns })
  requires emp
  returns  idx: (i: sz { SZ.v i < rows * columns /\ SZ.v i == mapping_fixed.f (SZ.v tid) })
  ensures  emp

#push-options "--print_implicits --print_bound_var_types"

fn kernel
  // (rows: nat) (shared: nat { shared < pow2 16 }) (columns: nat)
  (ga1 : gpu_array u64 (rows * shared)) (ga2 : gpu_array u64 (shared * columns)) (r : gpu_array u64 (rows * columns))
  (#s1: erased (seq u64) {Seq.length s1 == rows * shared})
  (#s2: erased (seq u64) {Seq.length s2 == shared * columns})
  (nblk : erased sz { 0 < reveal nblk /\ reveal nblk <= max_blocks })
  (nthr : erased sz { 0 < reveal nthr /\ reveal nthr <= max_threads /\ reveal nthr = tpb })
  (size : erased sz { SZ.v size == SZ.v SZ.(rows *^ columns) /\ reveal size == SZ.(nblk *^ nthr) })
  (ar: gpu_array u64 SZ.(2sz *^ nthr))
  (etid : erased tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr })
  requires gpu
    ** thread_id etid
    ** shared_pre nthr s1 s2 0 ar (SZ.v (tidx_x etid))
    //** (assert (thread_index etid < rows * columns); mapping_inv_lemma blocksize columns rows (thread_index etid);
    ** kpre shared rows columns ga1 ga2 r #s1 #s2 (SZ.v size) (mapping_fixed.f (thread_index etid))//)
  ensures  gpu
    ** thread_id etid
    ** shared_post nthr ar (SZ.v (tidx_x etid))
    //** (mapping_fixed_lemma (thread_index etid); assert (mapping_fixed.f (thread_index etid) < rows * columns);
    ** kpost shared rows columns ga1 ga2 r #s1 #s2 (SZ.v size) (mapping_fixed.f (thread_index etid))//)
{
  open FStar.SizeT;

  assert (pure (thread_index etid < rows * columns));
  mapping_inv_lemma blocksize columns rows (thread_index etid);
  mapping_fixed_lemma (thread_index etid);
  let tid : sz = thread_idx_all ();
  mapping_inv_lemma blocksize columns rows tid;
  mapping_fixed_lemma (SZ.v tid);
  let idx = (mapping_fixed.f (SZ.v tid));
  assert (pure (idx < rows * columns));
  let idx_sz = thread_id_to_idx_2_sz tid;
  assert (pure (SZ.v idx_sz == idx));

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
       ** Impure.gpu_pts_to_matrix #u64 rows shared ga1 (SZ.v size) s1
       ** Impure.gpu_pts_to_matrix #u64 shared columns ga2 (SZ.v size) s2
       ** shared_pre nthr (2 * v) ar (SZ.v (tidx_x etid))
  {
    let v = !i;
    let s = !sum;
    let v1 = Impure.gpu_matrix_read #u64 #rows #shared ga1 #(SZ.v size) #s1 trow v;
    let v2 = Impure.gpu_matrix_read #u64 #shared #columns ga2 #(SZ.v size) #s2 v tcol;

    i := SZ.add v 1sz;
    sum := U64.add_mod (U64.mul_mod v1 v2) s;

    assert (pure (trow < rows /\ tcol < columns));
    (**)Pure.matmul_single_lemma rows shared columns s1 s2 trow tcol (SZ.v (SZ.add v 1sz));
    drop_   (shared_pre nthr (2 * v)       ar (SZ.v (tidx_x etid)));
    assume_ (shared_pre nthr (2 * (v + 1)) ar (SZ.v (tidx_x etid)));
    ()
  };

  let s = !sum;
  gpu_array_write #u64 #(rows * columns) #idx #(idx + 1) r idx_sz s;

  with #v. assert (gpu_pts_to_array_slice r idx (idx + 1) v);
  (**)Seq.lemma_eq_intro v (Pure.singleton s);
  (**)rewrite gpu_pts_to_array_slice r idx (idx + 1) v
    as gpu_pts_to_array_slice r idx (idx + 1) (Pure.singleton s);

  fold kpost shared rows columns ga1 ga2 r #s1 #s2 (SZ.v size) idx;
  fold shared_post nthr ar (SZ.v (tidx_x etid));
  ()
}


ghost fn fold_pre_pair
  (rows shared columns: nat)
  (ga1: gpu_array u64 (rows * shared))
  (ga2: gpu_array u64 (shared * columns))
  (#s1: erased (seq u64) {Seq.length s1 == rows * shared})
  (#s2: erased (seq u64) {Seq.length s2 == shared * columns})
  (size: erased pos)
  (tid: nat)
  requires Impure.gpu_pts_to_matrix rows shared ga1 size s1
        ** Impure.gpu_pts_to_matrix shared columns ga2 size s2
  ensures  kpre_pair rows shared columns ga1 ga2 #s1 #s2 size
{
  fold kpre_pair rows shared columns ga1 ga2 #s1 #s2 size;
  ()
}

// let kpre (shared: nat)
//   (rows columns: (i: nat { i % SZ.v blocksize == 0 }))
//   (ga1: gpu_array u64 (rows * shared))
//   (ga2: gpu_array u64 (shared * columns))
//   (r: gpu_array u64 (rows * columns))
//   (#s1: erased (seq u64) )
//   (#s2: erased (seq u64))
//   (size: erased pos { reveal size == rows * columns })
//   (idx : nat)
//   : slprop
//   =
//   kpre_pair rows shared columns ga1 ga2 #s1 #s2 (hide (reveal size))
//   ** (exists* sr. gpu_pts_to_array_slice r idx (idx+1) sr)

ghost fn fold_pre
  (shared: nat)
  (rows columns: (i: nat { i % SZ.v blocksize == 0 }))
  (ga1: gpu_array u64 (rows * shared))
  (ga2: gpu_array u64 (shared * columns))
  (gr: gpu_array u64 (rows * columns))
  (#s1: erased (seq u64) {Seq.length s1 == rows * shared})
  (#s2: erased (seq u64) {Seq.length s2 == shared * columns})
  (#sr: (seq u64) {Seq.length sr == 1})
  (size: erased pos { (reveal size <: nat) == rows * columns })
  (idx : nat)
  requires kpre_pair rows shared columns ga1 ga2 #s1 #s2 (hide (reveal size))
        ** gpu_pts_to_array_slice #u64 #size gr idx (idx+1) sr
  ensures  kpre shared rows columns ga1 ga2 gr #s1 #s2 size idx
{
  fold kpre shared rows columns ga1 ga2 gr #s1 #s2 size idx;
  ()
}


// #push-options "--print_implicits --print_bound_var_types"

// let kpost (shared: nat)
//   (rows columns: (i: nat { i % SZ.v blocksize == 0 }))
//   (ga1: gpu_array u64 (rows * shared))
//   (ga2: gpu_array u64 (shared * columns))
//   (r: gpu_array u64 (rows * columns))
//   (#s1: erased (seq u64) {Seq.length s1 == rows * shared})
//   (#s2: erased (seq u64) {Seq.length s2 == shared * columns})
//   (size: erased pos { reveal size == rows * columns })
//   (idx : nat { idx < rows * columns })
//   : slprop
//   =
//   Impure.gpu_pts_to_matrix rows shared ga1 size s1
//   ** Impure.gpu_pts_to_matrix shared columns ga2 size s2
//   ** (lemma_div_lt idx rows columns;
//       gpu_pts_to_array_slice r idx (idx+1) (Pure.singleton (Pure.matmul_single rows shared columns s1 s2 (idx / columns) (idx % columns) shared)))
//   // ** (exists* s. gpu_pts_to_array_slice r tid (tid+1) s)


// ghost fn unfold_post
//   (shared: nat)
//   (rows columns: (i: pos { i % SZ.v blocksize == 0 }))
//   (ga1: gpu_array u64 (rows * shared))
//   (ga2: gpu_array u64 (shared * columns))
//   (gr: gpu_array u64 (rows * columns))
//   (#s1: erased (seq u64) {Seq.length s1 == rows * shared})
//   (#s2: erased (seq u64) {Seq.length s2 == shared * columns})
//   (size: erased pos { (reveal size <: nat) == rows * columns })
//   (idx : nat { idx < rows * columns })
//   requires kpost shared rows columns ga1 ga2 gr #s1 #s2 size idx
//   ensures  (lemma_div_lt idx rows columns;
//         Impure.gpu_pts_to_matrix rows shared ga1 size s1
//         ** Impure.gpu_pts_to_matrix shared columns ga2 size s2
//         ** gpu_pts_to_array_slice #u64 #(rows * columns) gr #1.0R idx (idx+1) (Pure.singleton (Pure.matmul_single rows shared columns s1 s2 (idx / columns) (idx % columns) shared)))
// {
//   lemma_div_lt idx rows columns;
//   unfold (kpost shared rows columns ga1 ga2 gr #s1 #s2 size idx);
//   ()
// }

// #pop-options
