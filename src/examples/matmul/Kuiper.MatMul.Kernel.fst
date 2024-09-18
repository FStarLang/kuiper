module Kuiper.MatMul.Kernel

#lang-pulse

open Kuiper
module SZ = FStar.SizeT

#set-options "--z3rlimit 40"

// NOTE: rows is actually unused in the extracted code. Erasing
// it involves some changes all around.
[@@CPrologue "__global__"]
fn kernel
  (rows: sz) (shared: sz) (columns: sz{rows * columns < pow2 64})
  (ga1 : gpu_array u64 (rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (r : gpu_array u64 (rows * columns))
  (#s1: erased (seq u64) {Seq.length s1 == rows * shared})
  (#s2: erased (seq u64) {Seq.length s2 == shared * columns})
  (nth : erased sz { SZ.v nth == SZ.v SZ.(rows *^ columns) })
  (etid : erased tid_t { gdim_x etid == SZ.v nth /\ bdim_x etid == 1 })
  requires gpu
    ** thread_id etid
    ** kpre rows shared columns ga1 ga2 r #s1 #s2 (SZ.v nth) (thread_index etid)
  ensures  gpu
    ** thread_id etid
    ** kpost rows shared columns ga1 ga2 r #s1 #s2 (SZ.v nth) (thread_index etid)
{
  open FStar.SizeT;

  let tid = block_idx_x () <: u32;
  let tid : sz = SZ.uint32_to_sizet tid;

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
       ** Impure.gpu_pts_to_matrix #u64 rows shared ga1 (SZ.v nth) s1
       ** Impure.gpu_pts_to_matrix #u64 shared columns ga2 (SZ.v nth) s2
  {
    let v = !i;
    let s = !sum;
    let v1 = Impure.gpu_matrix_read #_ #rows #shared ga1 #(SZ.v nth) #s1 trow v;
    let v2 = Impure.gpu_matrix_read #_ #shared #columns ga2 #(SZ.v nth) #s2 v tcol;

    sum := U64.add_mod (U64.mul_mod v1 v2) s;
    i := SZ.add v 1sz;
    assert (pure (trow < rows));
    assert (pure (tcol < columns));
    assert (pure (v+1 <= shared));
    assert (pure ((trow + 1) <= rows /\ (trow + 1) * shared <= rows * shared)); // Pulse #214, sigh
    (**)Pure.matmul_single_lemma rows shared columns s1 s2 trow tcol (SZ.v (SZ.add v 1sz));
    ()
  };

  let s = !sum;
  gpu_array_write #u64 #(rows * columns) #(SZ.v tid) #(SZ.v tid + 1) r tid s;

  with #v. assert (gpu_pts_to_array_slice r tid (tid + 1) v);
  (**)Seq.lemma_eq_intro v seq![s];
  (**)rewrite gpu_pts_to_array_slice r tid (tid + 1) v
    as gpu_pts_to_array_slice r tid (tid + 1) seq![s];

  ()
}
