module Kuiper.MatMul.Kernel

#lang-pulse

module U64 = FStar.UInt64
open Kuiper
module SZ = FStar.SizeT

#set-options "--z3rlimit 20 --retry 5"

// NOTE : rows is actually unused in the extracted code. But, erasing
// it involves some changes all around.
[@@CPrologue "__global__"]
fn kernel
  (rows : erased sz) (shared : sz) (columns : sz{reveal rows * columns < pow2 64})
  (ga1 : gpu_array u64 (reveal rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (r : gpu_array u64 (reveal rows * columns))
  (#s1 : erased (seq u64) {len s1 == reveal rows * shared})
  (#s2 : erased (seq u64) {len s2 == shared * columns})
  (nth : erased sz { SZ.v nth == reveal rows * columns })
  (etid : erased tid_t { gdim_x etid == SZ.v nth /\ bdim_x etid == 1 })
  requires gpu
    ** thread_id etid
    ** kpre (reveal rows) shared columns ga1 ga2 r s1 s2 (SZ.v nth) (thread_index etid)
  ensures  gpu
    ** thread_id etid
    ** kpost (reveal rows) shared columns ga1 ga2 r s1 s2 (SZ.v nth) (thread_index etid)
{
  open FStar.SizeT;

  let tid = block_idx_x ();
  rewrite each thread_index etid as tid;

  (* r[tid] = TODO *)
  let trow = SZ.div tid columns;
  let tcol = SZ.rem tid columns;
  // assert (pure (0 <= trow /\ trow < rows /\ 0 <= tcol /\ tcol < columns));

  assume (pure (SZ.fits (reveal (reveal rows) * columns))); // fixme, should come from ref
  let mut i = 0sz;
  let mut sum = 0UL;

  while (let v = !i; (v <^ shared))
     invariant b.
       exists* v.
       pure (0 <= shared /\ b == (SZ.v v < shared) /\ v <= shared /\ v >= 0) **
       pts_to i v **
       gpu **
       pts_to sum (P.matmul_single (reveal rows) shared columns s1 s2 trow tcol v)
       ** I.gpu_pts_to_matrix #u64 (reveal rows) shared ga1 (SZ.v nth) s1
       ** I.gpu_pts_to_matrix #u64 shared columns ga2 (SZ.v nth) s2
  {
    let v = !i;
    let s = !sum;
    (* Why doesn't this just work? *)
    let v1 = I.gpu_matrix_read #_ #_ #shared ga1 trow v;
    let v2 = I.gpu_matrix_read #_ #(hide shared) #columns ga2 v tcol;

    (* Using U64.(...) works but warns on every client (?) *)
    sum := FStar.UInt64.((v1 *%^ v2) +%^ s);
    i := SZ.add v 1sz;
    assert (pure (trow < (reveal rows)));
    assert (pure (tcol < columns));
    assert (pure (v+1 <= shared));
    assert (pure ((trow + 1) <= (reveal rows) /\ (trow + 1) * shared <= (reveal rows) * shared)); // Pulse #214
    (**)P.matmul_single_lemma (reveal rows) shared columns s1 s2 trow tcol (v + 1);
    ()
  };

  let s = !sum;
  gpu_array_write #_ #_ #tid #(tid + 1) r tid s; // r[tid] = s

  with v. assert (gpu_pts_to_slice r tid (tid + 1) v);
  (**)Seq.lemma_eq_intro v seq![s];
  (**)rewrite gpu_pts_to_slice r tid (tid + 1) v
    as gpu_pts_to_slice r (thread_index etid) (thread_index etid + 1) seq![s];

  ()
}
