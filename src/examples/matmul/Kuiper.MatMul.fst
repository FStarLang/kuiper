module Kuiper.MatMul
#lang-pulse

#push-options "--fuel 1 --ifuel 1"

open Kuiper

module A    = Pulse.Lib.Array
module SZ   = FStar.SizeT

module P = Kuiper.MatMul.Pure
module I = Kuiper.MatMul.Impure
module K = Kuiper.MatMul.Kernel

ghost
fn setup
  (rows: szp) (shared: szp) (columns: szp{rows * columns < pow2 64})
  (size: sz { SZ.v size == rows * columns })
  (ga1 : gpu_array u64 (rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (gr  : gpu_array u64 size)
  (v1: erased (seq u64) { len v1 == rows * shared })
  (v2: erased (seq u64) { len v2 == shared * columns })
  requires gpu_pts_to_array gr 's **
           gpu_pts_to_array ga1 v1 **
           gpu_pts_to_array ga2 v2
  ensures  bigstar 0 size (fun i ->
             K.kpre rows shared columns ga1 ga2 gr v1 v2 size i)
{
  // Sharing the input matrices (splitting permissions)
  fold I.gpu_pts_to_matrix rows   shared  ga1 1 v1;
  fold I.gpu_pts_to_matrix shared columns ga2 1 v2;
  I.gpu_matrix_share_underspec #_ #1 rows   shared  ga1 size v1;
  I.gpu_matrix_share_underspec #_ #2 shared columns ga2 size v2;

  // Sharing the output matrix (splitting each cell)
  gpu_pts_to_ref gr; (* obtain length v == size *)
  gpu_array_slice_1 #4 gr;

  // Join resources into a single bigstar
  bigstar_zip #1 #2 #3 0 size _ _;
  bigstar_zip #3 #4 #0 0 size _ _;

  // Rewrite inside the bigstar
  ghost
  fn aux (i:nat{0 <= i /\ i < size})
    requires
      I.gpu_pts_to_matrix rows   shared  ga1 size v1 **
      I.gpu_pts_to_matrix shared columns ga2 size v2 **
      gpu_pts_to_array_slice gr i (i + 1) seq!['s `Seq.index` i]
    ensures
      K.kpre rows shared columns ga1 ga2 gr v1 v2 size i
  {
    ()
  };
  bigstar_map #_ #_ #0 #size aux;
  bigstar_eta();
}

fn main
  (rows shared columns : szp)
  (a1 a2: array u64)
  (v1: erased (seq u64) { len v1 == rows * shared })
  (v2: erased (seq u64) { len v2 == shared * columns })
  preserves
    cpu **
    A.pts_to a1 v1 **
    A.pts_to a2 v2
  requires
    pure (
      rows * shared < pow2 64 /\
      shared * columns < pow2 64 /\
      rows * columns < pow2 64 /\
      rows * columns < max_blocks
    )
    // ^ Some of these could be ommited if we had some "core" pure inference from slprops.
    // Since we have a1 |-> v1, the length of v1 must fit, etc.
  returns  ar: array u64
  ensures
    A.pts_to ar (P.matmul rows shared columns v1 v2)
{
  open FStar.SizeT;
  let size = rows *^ columns;
  let ar = Pulse.Lib.Array.alloc 0UL size;

  let ga1 = gpu_array_alloc #u64 (rows *^ shared);
  let ga2 = gpu_array_alloc #u64 (shared *^ columns);

  Kuiper.Array.gpu_memcpy_host_to_device ga1 a1 (rows *^ shared);
  Kuiper.Array.gpu_memcpy_host_to_device ga2 a2 (shared *^ columns);

  let gr = gpu_array_alloc #u64 size;

  setup rows shared columns size ga1 ga2 gr v1 v2;

  launch_kernel_n #0
    size
    #(fun (tid: nat {0 <= tid /\ tid < size} ) -> K.kpre rows shared columns ga1 ga2 gr v1 v2 (SZ.v size) tid)
    #(fun (tid: nat {0 <= tid /\ tid < size} ) -> K.kpost rows shared columns ga1 ga2 gr v1 v2 (SZ.v size) tid)
    (fun etid -> K.kernel rows shared columns ga1 ga2 gr (hide size) etid);

  (**)bigstar_unzip 0 size _ _;
  (**)bigstar_unzip 0 size _ _;

  (**)I.gpu_matrix_unshare_underspec rows shared ga1 size v1;
  (**)I.gpu_matrix_unshare_underspec shared columns ga2 size v2;
  (**)unfold I.gpu_pts_to_matrix rows shared ga1 1 v1;
  (**)unfold I.gpu_pts_to_matrix shared columns ga2 1 v2;

  ghost
  fn aux1 (i:nat{0 <= i /\ i < size})
    requires
      gpu_pts_to_array_slice gr i (i + 1)
            seq![P.matmul_single rows shared columns
                   v1 v2 (i / columns) (i % columns) shared]
    ensures
      gpu_pts_to_array_slice gr i (i + 1)
            seq![P.matmul rows shared columns v1 v2 @! i]
  {
    P.lemma_matmul_index rows shared columns v1 v2 i;
    () (* cf. issue #181 in Pulse *)
  };
  bigstar_map #0 #0 #0 #size aux1;

  (**)gpu_array_unslice_1 gr;

  Kuiper.Array.gpu_memcpy_device_to_host ar gr size;

  gpu_array_free ga1;
  gpu_array_free ga2;
  gpu_array_free gr;

  ar
}
