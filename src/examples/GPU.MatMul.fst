module GPU.MatMul
#lang-pulse

#push-options "--fuel 1 --ifuel 1"

open GPU

module A    = Pulse.Lib.Array
module SZ   = FStar.SizeT
module Defs = GPU.MatMul.Defs

let matmul_single = Defs.matmul_single Defs.rows Defs.shared Defs.columns
let matmul = Defs.matmul Defs.rows Defs.shared Defs.columns

ghost
fn setup
  (size: sz { size == SZ.(Defs.rows *^ Defs.columns) })
  (ga1 : gpu_array u64 (Defs.rows * Defs.shared))
  (ga2 : gpu_array u64 (Defs.shared * Defs.columns))
  (gr  : gpu_array u64 size)
  (v1: erased (seq u64) { Seq.length v1 == Defs.rows * Defs.shared })
  (v2: erased (seq u64) { Seq.length v2 == Defs.shared * Defs.columns })
  requires gpu_pts_to_array gr 's **
           gpu_pts_to_array ga1 v1 **
           gpu_pts_to_array ga2 v2
  ensures  bigstar 0 size (fun i ->
             Defs.kpre Defs.rows Defs.shared Defs.columns ga1 ga2 gr #v1 #v2 size i)
{
  // Sharing the input matrices (splitting permissions)
  fold Defs.gpu_pts_to_matrix Defs.rows   Defs.shared  ga1 1 v1;
  fold Defs.gpu_pts_to_matrix Defs.shared Defs.columns ga2 1 v2;
  Defs.gpu_matrix_share_underspec #_ #1 Defs.rows   Defs.shared  ga1 size v1;
  Defs.gpu_matrix_share_underspec #_ #2 Defs.shared Defs.columns ga2 size v2;

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
      Defs.gpu_pts_to_matrix Defs.rows   Defs.shared  ga1 size v1 **
      Defs.gpu_pts_to_matrix Defs.shared Defs.columns ga2 size v2 **
      gpu_pts_to_array_slice gr i (i + 1) seq!['s `Seq.index` i]
    ensures
      Defs.kpre Defs.rows Defs.shared Defs.columns ga1 ga2 gr #v1 #v2 size i
  {
    fold Defs.kpre_pair Defs.rows Defs.shared Defs.columns ga1 ga2 #v1 #v2 size;
    fold Defs.kpre;
  };
  bigstar_map #_ #_ #0 #size aux;
  bigstar_eta();
}

fn main
  (a1 a2: array u64)
  (v1: erased (seq u64) { Seq.length v1 == Defs.rows * Defs.shared })
  (v2: erased (seq u64) { Seq.length v2 == Defs.shared * Defs.columns })
  requires cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2
  returns  ar: array u64
  ensures  cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2 ** A.pts_to ar (matmul v1 v2)
{
  open FStar.SizeT;
  let size = Defs.rows *^ Defs.columns;
  let ar = Pulse.Lib.Array.alloc 0UL size;

  let ga1 = gpu_array_alloc #u64 (Defs.rows *^ Defs.shared);
  let ga2 = gpu_array_alloc #u64 (Defs.shared *^ Defs.columns);

  GPU.Array.gpu_memcpy_host_to_device a1 ga1 (Defs.rows *^ Defs.shared);
  GPU.Array.gpu_memcpy_host_to_device a2 ga2 (Defs.shared *^ Defs.columns);

  let gr = gpu_array_alloc #u64 size;

  setup size ga1 ga2 gr v1 v2;

  launch_kernel_n #0
    size
    #(fun (tid: nat {0 <= tid /\ tid < size} ) -> Defs.kpre Defs.rows Defs.shared Defs.columns ga1 ga2 gr #v1 #v2 (SZ.v size) tid)
    #(fun (tid: nat {0 <= tid /\ tid < size} ) -> Defs.kpost Defs.rows Defs.shared Defs.columns ga1 ga2 gr #v1 #v2 (SZ.v size) tid)
    (fun etid -> Defs.kernel ga1 ga2 gr (hide size) etid);

  ghost
  fn aux0 (i:nat{0 <= i /\ i < size})
    requires
      Defs.kpost Defs.rows Defs.shared Defs.columns ga1 ga2 gr #v1 #v2 size i
    ensures
      Defs.gpu_pts_to_matrix Defs.rows   Defs.shared  ga1 size v1 **
      Defs.gpu_pts_to_matrix Defs.shared Defs.columns ga2 size v2 **
      gpu_pts_to_array_slice gr i (i + 1) seq![matmul_single v1 v2 (i / Defs.columns) (i % Defs.columns) Defs.shared]
  {
    unfold Defs.kpost;
  };
  bigstar_uneta();
  bigstar_map #_ #_ #0 #size aux0;

  (**)bigstar_unzip 0 size _ _;
  (**)bigstar_unzip 0 size _ _;

  (**)Defs.gpu_matrix_unshare_underspec #_ #_ Defs.rows Defs.shared ga1 size v1;
  (**)Defs.gpu_matrix_unshare_underspec #_ #_ Defs.shared Defs.columns ga2 size v2;
  (**)unfold Defs.gpu_pts_to_matrix Defs.rows Defs.shared ga1 1 v1;
  (**)unfold Defs.gpu_pts_to_matrix Defs.shared Defs.columns ga2 1 v2;

  ghost
  fn aux1 (i:nat{0 <= i /\ i < size})
    requires
      gpu_pts_to_array_slice gr i (i + 1)
            seq![matmul_single v1 v2 (i / Defs.columns) (i % Defs.columns) Defs.shared]
    ensures
      gpu_pts_to_array_slice gr i (i + 1)
            seq![Seq.index (Defs.matmul Defs.rows Defs.shared Defs.columns v1 v2) i]
  {
    Defs.lemma_matmul_index Defs.rows Defs.shared Defs.columns v1 v2 i;
    () (* cf. issue #181 in Pulse *)
  };
  bigstar_map #_ #_ #0 #size aux1;

  (**)gpu_array_unslice_1 #0 #_ #size gr #_ #(Defs.matmul Defs.rows Defs.shared Defs.columns v1 v2);

  GPU.Array.gpu_memcpy_device_to_host ar gr size;

  gpu_array_free ga1;
  gpu_array_free ga2;
  gpu_array_free gr;

  ar
}
