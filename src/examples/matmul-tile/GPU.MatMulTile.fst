module GPU.MatMulTile
#lang-pulse

#push-options "--fuel 1 --ifuel 1"

open GPU

module A    = Pulse.Lib.Array
module SZ   = FStar.SizeT
module Defs = GPU.MatMul.Defs
module Kernel = GPU.MatMulTile.Kernel
module Barrier = GPU.MatMulTile.Barrier

ghost
fn setup
  (nblk: sz { nblk == SZ.(Kernel.rows_tile *^ Kernel.columns_tile) })
  (nthr: sz { nthr == SZ.(Kernel.bdim *^ Kernel.bdim) })
  (ga1 : gpu_array u64 (Kernel.rows * Kernel.shared))
  (ga2 : gpu_array u64 (Kernel.shared * Kernel.columns))
  (gr  : gpu_array u64 (nblk * nthr))
  (v1: erased (seq u64) { Seq.length v1 == Kernel.rows * Kernel.shared })
  (v2: erased (seq u64) { Seq.length v2 == Kernel.shared * Kernel.columns })
  requires gpu_pts_to_array gr 's **
           gpu_pts_to_array ga1 v1 **
           gpu_pts_to_array ga2 v2
  ensures  bigstar 0 (nblk * nthr) (fun i ->
             Kernel.kpre Kernel.rows Kernel.shared Kernel.columns ga1 ga2 gr #v1 #v2 (nblk * nthr) (Kernel.tid_to_idx i))
{
  // Sharing the input matrices (splitting permissions)
  fold Defs.gpu_pts_to_matrix Kernel.rows   Kernel.shared  ga1 1 v1;
  fold Defs.gpu_pts_to_matrix Kernel.shared Kernel.columns ga2 1 v2;
  Defs.gpu_matrix_share_underspec #_ #1 Kernel.rows   Kernel.shared  ga1 (nblk * nthr) v1;
  Defs.gpu_matrix_share_underspec #_ #2 Kernel.shared Kernel.columns ga2 (nblk * nthr) v2;

  // Sharing the output matrix (splitting each cell)
  gpu_pts_to_ref gr; (* obtain length v == (nblk * nthr) *)
  gpu_array_slice_1 #4 gr;

  // Join resources into a single bigstar
  bigstar_zip #1 #2 #3 0 (nblk * nthr) _ _;
  bigstar_zip #3 #4 #0 0 (nblk * nthr) _ _;

  // Rewrite inside the bigstar
  ghost
  fn aux (i:nat{0 <= i /\ i < (nblk * nthr)})
    requires
      Defs.gpu_pts_to_matrix Kernel.rows   Kernel.shared  ga1 (nblk * nthr) v1 **
      Defs.gpu_pts_to_matrix Kernel.shared Kernel.columns ga2 (nblk * nthr) v2 **
      gpu_pts_to_array_slice gr i (i + 1) seq!['s `Seq.index` i]
    ensures
      Kernel.kpre Kernel.rows Kernel.shared Kernel.columns ga1 ga2 gr #v1 #v2 (nblk * nthr) i
  {
    fold gpu_pts_to_array1 gr i;
    ()
  };
  bigstar_map #_ #_ #0 #(nblk * nthr) aux;
  bigstar_permute #0 #0 #(nblk * nthr) #_ (Kernel.permute());
  ghost fn rewrite_permute_to_fn (i: nat {0 <= i /\ i < (nblk * nthr)})
    requires Kernel.kpre Kernel.rows Kernel.shared Kernel.columns ga1 ga2 gr #v1 #v2 (nblk * nthr) ((Kernel.permute()).f i)
    ensures  Kernel.kpre Kernel.rows Kernel.shared Kernel.columns ga1 ga2 gr #v1 #v2 (nblk * nthr) (Kernel.tid_to_idx i)
  {
    ()
  };
  bigstar_map #0 #0 #0 #(nblk * nthr) (fun i -> rewrite_permute_to_fn i);
  ()
}

ghost
fn breakdown
  (nblk: sz { nblk == SZ.(Kernel.rows_tile *^ Kernel.columns_tile) })
  (nthr: sz { nthr == SZ.(Kernel.bdim *^ Kernel.bdim) })
  (ga1 : gpu_array u64 (Kernel.rows * Kernel.shared))
  (ga2 : gpu_array u64 (Kernel.shared * Kernel.columns))
  (gr  : gpu_array u64 (nblk * nthr))
  (v1: erased (seq u64) { Seq.length v1 == Kernel.rows * Kernel.shared })
  (v2: erased (seq u64) { Seq.length v2 == Kernel.shared * Kernel.columns })
  requires bigstar 0 (nblk * nthr) (fun i ->
             Kernel.kpost Kernel.rows Kernel.shared Kernel.columns ga1 ga2 gr #v1 #v2 (nblk * nthr) (Kernel.tid_to_idx i))
  ensures  (exists* vr. gpu_pts_to_array gr vr) **
           gpu_pts_to_array ga1 v1 **
           gpu_pts_to_array ga2 v2
{
  let perm = perm_inv (Kernel.permute());
  bigstar_permute #0 #0 #(nblk * nthr) #_ perm;
  ghost fn rewrite_permute_to_fn (i: nat {0 <= i /\ i < (nblk * nthr)})
    requires Kernel.kpost Kernel.rows Kernel.shared Kernel.columns ga1 ga2 gr #v1 #v2 (nblk * nthr) (perm.f (Kernel.tid_to_idx i))
    ensures  Kernel.kpost Kernel.rows Kernel.shared Kernel.columns ga1 ga2 gr #v1 #v2 (nblk * nthr) i
  {
    let once: (j: nat{ 0 <= j /\ j < (nblk * nthr) }) = Kernel.tid_to_idx i;
    let once': (j: nat{ 0 <= j /\ j < (nblk * nthr) }) = perm.g i;
    assert (pure (once' == once));
    perm.proof once i;
    // f x == y <==> g y == x
    let double: (j: nat{ 0 <= j /\ j < (nblk * nthr) }) = perm.f once;
    assert (pure (double == i));
    rewrite (gpu_pts_to_array1 gr (perm.f (Kernel.tid_to_idx i)))
         as (gpu_pts_to_array1 gr i);
    ()
  };
  bigstar_map #0 #0 #0 #(nblk * nthr) (fun i -> rewrite_permute_to_fn i);

  // Join resources into a single bigstar
  bigstar_unzip #3 #4 #0 0 (nblk * nthr) _ _;
  bigstar_unzip #1 #2 #3 0 (nblk * nthr) _ _;

  gpu_array_unslice_1_underspec #4 gr #1.0R;

  // Unsharing the input matrices (gathering permissions)
  Defs.gpu_matrix_unshare_underspec #_ #1 Kernel.rows   Kernel.shared  ga1 (nblk * nthr) v1;
  Defs.gpu_matrix_unshare_underspec #_ #2 Kernel.shared Kernel.columns ga2 (nblk * nthr) v2;
  unfold Defs.gpu_pts_to_matrix Kernel.rows   Kernel.shared  ga1 1 v1;
  unfold Defs.gpu_pts_to_matrix Kernel.shared Kernel.columns ga2 1 v2;

  ()
}

fn main
  (a1 a2: array u64)
  (v1: erased (seq u64) { Seq.length v1 == Kernel.rows * Kernel.shared })
  (v2: erased (seq u64) { Seq.length v2 == Kernel.shared * Kernel.columns })
  requires cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2
  returns  ar: array u64
  ensures  cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2 ** (exists* vr. A.pts_to ar vr)
{
  open FStar.SizeT;
  let nblk = Kernel.rows_tile *^ Kernel.columns_tile;
  let nthr = Kernel.bdim *^ Kernel.bdim;
  let size = nblk *^ nthr;
  let ar = Pulse.Lib.Array.alloc 0UL size;

  let ga1 = gpu_array_alloc #u64 (Kernel.rows *^ Kernel.shared);
  let ga2 = gpu_array_alloc #u64 (Kernel.shared *^ Kernel.columns);

  GPU.Array.gpu_memcpy_host_to_device a1 ga1 (Kernel.rows *^ Kernel.shared);
  GPU.Array.gpu_memcpy_host_to_device a2 ga2 (Kernel.shared *^ Kernel.columns);

  let gr = gpu_array_alloc #u64 size;

  setup nblk nthr ga1 ga2 gr v1 v2;

  let smem_sz = 2sz *^ nthr;
  launch_kernel_n_m_sync #0
    nblk
    nthr
    #(fun (tid: nat {0 <= tid /\ tid < (nblk * nthr)} ) -> Kernel.kpre Kernel.rows Kernel.shared Kernel.columns ga1 ga2 gr #v1 #v2 (nblk * nthr) (Kernel.tid_to_idx tid))
    #(fun (tid: nat {0 <= tid /\ tid < (nblk * nthr)} ) -> Kernel.kpost Kernel.rows Kernel.shared Kernel.columns ga1 ga2 gr #v1 #v2 (nblk * nthr) (Kernel.tid_to_idx tid))
    u64
    smem_sz
    #(Barrier.shared_pre nthr 0)
    #(Barrier.shared_pre nthr (2 * Kernel.columns_tile))
    (Barrier.block_setup_ghost nthr smem_sz)
    (fun ear etid -> Kernel.kernel ga1 ga2 gr #v1 #v2 (hide nblk) (hide nthr) smem_sz ear etid);

  breakdown nblk nthr ga1 ga2 gr v1 v2;

  GPU.Array.gpu_memcpy_device_to_host ar gr size;

  gpu_array_free ga1;
  gpu_array_free ga2;
  gpu_array_free gr;

  ar
}
