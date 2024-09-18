module Kuiper.MatMulOpt

#lang-pulse

#push-options "--fuel 1 --ifuel 1"

// #push-options "--debug SMTFail --split_queries always --log_failing_queries"

open FStar.Mul
open Pulse.Lib.Array
open Pulse.Lib.Pervasives
module A = Pulse.Lib.Array
module SZ = FStar.SizeT
module U64 = FStar.UInt64
open Pulse.Lib.BigStar
open Kuiper
module Defs = Kuiper.MatMulOpt.Defs

let matmul_single = Defs.matmul_single Defs.rows Defs.shared Defs.columns
let matmul = Defs.matmul Defs.rows Defs.shared Defs.columns

ghost
fn setup
  (size: sz { size == SZ.(Defs.rows *^ Defs.columns) })
  (ga1 : gpu_array u64 (Defs.rows * Defs.shared)) (ga2 : gpu_array u64 (Defs.shared * Defs.columns)) (gr : gpu_array u64 size)
  (v1: erased (seq u64) { Seq.length v1 == Defs.rows * Defs.shared })
  (v2: erased (seq u64) { Seq.length v2 == Defs.shared * Defs.columns })
  requires cpu ** (exists* s. gpu_pts_to_array gr s) ** gpu_pts_to_array ga1 v1 ** gpu_pts_to_array ga2 v2
  ensures cpu ** bigstar 0 size (Defs.kpre Defs.shared Defs.rows Defs.columns ga1 ga2 gr #v1 #v2 (hide (SZ.v size)))
{
  admit();
  // Slicing the array
  rewrite gpu_pts_to_array ga1 #1.0R v1
    as gpu_pts_to_array ga1 #(1.0R /. of_int 1) v1;
  rewrite gpu_pts_to_array ga2 #1.0R v2
    as gpu_pts_to_array ga2 #(1.0R /. of_int 1) v2;

  (**)fold Defs.gpu_pts_to_matrix Defs.rows Defs.shared ga1 1 v1;
  (**)fold Defs.gpu_pts_to_matrix Defs.shared Defs.columns ga2 1 v2;
  (**)Defs.gpu_matrix_share_underspec #_ #1 Defs.rows Defs.shared ga1 (SZ.v size) v1;
  (**)Defs.gpu_matrix_share_underspec #_ #2 Defs.shared Defs.columns ga2 (SZ.v size) v2;

  // Boring combination of resources
  (**)bigstar_zip #1 #2 #3 0 size _ _;
  (**)bigstar_map #3 #3 #0 #size #_ #_
       (fun i -> Defs.fold_pre_pair Defs.rows Defs.shared Defs.columns ga1 ga2 #v1 #v2 size i);

  with #f v. assert (gpu_pts_to_array gr #f v);
  assume (pure (Seq.length v == size)); // FIXME

  (**)gpu_array_slice_1 #4 #_ gr #f #v;
  (**)bigstar_zip #3 #4 #5 0 size _ _;
  (**)bigstar_map #5 #0 #0 #size #_ #_
        (fun i -> Defs.fold_pre Defs.shared Defs.rows Defs.columns ga1 ga2 gr #v1 #v2 #(Seq.cons #u64 _ (Seq.empty #u64)) size i);
}

#push-options "--print_implicits --print_bound_var_types"

fn main
  (a1 a2: array u64)
  (v1: erased (seq u64) { Seq.length v1 == Defs.rows * Defs.shared })
  (v2: erased (seq u64) { Seq.length v2 == Defs.shared * Defs.columns })
  requires cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2
  returns ar: array u64
  ensures  cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2 ** A.pts_to ar (matmul v1 v2)
{
  open FStar.SizeT;
  let size: sz = Defs.rows *^ Defs.columns;
  let ar = Pulse.Lib.Array.alloc #u64 0UL size;

  let ga1 = gpu_array_alloc #u64 (Defs.rows *^ Defs.shared);
  let ga2 = gpu_array_alloc #u64 (Defs.shared *^ Defs.columns);

  Kuiper.Array.gpu_memcpy_host_to_device ga1 a1 (Defs.rows *^ Defs.shared);
  Kuiper.Array.gpu_memcpy_host_to_device ga2 a2 (Defs.shared *^ Defs.columns);
  
  let gr = gpu_array_alloc #u64 size;

  setup size ga1 ga2 gr v1 v2;

  let nthr = Defs.tpb;
  let nblk = SZ.div size nthr;

  let smem_sz = 2sz *^ nthr;

  rewrite (bigstar 0 (sizet_to_nat size)     (Defs.kpre Defs.shared Defs.rows Defs.columns ga1 ga2 gr #v1 #v2 (hide (SZ.v size))))
       as (bigstar 0 (SZ.v nblk * SZ.v nthr) (Defs.kpre Defs.shared Defs.rows Defs.columns ga1 ga2 gr #v1 #v2 (SZ.v size)));

  bigstar_eta();

  // admit();
  Defs.mapping_inv_lemma (SZ.v nblk) (SZ.v nthr);
  // TODO: fix this assume
  assume (pure (SZ.v nblk * SZ.v nthr == SZ.v Defs.blocksize * SZ.v Defs.rows * SZ.v Defs.columns));
  let perm: erased (permutation (i: erased nat{0 <= i /\ i < (SZ.v nblk * SZ.v nthr)})) = Defs.mapping_fixed;
  bigstar_permute #0 #0 #(SZ.v nblk * SZ.v nthr) #(Defs.kpre Defs.shared Defs.rows Defs.columns ga1 ga2 gr #v1 #v2 (SZ.v size)) perm;

  launch_kernel_n_m_sync #0
    nblk
    nthr
    #(fun (tid: nat {0 <= tid /\ tid < (SZ.v nblk * SZ.v nthr)} ) -> Defs.kpre Defs.shared Defs.rows Defs.columns ga1 ga2 gr #v1 #v2 (SZ.v size) (Defs.mapping_fixed.f tid))
    #(fun (tid: nat {0 <= tid /\ tid < (SZ.v nblk * SZ.v nthr)} ) -> Defs.kpost Defs.shared Defs.rows Defs.columns ga1 ga2 gr #v1 #v2 (SZ.v size) (Defs.mapping_fixed.f tid))
    u64
    smem_sz
    #(Defs.shared_pre nthr 0)
    #(Defs.shared_post nthr)
    (Defs.block_setup_ghost nblk nthr smem_sz)
    ((fun (ar: gpu_array u64 smem_sz) (etid: erased tid_t {gdim_x etid == nblk /\ bdim_x etid == nthr} ) ->
      Defs.kernel ga1 ga2 gr #v1 #v2 (hide nblk) (hide nthr) (hide size) ar etid));

  bigstar_uneta();

  rewrite (bigstar #0 0 (SZ.v nblk * SZ.v nthr)
    (Defs.kpost Defs.shared Defs.rows Defs.columns ga1 ga2 gr #v1 #v2 (SZ.v size)))
      as  (bigstar #0 0 size
    (Defs.kpost Defs.shared Defs.rows Defs.columns ga1 ga2 gr #v1 #v2 (SZ.v size)));

  drop_   (bigstar #0 0 size (Defs.kpost Defs.shared Defs.rows Defs.columns ga1 ga2 gr #v1 #v2 (SZ.v size)));
  assume (bigstar #0 0 size (fun idx -> Defs.gpu_pts_to_matrix Defs.rows Defs.shared ga1 (SZ.v size) v1
                            ** Defs.gpu_pts_to_matrix Defs.shared Defs.columns ga2 (SZ.v size) v2
                            ** gpu_pts_to_array_slice gr idx (idx+1) (Defs.singleton (Defs.matmul_single Defs.rows Defs.shared Defs.columns v1 v2 (idx / Defs.columns) (idx % Defs.columns) Defs.shared))));

  // (**)bigstar_map #0 #5 #0 #size #_ #_
  //       (fun i -> Defs.unfold_post Defs.shared Defs.rows Defs.columns ga1 ga2 gr #v1 #v2 size i);

  // (**)rw_assume
  //   (bigstar 0 size (kpre ga1 ga2 gr size))
  //   (bigstar 0 size
  //     (fun i -> gpu_pts_to_array1 gr i **
  //               gpu_pts_to_array1 ga2 i **
  //               gpu_pts_to_array1 ga1 i));
  (**)bigstar_unzip 0 size _ _;
  (**)bigstar_unzip 0 size _ _;

  (**)Defs.gpu_matrix_unshare_underspec #_ #_ Defs.rows Defs.shared ga1 (hide #pos (SZ.v size)) v1;
  (**)Defs.gpu_matrix_unshare_underspec #_ #_ Defs.shared Defs.columns ga2 (hide #pos (SZ.v size)) v2;
  (**)unfold Defs.gpu_pts_to_matrix Defs.rows Defs.shared ga1 1 v1;
  (**)unfold Defs.gpu_pts_to_matrix Defs.shared Defs.columns ga2 1 v2;


  // // Unslicing
  // (**)gpu_array_unslice_1_underspec ga1;
  // (**)gpu_array_unslice_1_underspec ga2;

  // Defs.lemma_matmul_index

  // admit();
  bigstar_rw_congr 0 size 
        (fun (i: nat{b2t (0 <= i) /\ b2t (i < size)}) ->
          gpu_pts_to_array_slice #u64
            #(Defs.rows * Defs.columns)
            gr
            #1.0R
            i
            (i + 1)
            (Defs.singleton #u64
                (reveal #u64
                    (hide (Defs.matmul_single Defs.rows
                        Defs.shared
                        Defs.columns
                        v1
                        v2
                        (hide #nat (i / Defs.columns))
                        (hide #nat (i % Defs.columns))
                        Defs.shared)))))
        (fun i -> gpu_pts_to_array_slice #u64 #size gr #1.0R i
            (i + 1)
            (Defs.singleton #u64
                (reveal #u64
                    (hide (Seq.index #u64
                      (reveal #(seq u64)
                          (hide #(seq u64) (Defs.matmul Defs.rows Defs.shared Defs.columns v1 v2)))
                      i)))))
        (fun i -> Defs.lemma_matmul_index Defs.rows Defs.shared Defs.columns v1 v2 i);

  (**)gpu_array_unslice_1 #0 #_ #size gr #_ #(Defs.matmul Defs.rows Defs.shared Defs.columns v1 v2);
  
  Kuiper.Array.gpu_memcpy_device_to_host ar gr size;

  (**)rewrite (gpu_pts_to_array #u64
      #(SZ.v MatMulOpt.Kernel.rows * SZ.v MatMulOpt.Kernel.shared)
      ga1
      #(1.0R /. 1.0R)
      (reveal #(seq u64) v1)) as
      (gpu_pts_to_array #u64
      #(SZ.v (MatMulOpt.Kernel.rows *^ MatMulOpt.Kernel.shared))
      ga1
      #1.0R
      (reveal #(seq u64) v1));
  (**)rewrite (gpu_pts_to_array #u64
      #(SZ.v MatMulOpt.Kernel.shared * SZ.v MatMulOpt.Kernel.columns)
      ga2
      #(1.0R /. 1.0R)
      (reveal #(seq u64) v2)) as
      (gpu_pts_to_array #u64
      #(SZ.v (MatMulOpt.Kernel.shared *^ MatMulOpt.Kernel.columns))
      ga2
      #1.0R
      (reveal #(seq u64) v2));

  gpu_array_free ga1;
  gpu_array_free ga2;
  gpu_array_free gr;

  ar
}
