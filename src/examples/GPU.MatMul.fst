module GPU.MatMul
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
open GPU
module Defs = GPU.MatMul.Defs

let matmul_single = Defs.matmul_single Defs.rows Defs.shared Defs.columns
let matmul = Defs.matmul Defs.rows Defs.shared Defs.columns

// #push-options "--print_implicits --print_bound_var_types"

ghost
fn setup
  (size: SZ.t { size == SZ.(Defs.rows *^ Defs.columns) })
  (ga1 : gpu_array U64.t (Defs.rows * Defs.shared)) (ga2 : gpu_array U64.t (Defs.shared * Defs.columns)) (gr : gpu_array U64.t size)
  (v1: erased (Seq.Base.seq U64.t) { Seq.Base.length v1 == Defs.rows * Defs.shared })
  (v2: erased (Seq.Base.seq U64.t) { Seq.Base.length v2 == Defs.shared * Defs.columns })
  requires (exists* s. gpu_pts_to_array gr s) ** gpu_pts_to_array ga1 v1 ** gpu_pts_to_array ga2 v2
  ensures  bigstar 0 size (fun i -> Defs.kpre Defs.rows Defs.shared Defs.columns ga1 ga2 gr #v1 #v2 (hide (SZ.v size)) i)
{
  // Slicing the array
  // rewrite gpu_pts_to_array ga1 #1.0R v1
  //   as gpu_pts_to_array ga1 #(1.0R /. of_int 1) v1;
  // rewrite gpu_pts_to_array ga2 #1.0R v2
  //   as gpu_pts_to_array ga2 #(1.0R /. of_int 1) v2;

  (**)fold Defs.gpu_pts_to_matrix Defs.rows   Defs.shared  ga1 1 v1;
  (**)fold Defs.gpu_pts_to_matrix Defs.shared Defs.columns ga2 1 v2;
  (**)Defs.gpu_matrix_share_underspec #_ #1 (SZ.v Defs.rows) (SZ.v Defs.shared) ga1 (SZ.v size) v1;
  (**)Defs.gpu_matrix_share_underspec #_ #2 Defs.shared Defs.columns ga2 (SZ.v size) v2;

  // Boring combination of resources
  (**)bigstar_zip #1 #2 #3 0 size _ _;
  (**)bigstar_map #3 #3 #0 #size #_ #_
       (fun i -> Defs.fold_pre_pair Defs.rows Defs.shared Defs.columns ga1 ga2 #v1 #v2 size i);

  with #f v. assert (gpu_pts_to_array gr #f v);
  assume_ (pure (Seq.length v == size)); // FIXME

  (**)gpu_array_slice_1 #4 #_ gr #f #v;
  (**)bigstar_zip #3 #4 #5 0 size _ _;
  (**)bigstar_map #5 #0 #0 #size #_ #_
        (fun i -> Defs.fold_pre Defs.rows Defs.shared Defs.columns ga1 ga2 gr #v1 #v2 #(Seq.Base.cons #U64.t _ (Seq.Base.empty #U64.t)) size i);

  bigstar_eta();
}

fn main
  (a1 a2: array U64.t)
  (v1: erased (Seq.Base.seq U64.t) { Seq.Base.length v1 == Defs.rows * Defs.shared })
  (v2: erased (Seq.Base.seq U64.t) { Seq.Base.length v2 == Defs.shared * Defs.columns })
  requires cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2
  returns ar: array U64.t
  ensures  cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2 ** A.pts_to ar (matmul v1 v2)
{
  open FStar.SizeT;
  let size = Defs.rows *^ Defs.columns;
  let ar = Pulse.Lib.Array.alloc #U64.t 0UL size;

  let ga1 = gpu_array_alloc #U64.t (Defs.rows *^ Defs.shared);
  let ga2 = gpu_array_alloc #U64.t (Defs.shared *^ Defs.columns);

  GPU.Array.gpu_memcpy_host_to_device a1 ga1 (Defs.rows *^ Defs.shared);
  GPU.Array.gpu_memcpy_host_to_device a2 ga2 (Defs.shared *^ Defs.columns);
  
  let gr = gpu_array_alloc #U64.t size;

  setup size ga1 ga2 gr v1 v2;

  launch_kernel_n #0
    size
    #(fun (tid: nat {0 <= tid /\ tid < size} ) -> Defs.kpre Defs.rows Defs.shared Defs.columns ga1 ga2 gr #v1 #v2 (SZ.v size) tid)
    #(fun (tid: nat {0 <= tid /\ tid < size} ) -> Defs.kpost Defs.rows Defs.shared Defs.columns ga1 ga2 gr #v1 #v2 (SZ.v size) tid)
    (fun (etid: erased tid_t {(gdim_x etid) == size /\ bdim_x etid == 1sz} ) -> Defs.kernel ga1 ga2 gr (hide size) etid);

  bigstar_uneta();

  (**)bigstar_map #0 #5 #0 #size #_ #_
        (fun i -> Defs.unfold_post Defs.rows Defs.shared Defs.columns ga1 ga2 gr #v1 #v2 size i);

  // (**)rw_assume
  //   (bigstar 0 size (kpre ga1 ga2 gr size))
  //   (bigstar 0 size
  //     (fun i -> gpu_pts_to_array1 gr i **
  //               gpu_pts_to_array1 ga2 i **
  //               gpu_pts_to_array1 ga1 i));
  (**)bigstar_unzip 0 size _ _;
  (**)bigstar_unzip 0 size _ _;

  (**)Defs.gpu_matrix_unshare_underspec #_ #_ Defs.rows Defs.shared ga1 size v1;
  (**)Defs.gpu_matrix_unshare_underspec #_ #_ Defs.shared Defs.columns ga2 size v2;
  (**)unfold Defs.gpu_pts_to_matrix Defs.rows Defs.shared ga1 1 v1;
  (**)unfold Defs.gpu_pts_to_matrix Defs.shared Defs.columns ga2 1 v2;


  // // Unslicing
  // (**)gpu_array_unslice_1_underspec ga1;
  // (**)gpu_array_unslice_1_underspec ga2;

  // Defs.lemma_matmul_index

  bigstar_rw_congr 0 size 
        (fun (i: nat{b2t (0 <= i) /\ b2t (i < size)}) ->
          gpu_pts_to_array_slice #U64.t
            #(Defs.rows * Defs.columns)
            gr
            #1.0R
            i
            (i + 1)
            (Defs.singleton #U64.t
                (reveal #U64.t
                    (hide (Defs.matmul_single Defs.rows
                        Defs.shared
                        Defs.columns
                        v1
                        v2
                        (hide #nat (i / Defs.columns))
                        (hide #nat (i % Defs.columns))
                        Defs.shared)))))
        (fun i -> gpu_pts_to_array_slice #U64.t #size gr #1.0R i
            (i + 1)
            (Defs.singleton #U64.t
                (reveal #U64.t
                    (hide (Seq.Base.index #U64.t
                      (reveal #(Seq.Base.seq U64.t)
                          (hide #(Seq.Base.seq U64.t) (Defs.matmul Defs.rows Defs.shared Defs.columns v1 v2)))
                      i)))))
        (fun i -> Defs.lemma_matmul_index Defs.rows Defs.shared Defs.columns v1 v2 i);

  (**)gpu_array_unslice_1 #0 #_ #size gr #_ #(Defs.matmul Defs.rows Defs.shared Defs.columns v1 v2);
  
  GPU.Array.gpu_memcpy_device_to_host ar gr size;

  gpu_array_free ga1;
  gpu_array_free ga2;
  gpu_array_free gr;

  ar
}
