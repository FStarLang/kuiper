module GPU.MatMul

#push-options "--fuel 1 --ifuel 1"

// #push-options "--debug SMTFail --split_queries always --log_failing_queries"

#push-options "--print_implicits"

let admit_assume(p: prop) : Lemma (p) = admit()

open FStar.Mul
open Pulse.Lib.Array
open Pulse.Lib.Pervasives
module A = Pulse.Lib.Array
module SZ = FStar.SizeT
open Pulse.Lib.BigStar
open GPU
module Defs = GPU.MatMul.Defs

let matmul_single = Defs.matmul_single Defs.rows Defs.shared Defs.columns
let matmul = Defs.matmul Defs.rows Defs.shared Defs.columns

// #push-options "--print_implicits"

```pulse
fn main
  (a1 a2: array int)
  (v1: erased (Seq.Base.seq int) { Seq.Base.length v1 == Defs.rows * Defs.shared })
  (v2: erased (Seq.Base.seq int) { Seq.Base.length v2 == Defs.shared * Defs.columns })
  requires cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2
  returns ar: array int
  ensures  cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2 ** A.pts_to ar (matmul v1 v2)
{
  let size = Defs.rows * Defs.columns;
  let ar = Pulse.Lib.Box.alloc();

  let ga1 = gpu_array_alloc #int (Defs.rows * Defs.shared);
  let ga2 = gpu_array_alloc #int (Defs.shared * Defs.columns);

  GPU.Array.gpu_memcpy_host_to_device a1 ga1;
  GPU.Array.gpu_memcpy_host_to_device a2 ga2;
  
  let gr = gpu_array_alloc #int size;

  // Slicing the array
  rewrite gpu_pts_to_array ga1 #1.0R v1
    as gpu_pts_to_array ga1 #(1.0R /. of_int 1) v1;
  rewrite gpu_pts_to_array ga2 #1.0R v2
    as gpu_pts_to_array ga2 #(1.0R /. of_int 1) v2;

  (**)fold Defs.gpu_pts_to_matrix Defs.rows Defs.shared ga1 1 v1;
  (**)fold Defs.gpu_pts_to_matrix Defs.shared Defs.columns ga2 1 v2;
  (**)Defs.gpu_matrix_share_underspec #_ #1 Defs.rows Defs.shared ga1 size v1;
  (**)Defs.gpu_matrix_share_underspec #_ #2 Defs.shared Defs.columns ga2 size v2;

  // Boring combination of resources
  (**)bigstar_zip #1 #2 #3 0 size _ _;
  (**)bigstar_map #3 #0 #size #_ #_ (Defs.fold_pre_pair Defs.rows Defs.shared Defs.columns ga1 ga2 #v1 #v2 size);

  (**)gpu_array_slice_1 #_ #4 gr;
  (**)bigstar_zip #3 #4 #5 0 size _ _;
  (**)bigstar_map #5 #0 #size #_ #_ (Defs.fold_pre Defs.rows Defs.shared Defs.columns ga1 ga2 gr #v1 #v2 #_ size);

  admit();

  launch_kernel_n
    size
    #(Defs.kpre Defs.rows Defs.shared Defs.columns ga1 ga2 gr #v1 #v2 #_ size)
    #_
    (Defs.kernel ga1 ga2 gr size);

  admit();

  // (**)rw_assume
  //   (bigstar 0 size (kpre ga1 ga2 gr size))
  //   (bigstar 0 size
  //     (fun i -> gpu_pts_to_array1 gr i **
  //               gpu_pts_to_array1 ga2 i **
  //               gpu_pts_to_array1 ga1 i));
  // (**)bigstar_unzip 0 size _ _;
  // (**)bigstar_unzip 0 size _ _;
  
  // // Unslicing
  // (**)gpu_array_unslice_1_underspec ga1;
  // (**)gpu_array_unslice_1_underspec ga2;
  // (**)gpu_array_unslice_1_underspec gr;
  
  // GPU.Array.gpu_memcpy_device_to_host ar gr;
  // gpu_array_free ga1;
  // gpu_array_free ga2;
  // gpu_array_free gr;

  // let mut i = 0sz;
  // let mut psum = 0;
  // while (let v = !i; (SZ.v v < size))
  //    invariant b. exists* vi vpsum.
  //      pts_to i vi **
  //      pts_to psum vpsum  **
  //      (exists* s. A.pts_to ar s ** pure (Seq.length s == size)) **
  //      pure (b == (SZ.v vi < size))
  // {
  //   let vi = !i;
  //   let ri = ar.(vi);
  //   let vpsum = !psum;
  //   psum := vpsum + ri;
  //   i := SZ.add vi 1sz;
  // };
  
  // A.free a1;
  // A.free a2;
  // A.free ar;

  ar
}
```
