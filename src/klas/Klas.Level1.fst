module Klas.Level1

(* Concrete instantiations of the basic BLAS level-1 (vector) operations,
   mirroring the cuBLAS API.  Each operation is implemented by reusing the
   verified pointwise-map kernels in [Kuiper.Kernel.Map]; see the interface
   [Klas.Level1.fsti] for the (exact) specifications. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Complex32           (* cf32 + its scalar instance -> cuFloatComplex *)
open Kuiper.Complex64           (* cf64 + its scalar instance -> cuDoubleComplex *)
module Map = Kuiper.Kernel.Map

(* SCAL: x := alpha * x. Corresponds to cublasSscal/Dscal/... *)
inline_for_extraction noextract
fn scal_gen (#et:Type0) {| scalar et |}
  (alpha : et)
  (lena : szp { lena <= max_blocks * max_threads })
  (a : array1 et (l1_forward lena) { is_global a })
  (#s : erased (lseq et lena))
  preserves cpu
  requires on gpu_loc (a |-> s)
  ensures  on gpu_loc (a |-> s_scal alpha s)
{
  Map.map_gpu (fun (x:et) -> mul alpha x) lena a;
}

let scal_f16 = scal_gen #f16
let scal_f32 = scal_gen #f32
let scal_f64 = scal_gen #f64
let scal_u32 = scal_gen #u32
let scal_u64 = scal_gen #u64
let scal_cf32 = scal_gen #cf32   (* cuBLAS Cscal: complex alpha, complex x *)
let scal_cf64 = scal_gen #cf64   (* cuBLAS Zscal *)

(* AXPY: y := alpha * x + y. Corresponds to cublasSaxpy/Daxpy/... *)
inline_for_extraction noextract
fn axpy_gen (#et:Type0) {| scalar et |}
  (alpha : et)
  (lena : szp { lena <= max_blocks * max_threads })
  (y : array1 et (l1_forward lena) { is_global y })
  (x : array1 et (l1_forward lena) { is_global x })
  (#sy : erased (lseq et lena))
  (#sx : erased (lseq et lena))
  (#fx : perm)
  norewrite
  preserves cpu ** on gpu_loc (x |-> Frac fx sx)
  requires on gpu_loc (y |-> sy)
  ensures  on gpu_loc (y |-> s_axpy alpha sx sy)
{
  Map.map_gpu2 (fun (yi xi : et) -> add (mul alpha xi) yi) lena y x;
}

let axpy_f16 = axpy_gen #f16
let axpy_f32 = axpy_gen #f32
let axpy_f64 = axpy_gen #f64
let axpy_u32 = axpy_gen #u32
let axpy_u64 = axpy_gen #u64

(* COPY: y := x. Corresponds to cublasScopy/Dcopy/... *)
inline_for_extraction noextract
fn copy_gen (#et:Type0) {| scalar et |}
  (lena : szp { lena <= max_blocks * max_threads })
  (y : array1 et (l1_forward lena) { is_global y })
  (x : array1 et (l1_forward lena) { is_global x })
  (#sy : erased (lseq et lena))
  (#sx : erased (lseq et lena))
  (#fx : perm)
  norewrite
  preserves cpu ** on gpu_loc (x |-> Frac fx sx)
  requires on gpu_loc (y |-> sy)
  ensures  on gpu_loc (y |-> sx)
{
  Map.map_gpu2 (fun (_yi xi : et) -> xi) lena y x;
  assert pure (Seq.equal (Map.lseq_map2 (fun (_yi xi : et) -> xi) sy sx) (reveal sx));
}

let copy_f16 = copy_gen #f16
let copy_f32 = copy_gen #f32
let copy_f64 = copy_gen #f64
let copy_u32 = copy_gen #u32
let copy_u64 = copy_gen #u64
let copy_cf32 = copy_gen #cf32   (* cuBLAS Ccopy *)
let copy_cf64 = copy_gen #cf64   (* cuBLAS Zcopy *)

(* SWAP: x <-> y, via a temporary and three device-to-device copies.
   Corresponds to cublasSswap/Dswap/... *)
inline_for_extraction noextract
fn swap_gen (#et:Type0) {| scalar et |}
  (lena : szp)
  (x : array1 et (l1_forward lena) { is_global x })
  (y : array1 et (l1_forward lena) { is_global y })
  (#sx : erased (lseq et lena))
  (#sy : erased (lseq et lena))
  preserves cpu
  requires on gpu_loc (x |-> sx) ** on gpu_loc (y |-> sy)
  ensures  on gpu_loc (x |-> sy) ** on gpu_loc (y |-> sx)
{
  let tmp = alloc0 #et lena (l1_forward lena);
  memcpy_device_to_device tmp x lena;
  memcpy_device_to_device x y lena;
  memcpy_device_to_device y tmp lena;
  free tmp;
}

let swap_f16 = swap_gen #f16
let swap_f32 = swap_gen #f32
let swap_f64 = swap_gen #f64
let swap_u32 = swap_gen #u32
let swap_u64 = swap_gen #u64
let swap_cf32 = swap_gen #cf32   (* cuBLAS Cswap *)
let swap_cf64 = swap_gen #cf64   (* cuBLAS Zswap *)
