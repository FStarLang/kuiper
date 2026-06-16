module Klas.Caxpy

(* Complex axpy (cuBLAS caxpy / zaxpy): y := alpha*x + y.

   There is ONE generic, verified element-wise map kernel (caxpy_gen, over any
   [scalar]); the per-precision entry points are one-liners that instantiate it
   at the complex scalar instances. cf32 extracts to cuFloatComplex (cuCaddf /
   cuCmulf), cf64 to cuDoubleComplex (cuCadd / cuCmul). This is the same kernel
   used for the real BLAS-1 ops (Klas.Level1): "complex BLAS for free". *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Complex32           (* cf32 + its scalar instance *)
open Kuiper.Complex64           (* cf64 + its scalar instance *)
module Map = Kuiper.Kernel.Map

(* Functional spec: elementwise y[i] := alpha * x[i] + y[i]. *)
let s_caxpy (#et:Type0) {| scalar et |} (#n:nat) (alpha:et) (sx sy : lseq et n)
  : GTot (lseq et n)
  = Map.lseq_map2 (fun (yi xi : et) -> add (mul alpha xi) yi) sy sx

inline_for_extraction noextract
fn caxpy_gen (#et:Type0) {| scalar et |}
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
  ensures  on gpu_loc (y |-> s_caxpy alpha sx sy)
{
  Map.map_gpu2 (fun (yi xi : et) -> add (mul alpha xi) yi) lena y x;
}

(* Per-precision entry points: the same kernel at different scalar instances. *)
let caxpy = caxpy_gen #cf32   (* -> cuFloatComplex  *)
let zaxpy = caxpy_gen #cf64   (* -> cuDoubleComplex *)
