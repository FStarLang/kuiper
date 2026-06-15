module Klas.Caxpy

(* cuBLAS caxpy: y := alpha*x + y over single-precision complex.

   This is the *same* verified element-wise map kernel used for the real
   BLAS-1 ops (Klas.Level1), now instantiated at the complex scalar instance
   Kuiper.Complex32. Because Kuiper.Complex32.Base.t is a [scalar], the generic
   kernel goes through unchanged and extracts to CUDA's cuFloatComplex
   arithmetic (cuCaddf / cuCmulf). It demonstrates "complex BLAS for free". *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Complex32           (* brings cf32 and its scalar instance *)
module Map = Kuiper.Kernel.Map
module C = Kuiper.Complex32.Base

(* Functional spec: elementwise y[i] := alpha * x[i] + y[i] (complex). *)
let s_caxpy (#n:nat) (alpha:C.t) (sx sy : lseq C.t n) : GTot (lseq C.t n)
  = Map.lseq_map2 (fun (yi xi : C.t) -> add (mul alpha xi) yi) sy sx

inline_for_extraction noextract
fn caxpy_gen
  (alpha : C.t)
  (lena : szp { lena <= max_blocks * max_threads })
  (y : array1 C.t (l1_forward lena) { is_global y })
  (x : array1 C.t (l1_forward lena) { is_global x })
  (#sy : erased (lseq C.t lena))
  (#sx : erased (lseq C.t lena))
  (#fx : perm)
  norewrite
  preserves cpu ** on gpu_loc (x |-> Frac fx sx)
  requires on gpu_loc (y |-> sy)
  ensures  on gpu_loc (y |-> s_caxpy alpha sx sy)
{
  Map.map_gpu2 (fun (yi xi : C.t) -> add (mul alpha xi) yi) lena y x;
}

let caxpy = caxpy_gen
