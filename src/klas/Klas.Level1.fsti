module Klas.Level1

(* Basic BLAS level-1 (vector) operations, instantiated for concrete element
   types, mirroring the cuBLAS API.

     scal_*   ~  cublas{S,D,H}scal    x := alpha * x
     axpy_*   ~  cublas{S,D,H}axpy    y := alpha * x + y
     copy_*   ~  cublas{S,D,H}copy    y := x

   Differences from cuBLAS: there is no stride (incx/incy) argument, so
   vectors are contiguous; there is no handle.  Pointers are device (GPU)
   pointers, as in cuBLAS.  Specifications are *exact* at the element type:
   e.g. [scal] yields exactly the elementwise (floating-point) product, with
   no approximation, since these are pure pointwise computations.

   We provide instances for f16/f32/f64 (cuBLAS H/S/D) and additionally for
   the unsigned integer types u32/u64, which cuBLAS does not offer.  Because
   scal/copy/swap need only the [scalar] class, the SAME verified kernels also
   instantiate at the complex scalar types cf32/cf64, giving cuBLAS
   Cscal/Zscal, Ccopy/Zcopy, Cswap/Zswap for free (they extract to
   cuFloatComplex / cuDoubleComplex arithmetic). *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Seq.Common { lseq_map }
open Kuiper.Complex32           (* cf32 -> cuFloatComplex *)
open Kuiper.Complex64           (* cf64 -> cuDoubleComplex *)
module Map = Kuiper.Kernel.Map

(* ----- SCAL: x := alpha * x ----- *)

let s_scal (#et:Type0) {| scalar et |} (#n:nat) (alpha:et) (s : lseq et n)
  : GTot (lseq et n)
  = lseq_map (fun (x:et) -> mul alpha x) s

inline_for_extraction noextract
type scal_ty (et:Type0) {| scalar et |} =
  fn (alpha : et)
     (lena : szp { lena <= max_blocks * max_threads })
     (a : array1 et (l1_forward lena) { is_global a })
     (#s : erased (lseq et lena))
  preserves cpu
  requires on gpu_loc (a |-> s)
  ensures  on gpu_loc (a |-> s_scal alpha s)

val scal_f16 : scal_ty f16
val scal_f32 : scal_ty f32
val scal_f64 : scal_ty f64
val scal_u32 : scal_ty u32
val scal_u64 : scal_ty u64
val scal_cf32 : scal_ty cf32   (* cuBLAS Cscal *)
val scal_cf64 : scal_ty cf64   (* cuBLAS Zscal *)

(* ----- AXPY: y := alpha * x + y ----- *)

let s_axpy (#et:Type0) {| scalar et |} (#n:nat) (alpha:et) (sx sy : lseq et n)
  : GTot (lseq et n)
  = Map.lseq_map2 (fun (yi xi : et) -> add (mul alpha xi) yi) sy sx

inline_for_extraction noextract
type axpy_ty (et:Type0) {| scalar et |} =
  fn (alpha : et)
     (lena : szp { lena <= max_blocks * max_threads })
     (y : array1 et (l1_forward lena) { is_global y })
     (x : array1 et (l1_forward lena) { is_global x })
     (#sy : erased (lseq et lena))
     (#sx : erased (lseq et lena))
     (#fx : perm)
  preserves cpu ** on gpu_loc (x |-> Frac fx sx)
  requires on gpu_loc (y |-> sy)
  ensures  on gpu_loc (y |-> s_axpy alpha sx sy)

val axpy_f16 : axpy_ty f16
val axpy_f32 : axpy_ty f32
val axpy_f64 : axpy_ty f64
val axpy_u32 : axpy_ty u32
val axpy_u64 : axpy_ty u64

(* ----- COPY: y := x ----- *)

inline_for_extraction noextract
type copy_ty (et:Type0) {| scalar et |} =
  fn (lena : szp { lena <= max_blocks * max_threads })
     (y : array1 et (l1_forward lena) { is_global y })
     (x : array1 et (l1_forward lena) { is_global x })
     (#sy : erased (lseq et lena))
     (#sx : erased (lseq et lena))
     (#fx : perm)
  preserves cpu ** on gpu_loc (x |-> Frac fx sx)
  requires on gpu_loc (y |-> sy)
  ensures  on gpu_loc (y |-> sx)

val copy_f16 : copy_ty f16
val copy_f32 : copy_ty f32
val copy_f64 : copy_ty f64
val copy_u32 : copy_ty u32
val copy_u64 : copy_ty u64
val copy_cf32 : copy_ty cf32   (* cuBLAS Ccopy *)
val copy_cf64 : copy_ty cf64   (* cuBLAS Zcopy *)

(* ----- SWAP: x <-> y ----- *)
(* Implemented out-of-place (temp + three device copies); cuBLAS swaps in a
   single pass, but the observable result is identical. *)

inline_for_extraction noextract
type swap_ty (et:Type0) {| scalar et |} =
  fn (lena : szp)
     (x : array1 et (l1_forward lena) { is_global x })
     (y : array1 et (l1_forward lena) { is_global y })
     (#sx : erased (lseq et lena))
     (#sy : erased (lseq et lena))
  preserves cpu
  requires on gpu_loc (x |-> sx) ** on gpu_loc (y |-> sy)
  ensures  on gpu_loc (x |-> sy) ** on gpu_loc (y |-> sx)

val swap_f16 : swap_ty f16
val swap_f32 : swap_ty f32
val swap_f64 : swap_ty f64
val swap_u32 : swap_ty u32
val swap_u64 : swap_ty u64
val swap_cf32 : swap_ty cf32   (* cuBLAS Cswap *)
val swap_cf64 : swap_ty cf64   (* cuBLAS Zswap *)
