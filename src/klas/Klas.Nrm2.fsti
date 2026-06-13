module Klas.Nrm2

(* BLAS level-1 nrm2: Euclidean norm ‖x‖₂ = sqrt(Σ xᵢ²).
   Corresponds to cublasSnrm2/Dnrm2 (real element types).

   Implemented by squaring x elementwise into a scratch array (copy x, then
   in-place pointwise multiply by x), summing with the verified parallel
   reduction Kuiper.Kernel.HReduce.reduce, and taking the floating-point sqrt.

   As in cuBLAS the input is a device pointer; the spec is the real-valued
   norm that the floating-point result approximates. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Seq.Common { seq_map }
open Kuiper.Tensor.Layout.Alg { l1_forward }
module SZ = Kuiper.SizeT
module Map = Kuiper.Kernel.Map

(* Spec: the real Euclidean norm sqrt(Σ rxᵢ²). *)
let s_nrm2 (#n:nat) (rx : lseq real n) : real =
  realsqrt (rsum (Map.lseq_map2 ( *. ) rx rx))

inline_for_extraction noextract
type nrm2_ty (et : Type0) {| floating et, real_like et, floating_real_like et |} =
  fn (nth : szp { nth <= max_threads })
     (lena : szp { lena <= max_blocks * max_threads /\ SZ.fits (lena + nth) })
     (x : array1 et (l1_forward lena) { is_global x })
     (#vx : erased (lseq et lena))
     (#fx : perm)
     (rx : erased (lseq real lena))
  preserves
    cpu ** on gpu_loc (x |-> Frac fx vx)
  requires
    pure (vx %~ rx)
  returns
    res : et
  ensures
    pure (res %~ s_nrm2 rx)

val nrm2_f16 : nrm2_ty f16
val nrm2_f32 : nrm2_ty f32
val nrm2_f64 : nrm2_ty f64
