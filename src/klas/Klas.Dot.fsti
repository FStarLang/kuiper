module Klas.Dot

(* BLAS level-1 dot product: res = Σ xᵢ·yᵢ.
   Corresponds to cublasSdot/Ddot (real element types).

   Implemented by multiplying x and y elementwise into a scratch array (reusing
   the verified copy + pointwise-map kernels) and then summing with the verified
   parallel reduction Kuiper.Kernel.HReduce.reduce.

   As in cuBLAS the inputs are device pointers; unlike cuBLAS there is no stride
   argument and the spec is the real-valued dot product that the floating-point
   result approximates. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Seq.Common { seq_map }
open Kuiper.Tensor.Layout.Alg { l1_forward }
module SZ = Kuiper.SizeT
module Map = Kuiper.Kernel.Map

(* Spec: the real dot product Σ rxᵢ·ryᵢ. *)
let s_dot (#n:nat) (rx ry : lseq real n) : real =
  rsum (Map.lseq_map2 ( *. ) rx ry)

inline_for_extraction noextract
type dot_ty (et : Type0) {| floating et, real_like et, floating_real_like et |} =
  fn (nth : szp { nth <= max_threads })
     (lena : szp { lena <= max_blocks * max_threads /\ SZ.fits (lena + nth) })
     (x : array1 et (l1_forward lena) { is_global x })
     (y : array1 et (l1_forward lena) { is_global y })
     (#vx : erased (lseq et lena))
     (#vy : erased (lseq et lena))
     (#fx : perm)
     (#fy : perm)
     (rx : erased (lseq real lena))
     (ry : erased (lseq real lena))
  preserves
    cpu ** on gpu_loc (x |-> Frac fx vx) ** on gpu_loc (y |-> Frac fy vy)
  requires
    pure (vx %~ rx /\ vy %~ ry)
  returns
    res : et
  ensures
    pure (res %~ s_dot rx ry)

val dot_f16 : dot_ty f16
val dot_f32 : dot_ty f32
val dot_f64 : dot_ty f64
