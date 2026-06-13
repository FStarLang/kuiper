module Klas.Asum

(* BLAS level-1 asum: sum of absolute values, res = Σ |xᵢ|.
   Corresponds to cublasSasum/Dasum (real element types only).

   Implemented on top of the verified parallel reduction
   [Kuiper.Kernel.HReduce.reduce], using an fmax-based absolute value
   (|x| = max(x, -x)) whose real approximation follows from the existing
   fmax/sub approximation laws, so no new typeclass axioms are needed.

   As in cuBLAS the input is a device pointer; unlike cuBLAS there is no
   stride argument and the spec is the real-valued sum of magnitudes that
   the floating-point result approximates. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Seq.Common { seq_map }
open Kuiper.Tensor.Layout.Alg { l1_forward }
module SZ = Kuiper.SizeT

(* Real absolute value, expressed so it matches the conclusion of
   [fmax_approx] (which yields [rmax]). *)
unfold let rabs (r : real) : real = rmax r (0.0R -. r)

(* Spec: the real sum of magnitudes. *)
let s_asum (#n:nat) (vr : lseq real n) : real = rsum (seq_map rabs vr)

inline_for_extraction noextract
type asum_ty (et : Type0) {| floating et, real_like et, floating_real_like et |} =
  fn (nth : szp { nth <= max_threads })
     (lena : szp { SZ.fits (lena + nth) })
     (a : array1 et (l1_forward lena) { is_global a })
     (#va : erased (lseq et lena))
     (vr : erased (lseq real lena))
  preserves
    cpu ** on gpu_loc (a |-> va)
  requires
    pure (va %~ vr)
  returns
    res : et
  ensures
    pure (res %~ s_asum vr)

val asum_f16 : asum_ty f16
val asum_f32 : asum_ty f32
val asum_f64 : asum_ty f64
