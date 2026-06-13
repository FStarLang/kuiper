module Klas.Asum

(* See Klas.Asum.fsti for the specification.

   asum = Σ |xᵢ|, computed with the verified parallel reduction
   Kuiper.Kernel.HReduce.reduce and an fmax-based absolute value. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
module SZ = Kuiper.SizeT
module K = Kuiper.Kernel.HReduce

(* |x| = max(x, -x).  Computed via fmax/sub so its real approximation
   follows from the existing fmax/sub approximation laws.  Marked [unfold]
   so the approximation proof reduces to fmax_approx directly. *)
unfold let abs_fmax (#et:Type0) {| floating et |} (x : et) : et = fmax x (sub zero x)

(* abs_fmax approximates rabs.  With abs_fmax/rabs unfolded, the goal is
   [v_approximates (fmax x (sub zero x)) (rmax r (0-r))], which the
   sub_approx/fmax_approx SMT-pattern lemmas close once [zero %~ 0] (a0) is
   in scope. *)
let abs_approx
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (x : et) (r : real)
  : Lemma (requires v_approximates x r)
          (ensures v_approximates (abs_fmax x) (rabs r))
  = let _ : squash (v_approximates (zero #et) 0.0R) = a0 in
    ()

(* Lift to the function-approximation [abs_fmax %~ rabs] required by reduce. *)
let abs_fun_approx
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  ()
  : Lemma (ensures (abs_fmax #et) %~ rabs)
  = introduce forall (x:et) (y:real). x %~ y ==> abs_fmax x %~ rabs y
    with introduce _ ==> _
    with _. abs_approx x y

inline_for_extraction noextract
fn asum_gen (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp { nth <= max_threads })
  (lena : szp { SZ.fits (lena + nth) })
  (a : array1 et (l1_forward lena) { is_global a })
  (#va : erased (lseq et lena))
  (vr : erased (lseq real lena))
  norewrite
  preserves
    cpu ** on gpu_loc (a |-> va)
  requires
    pure (va %~ vr)
  returns
    res : et
  ensures
    pure (res %~ s_asum vr)
{
  abs_fun_approx #et ();
  K.reduce abs_fmax rabs nth lena a vr;
}

let asum_f16 = asum_gen #f16
let asum_f32 = asum_gen #f32
let asum_f64 = asum_gen #f64
