module Klas.UGS.Projection

#lang-pulse

(* Foundation-level bitwise-equivalence witness for the projection pass of
   wmma_reference/wmma_ugs.cu's [projection_wmma_kernel]:

     FP16 row-major A[M,K] @ logical FP16 W[K,N], computed with
     m16n16k16 WMMA tiles, FP32 accumulation initialized to zero, K tiles
     visited in increasing order, result rounded to FP16 via
     __float2half_rn and stored into P[M,2N].

   This wraps three already-verified/foundation pieces instead of
   reimplementing the WMMA tiling from scratch:

     1. [Klas.GEMM.TensorCore2D.Inst.spec] (unmodified, reused)
        instantiated with [et_ab = half], [et_c = float],
        [bm = bn = 64], [bk = 16], [tm = tn = tk = 16],
        [wm = wn = 4] -- one 64x64 output tile per CUDA block, computed
        by a single warp (32 threads) whose 4x4 grid of 16x16x16 WMMA
        fragments covers the whole tile, with the full (dynamic) K
        handled by that warp's own [K/16]-iteration loop -- structurally
        analogous to wmma_ugs.cu's per-warp `for (k = 0; k < K; k += 16)`
        loop, just with 16 (4x4) WMMA tiles computed per warp instead of
        1. Unlike the previously-used [Klas.GEMM.TensorCore.Inst.specialize_gpu],
        [spec]'s own postcondition proves real-number ([Kuiper.Approximates])
        correctness of its FP32 output against [Kuiper.Spec.GEMM.matmul],
        and it zero-initializes (rather than accumulates into) its
        output, so no caller-side zero-init of the FP32 scratch array is
        needed.
     2. [Kuiper.Kernel.UGS.Copy.copy_to_row_major]: a deliberately
        scalar materialization from an arbitrary logical weight layout
        into private row-major storage. This keeps physical packing out
        of the projection's mathematics.
     3. [Kuiper.Kernel.UGS.Round.round] (new, this witness): a small
        elementwise kernel that converts that FP32 scratch array into the
        final FP16 output via the same [__float2half_rn] rounding
        wmma_ugs.cu applies (per cell instead of via a [__shared__]
        staging tile + strided per-lane loop; the rounded values are
        identical either way), together with
        [Kuiper.Kernel.UGS.Round.round_preserves_approx], which lifts
        [spec]'s real-number approximation guarantee across this
        rounding step. *)

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore
open Kuiper.Float.Casts.Base
open Kuiper.EMatrix

open Klas.GEMM.TensorCore2D.Inst
open Kuiper.Kernel.UGS.Copy
open Kuiper.Kernel.UGS.Round

module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

let aligned_128_implies_16 (#a : Type0) (#n : nat) (x : larray a n)
  : Lemma (requires aligned 128 x)
          (ensures aligned 16 x)
  = if base_address x = 0 then ()
    else lemma_divides_trans 16 128 (base_address x)

inline_for_extraction noextract
fn projection
  (m k n2 : szp)
  (#lW : layout2 k n2) {| ctlayout lW |}
  (#_ : squash (SZ.fits (m * k)))
  (#_ : squash (SZ.fits (k * n2)))
  (#_ : squash (SZ.fits (m * n2)))
  (#_ : squash (SZ.v m % 64 == 0))
  (#_ : squash (SZ.v k % 16 == 0))
  (#_ : squash (SZ.v n2 % 64 == 0))
  (gA : array2 half (rm m k) { is_global gA })
  (gW : array2 half lW { is_global gW })
  (gPf32 : array2 float (rm m n2) { is_global gPf32 })
  (gP : array2 half (rm m n2) { is_global gP })
  (#_ : squash (aligned 16 (core gA)))
  (#eA : chest2 half m k)
  (#eW : chest2 half k n2)
  (#eC0 : chest2 float m n2)
  (#eP0 : chest2 half m n2)
  (#fA #fW : perm)
  preserves
    cpu **
    pure ((SZ.v m / 64) * (SZ.v n2 / 64) <= SZ.v max_blocks) **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gW |-> Frac fW eW)
  requires
    on gpu_loc (gPf32 |-> eC0) **
    on gpu_loc (gP |-> eP0)
  ensures
    (exists* (eC' : chest2 float (SZ.v m) (SZ.v n2)).
      on gpu_loc (gPf32 |-> eC') **
      on gpu_loc (gP |-> chest_map cast_f32_to_f16 eC') **
      pure (
        eC' %~ MS.matmul (to_real_matrix eA) (to_real_matrix eW) /\
        chest_map cast_f32_to_f16 eC' %~ MS.matmul (to_real_matrix eA) (to_real_matrix eW)))
{
  let gWrmArray = gpu_array_alloc #half (SZ.mul k n2);
  with eWrmSeq. assert on gpu_loc (gWrmArray |-> eWrmSeq);
  let gWrm0 = Kuiper.Tensor.from_array (rm k n2) gWrmArray;
  assert rewrites_to gWrm0
    (Kuiper.Tensor.from_array (rm k n2) gWrmArray);
  map_loc gpu_loc
    (fun () ->
      Kuiper.Tensor.tensor_abs' (rm k n2) gWrmArray #1.0R #eWrmSeq);
  Kuiper.Tensor.lem_from_array_core
    #half #2 #(k @| n2 @| INil) #(rm k n2) gWrmArray;
  Kuiper.Tensor.lem_is_global_iff_core gWrm0;
  assert pure (is_global gWrm0);
  assert pure (aligned 128 (core gWrm0));
  aligned_128_implies_16 (core gWrm0);
  assert pure (aligned 16 (core gWrm0));
  with eWrm0. assert on gpu_loc (gWrm0 |-> eWrm0);
  launch_kernel_1
    (fun _ -> copy_to_row_major gW gWrm0 #fW #eW #eWrm0);
  assert pure (aligned 16 (core gWrm0));

  spec half float 64sz 64sz 16sz 16sz 16sz 16sz 4sz 4sz
    m k n2 gA gWrm0 gPf32;

  with eC'. assert on gpu_loc (gPf32 |-> eC');
  Kuiper.Tensor.free gWrm0 #eW;
  assert pure (eC' %~ MS.matmul (to_real_matrix eA) (to_real_matrix eW));

  launch_kernel_1 (fun _ -> round gPf32 gP #1.0R #eC' #eP0);

  round_preserves_approx eC' (MS.matmul (to_real_matrix eA) (to_real_matrix eW));

  ()
}
