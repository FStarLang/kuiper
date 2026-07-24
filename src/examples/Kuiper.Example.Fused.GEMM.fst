module Kuiper.Example.Fused.GEMM

(* A full example exercising every feature of the generalized GEMM kernel
   [Kuiper.Kernel.GEMM.Naive2.gmmcomb_gpu_exact]:

     - four independent element types [ta] [tb] [tc] [tacc],
     - fused input maps  [mapA : ta -> tacc]  and  [mapB : tb -> tacc],
     - a decoupled combine  [comb : tc -> tacc -> tc].

   Concretely we compute

       C <- sqrt(A) @ B

   where A, B and C are all stored in fp16 ([ta = tb = tc = f16]) but the
   dot-product accumulation is carried out in fp32 ([tacc = f32]):

     - [sqrtA : f16 -> f32]  casts each A element up to fp32 and takes its
       square root (the fused elementwise op),
     - [mapB  : f16 -> f32]  simply casts each B element up to fp32,
     - [assign : f16 -> f32 -> f16]  discards the previous C cell and rounds
       the fp32 accumulator back down to fp16 (an overwrite, like [MS.comb2]).

   Unfolding the postcondition, each output cell (i, j) is

       eC'.(i, j) == fcast (matmul_single (chest_map sqrtA eA)
                                           (chest_map mapB eB) i j)

   i.e. the fp32 dot product of the fp16 rows/columns mapped up through
   [sqrtA] / [mapB], rounded back to fp16. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Float.Casts

module MS = Kuiper.Spec.GEMM
module K  = Kuiper.Kernel.GEMM.Naive2

(* Fused elementwise map on A: widen fp16 -> fp32, then take the square root. *)
inline_for_extraction noextract
let sqrtA (x : f16) : f32 = sqrt (fcast x)

(* Map on B: widen fp16 -> fp32. *)
inline_for_extraction noextract
let mapB (x : f16) : f32 = fcast x

(* Combine: overwrite the fp16 output with the rounded fp32 accumulator. *)
inline_for_extraction noextract
let assign (_ : f16) (s : f32) : f16 = fcast s

[@@Comment
"C <- sqrt(A) @ B, with A, B, C in fp16 and the accumulation done in fp32."]
fn gemm_sqrt_fused
  (m n k : szp)
  (gA : tensor f16 (l2_row_major m k) { is_global gA })
  (gB : tensor f16 (l2_row_major k n) { is_global gB })
  (gC : tensor f16 (l2_row_major m n) { is_global gC })
  (#eA #eB #eC : chest2 f16 _ _)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (K.size_req m n k) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.gmmcomb sqrtA mapB assign eC eA eB)
{
  (* Bring the length-fits facts into scope so the [ctlayout] instances for
     the row-major layouts can be resolved (cf. [Kuiper.Example.MatMulTranspose]). *)
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gA);
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gB);
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gC);

  K.gmmcomb_gpu_exact sqrtA mapB assign gA gB gC;
}
