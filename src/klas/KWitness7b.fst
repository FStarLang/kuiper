module KWitness7b

(* An ALTERNATIVE witness for imp7.cu that avoids writing any new transpose kernel.

   imp7.cu computes A*B and stores it transposed (C[c*m+r] = sum). Rather than
   hand-rolling a transpose (KWitness7) or copying through an intermediate buffer,
   this witness uses the idiomatic Kuiper "transpose = view shift" trick (see
   Kuiper.Example.MatMulTranspose and Kuiper.Ghost.TensorTranspose):

     1. ghost-reinterpret the n x m row-major output gC as an m x n COL-MAJOR matrix
        (row2col gC) -- a zero-cost view shift, no data moves;
     2. run the rank-2 wrapper around the EXISTING verified naive forward matmul
        (Kuiper.Kernel.GEMM.Naive1, reused by KWitness1/KWitness6), writing A*B
        into that col-major view; and
     3. ghost-reinterpret back: the buffer now read row-major is exactly (A*B)^T.

   Because the matmul writes (A*B)[r][c] to the col-major offset c*m+r of gC's
   buffer -- precisely imp7.cu's `C[c*m+r] = sum` -- with the same forward,
   left-associated dot product as imp1/imp6/imp7, the result is bit-for-bit
   identical to imp7.cu, and there is no transpose kernel and no copy at all.

   The public entry point is `matmul_f32`; `mtranspose_approx` is private. *)

#lang-pulse
open Kuiper
open Kuiper.Chest
open Kuiper.EMatrix
module Alg = Kuiper.Tensor.Layout.Alg
module MS = Kuiper.Spec.GEMM
module M = Kuiper.Tensor
module SZ = Kuiper.SizeT
module K = Kuiper.Kernel.GEMM.Naive1
module TGT = Kuiper.Ghost.TensorTranspose

fn matmul_f32
  (m n k : szp)
  (gA : M.array2 f32 (Alg.l2_row_major m k) { M.is_global gA })
  (gB : M.array2 f32 (Alg.l2_row_major k n) { M.is_global gB })
  (gC : M.array2 f32 (Alg.l2_row_major n m) { M.is_global gC })
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (#eA : chest2 f32 m k)
  (#eB : chest2 f32 k n)
  (#eC : chest2 f32 n m)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (SZ.v m * SZ.v n <= SZ.v max_blocks) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : chest2 f32 n m).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ mtranspose (MS.matmul rA rB)))
{
  map_loc gpu_loc (fun () -> M.tensor_pts_to_ref gA);
  map_loc gpu_loc (fun () -> M.tensor_pts_to_ref gB);
  map_loc gpu_loc (fun () -> M.tensor_pts_to_ref gC);

  (* Reinterpret gC (n x m row-major) as an m x n col-major matrix holding the
     transpose of its current contents. *)
  map_loc gpu_loc (fun () -> TGT.ghost_transpose1 gC);

  (* Compute A*B into the col-major view; result %~ matmul rA rB. *)
  K.mmcomb_gpu_approx
    (fun _ x -> x) (fun _ x -> x)
    gA gB (TGT.row2col gC)
    rA rB (to_real_matrix (mtranspose eC));

  (* Reinterpret back: the buffer read row-major is now (A*B)^T. *)
  map_loc gpu_loc (fun () -> TGT.ghost_transpose1_back gC);
}
