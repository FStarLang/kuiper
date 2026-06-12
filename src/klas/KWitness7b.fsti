module KWitness7b

(* An ALTERNATIVE, fully-Kuiper, verified witness that is bit-for-bit equivalent
   to imp7.cu -- built without writing any new transpose kernel.

   imp7.cu computes A*B and stores it transposed (C[c*m+r] = sum), i.e. it fuses a
   matmul and a transpose into one kernel. KWitness7 reproduces that fused kernel
   directly, with a hand-written transposed-store loop and its own invariant. This
   witness instead uses the idiomatic Kuiper "transpose = view shift" trick (see
   Kuiper.Example.MatMulTranspose and Kuiper.Ghost.TensorTranspose):

     1. ghost-reinterpret the n x m row-major output C as an m x n COL-MAJOR matrix
        (a zero-cost view shift -- no data moves);
     2. run the EXISTING verified naive forward matmul (Kuiper.Kernel.GEMM.Naive,
        reused by KWitness1/KWitness6) writing A*B into that col-major view; and
     3. ghost-reinterpret the buffer back to row-major, where it now holds (A*B)^T.

   The matmul writes (A*B)[r][c] to the col-major offset c*m+r -- precisely imp7's
   `C[c*m+r] = sum` -- with the same forward dot product as imp1/imp6/imp7, so the
   result is bit-for-bit identical to imp7.cu, with NO transpose kernel and NO copy
   (the ghost view shifts are erased; the extracted code is a single matmul kernel).

   The spec is the SAME as KWitness7's -- the f32 result (n x m) approximates
   `mtranspose (MS.matmul rA rB)`. It differs only in the size precondition, which
   here is inherited from the naive matmul kernel (one block per output cell). *)

#lang-pulse
open Kuiper
open Kuiper.EMatrix
module Alg = Kuiper.Tensor.Layout.Alg
module MS = Kuiper.Spec.GEMM
module M = Kuiper.Array2
module SZ = Kuiper.SizeT

(* Concrete, monomorphic spec: f32, row-major A (m x k), B (k x n), C (n x m).
   - requires SZ.v m * SZ.v n <= SZ.v max_blocks: the naive matmul launches one
     block per output cell of the intermediate m x n product.
   - requires the f32 inputs approximate the real matrices rA, rB.
   - ensures the resulting f32 matrix approximates the TRANSPOSE of the exact
     real matmul. *)
fn matmul_f32
  (m n k : szp)
  (gA : M.array2 f32 (Alg.l2_row_major m k) { M.is_global gA })
  (gB : M.array2 f32 (Alg.l2_row_major k n) { M.is_global gB })
  (gC : M.array2 f32 (Alg.l2_row_major n m) { M.is_global gC })
  (rA : ematrix real m k)
  (rB : ematrix real k n)
  (#eA : ematrix f32 m k)
  (#eB : ematrix f32 k n)
  (#eC : ematrix f32 n m)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (SZ.v m * SZ.v n <= SZ.v max_blocks) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : ematrix f32 n m).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ mtranspose (MS.matmul rA rB)))
