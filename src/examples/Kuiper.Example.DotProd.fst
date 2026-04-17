module Kuiper.Example.DotProd

(* Matmul dot product implemented by extracting a row and column
   as Array1's, then computing a dot product between them. *)

#lang-pulse

open Kuiper
open Kuiper.EMatrix { ematrix }
open Kuiper.Tensor.Layout.Alg
module Array2 = Kuiper.Array2
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT

[@@Comment
"Compute the (i,j) element of the matrix product of A and B by extracting the
i-th row of A and the j-th column of B, then computing their dot product. The
resulting code looks just like the usual implementation."]
fn matmul_dotprod_via_slice_f32
  (#m #n #k : SZ.t)
  (gA : Array2.t f32 (l2_row_major m k))
  (gB : Array2.t f32 (l2_row_major k n))
  (i : szlt m)
  (j : szlt n)
  (#eA #eB : ematrix f32 _ _)
  (#fA #fB : perm)
  preserves
    gpu ** gA |-> Frac fA eA ** gB |-> Frac fB eB
  returns
    res : f32
  ensures
    pure (res == MS.matmul_single eA eB i j)
{
  // Recall some lengths
  Array2.pts_to_ref gA;
  Array2.pts_to_ref gB;

  // Just call the library function that does that.
  Kuiper.DotProd.matmul_dotprod gA gB i j;
}
