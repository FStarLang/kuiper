module Kuiper.Kernel.GEMM.Naive

(* This is a very naive matmul, spawning MxN blocks of 1 thread
to do the computation. *)

#lang-pulse

open Kuiper

module T = Kuiper.Tensor
module M = Kuiper.Array2
module MS = Kuiper.Spec.GEMM
open Kuiper.EMatrix
open Kuiper.Kernel.GEMMGPU.Type

inline_for_extraction noextract
let size_req : size_req_t =
  fun m n k -> m * n <= max_blocks

inline_for_extraction noextract
val kdesc
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : M.layout m k)
  (#lB : M.layout k n)
  (#lC : M.layout m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : M.array2 et lA { M.is_global gA })
  (gB : M.array2 et lB { M.is_global gB })
  (gC : M.array2 et lC { M.is_global gC })
  (#eA #eB #eC : ematrix et _ _)
  (#fA #fB : perm)
  (#_ : squash (size_req m n k))
  : kernel_desc
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.mmcomb comb eC eA eB)

inline_for_extraction noextract
val mmcomb_gpu_exact : matmulcomb_gpu_ty size_req

inline_for_extraction noextract
val mmcomb_gpu_approx : matmulcomb_gpu_approx_ty size_req
