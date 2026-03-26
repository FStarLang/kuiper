module Kuiper.Poly.GEMM.Naive

(* This is a very naive matmul, spawning MxN blocks of 1 thread
to do the computation. *)

#lang-pulse

open Kuiper
open Kuiper.Poly.GEMMGPU.Type

(* Exposing kernel alone to call it async. *)
module M = Kuiper.Matrix
open Kuiper.Matrix.Reprs
open Kuiper.EMatrix
module MS = Kuiper.Spec.GEMM

inline_for_extraction noextract
val kdesc
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : M.gpu_matrix et lA { M.is_global_matrix gA })
  (#fA : perm)
  (gB : M.gpu_matrix et lB { M.is_global_matrix gB })
  (#fB : perm)
  (gC : M.gpu_matrix et lC { M.is_global_matrix gC })
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  (#_ : squash (rows * cols <= max_blocks))
  : kernel_desc
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.mmcomb comb eC eA eB)
(* / kernel *)

inline_for_extraction noextract
val mmcomb_gpu_exact :
  matmulcomb_gpu_ty
    (fun rows shared cols -> rows * cols <= max_blocks)

inline_for_extraction noextract
val mmcomb_gpu_approx :
  matmulcomb_gpu_approx_ty
    (fun rows shared cols -> rows * cols <= max_blocks)
