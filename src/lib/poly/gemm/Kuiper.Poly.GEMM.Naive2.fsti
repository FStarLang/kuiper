module Kuiper.Poly.GEMM.Naive2

(* This is a less naive matmul. It spawns full blocks of 1024 threads,
going in row major order through the output matrix and with each thread
computing a full dot product. *)

#lang-pulse

open Kuiper
open Kuiper.Approximates
open Kuiper.Poly.GEMMGPU.Type
module MU = Kuiper.Poly.GEMM.Util
module MS = Kuiper.Spec.GEMM
module M = Kuiper.Matrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.EMatrix

inline_for_extraction noextract
val mmcomb_gpu_exact :
  matmulcomb_gpu_ty
    (fun rows shared cols -> rows * cols <= max_blocks * max_threads)

inline_for_extraction noextract
val mmcomb_gpu_approx
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#rows #shared #cols : szp)
  (#lA : full_mlayout rows shared)
  (#lB : full_mlayout shared cols)
  (#lC : full_mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : M.gpu_matrix et lA { M.is_global_matrix gA })
  (#fA : perm)
  (gB : M.gpu_matrix et lB { M.is_global_matrix gB })
  (#fB : perm)
  (gC : M.gpu_matrix et lC { M.is_global_matrix gC })
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
  : stt unit
    (requires
      (cpu ** on gpu_loc (gA |-> Frac fA eA) ** on gpu_loc (gB |-> Frac fB eB)) **
      (pure (rows * cols <= max_blocks * max_threads) **
       pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
       on gpu_loc (gC |-> eC)))
    (ensures fun _ ->
      (cpu ** on gpu_loc (gA |-> Frac fA eA) ** on gpu_loc (gB |-> Frac fB eB)) **
      (exists* (eC' : ematrix et rows cols).
        on gpu_loc (gC |-> eC') **
        pure (eC' %~ MS.mmcomb comb_r rC rA rB)))
