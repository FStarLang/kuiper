module Kuiper.Kernel.GEMM.NaiveT

(* This is a very naive matmul, spawning MxN blocks of 1 thread
to do the computation. *)

#lang-pulse

open Kuiper

module MS = Kuiper.Spec.GEMM
open Kuiper.Tensor
open Kuiper.Index
open Kuiper.Chest
open Kuiper.Kernel.GEMMGPU.Type

inline_for_extraction noextract
fn mmcomb_gpu_approx
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA { is_global gA })
  (gB : tensor et lB { is_global gB })
  (gC : tensor et lC { is_global gC })
  (#eA : chest2 et m k)
  (#eB : chest2 et k n)
  (#eC : chest2 et m n)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (m * n <= max_blocks) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    exists* (eC' : chest2 et m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB)
