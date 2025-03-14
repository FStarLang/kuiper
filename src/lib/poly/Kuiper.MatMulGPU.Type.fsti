module Kuiper.MatMulGPU.Type

#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
module MS = Kuiper.Spec.MatMul
module M  = Kuiper.Matrix

unfold
inline_for_extraction
type matmul_gpu_ty =
  (#et : Type0) -> {| scalar et |} ->
  (#rows : szp) ->
  (#shared : szp) ->
  (#cols : szp{three_fits rows shared cols}) ->
  (#lA : mlayout rows shared) ->
  (#lB : mlayout shared cols) ->
  (#lC : mlayout rows cols) ->
  {| clayout lA |} ->
  {| clayout lB |} ->
  {| clayout lC |} ->
  (gA : M.gpu_matrix et lA) ->
  (gB : M.gpu_matrix et lB) ->
  (gC : M.gpu_matrix et lC) ->
  (#eA : ematrix et rows shared) ->
  (#eB : ematrix et shared cols) ->
  (#eC : ematrix et rows cols) ->
  (* This has a preserves. *)
  stt unit
    (requires
      (cpu ** (gA |-> eA) ** (gB |-> eB)) **
      (pure (rows * cols <= max_blocks) **
       (gC |-> eC)))
    (ensures fun _ ->
      (cpu ** (gA |-> eA) ** (gB |-> eB)) **
      (gC |-> MS.matmul eA eB))
