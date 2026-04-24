module Kuiper.Kernel.Stencil

#lang-pulse

open Kuiper
module M = Kuiper.Matrix
module STS = Kuiper.Spec.Stencil
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type

inline_for_extraction noextract
fn specialize_host_simple_stencil
  (et: Type0) {| scalar et |}
  (stencil: (i: natlt 3) -> (j: natlt 3) -> et)
  (rIn rOut : mrepr)
  {| cIn : crepr rIn |}
  {| cOut : crepr rOut |}
  (#rows #cols : (x:szp{x >= 3}))
  (gIn : M.gpu_matrix et (rIn rows cols) { M.is_global_matrix gIn })
  (gOut : M.gpu_matrix et (rOut (rows - 2) (cols - 2)) { M.is_global_matrix gOut })
  (#fIn : perm)
  (#eIn : ematrix et rows cols)
  (#eOut : ematrix et (rows - 2) (cols - 2))
  preserves
    cpu **
    on gpu_loc (gIn |-> Frac fIn eIn)
  requires
    pure (rows * cols <= max_blocks) **
    on gpu_loc (gOut |-> eOut)
  ensures
    on gpu_loc (gOut |-> STS.stencil_result stencil eIn)