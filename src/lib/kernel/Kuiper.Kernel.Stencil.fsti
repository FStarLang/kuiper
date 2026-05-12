module Kuiper.Kernel.Stencil

#lang-pulse

open Kuiper
open Kuiper.Array2
open Kuiper.EMatrix
open Kuiper.Tensor.Layout
module STS = Kuiper.Spec.Stencil
module Array2 = Kuiper.Array2

inline_for_extraction noextract
fn specialize_host_simple_stencil
  (et: Type0) {| scalar et |}
  (stencil: (i: natlt 3) -> (j: natlt 3) -> et)
  (rIn rOut : trepr2)
  {| ctrepr2 rIn, ctrepr2 rOut |}
  (rows cols : (x:szp{x >= 3}))
  (gIn : array2 et (rIn rows cols) { Array2.is_global gIn })
  (gOut : array2 et (rOut (rows - 2) (cols - 2)) { Array2.is_global gOut })
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
