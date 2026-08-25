module Kuiper.Kernel.GEMM.TensorCore2D.Epilogue

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc { constraints }
#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 20"

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.Float16
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Kernel.GEMM.Copy.Vec2
open Kuiper.Kernel.GEMM.Tiled.Common.Vec
open Kuiper.Spec.GEMM
open Kuiper.TensorCore
open Pulse.Lib.Array
open Pulse.Lib.Trade

module SZ = Kuiper.SizeT
module Chest = Kuiper.Chest

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.FragmentAcc
open Kuiper.Kernel.GEMM.TensorCore2D.Fade
open Kuiper.Kernel.GEMM.TensorCore2D.FadeUpdate

#push-options "--fuel 1 --ifuel 1"
inline_for_extraction noextract
fn epilogue
  (#et : Type0) {| scalar et, real_like et |}
  (#m : erased nat)
  // n is concretized so using size is more succinct
  (#n : sz)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (accumFrags : array (fragment et FragAcc tm tn tk FragLAcc))
  (rAcc : chest2 real (wm*tm) (wn*tn))
  (gC : array2 et (rm m n))
  // Fused output combine: the real-domain [comb] and its approximation-compatible
  // device element combine [ecomb].  The stored result approximates
  // [chest_comb comb rCtile rAcc] (old C combined with the accumulator).
  (comb : real -> real -> real)
  (ecomb : et -> et -> et)
  (rCtile : chest2 real (wm*tm) (wn*tn))
  (#_ : squash (Kuiper.Approximates.approx2 ecomb comb))
  // (#eC : chest2 et m n)
  (#_ : squash (SZ.fits (m * n)))
  (bid : szlt (m/bm * (n/bn)))
  (wid : szlt (bm/(wm*tm) * (bn/(wn*tn))))
  (#_ : squash (Pulse.Lib.Array.length accumFrags == wm*wn))
  preserves
    gpu **
    fragarrayAcc_approximates wm wn accumFrags rAcc
  requires
    pure (SZ.fits (wm * wn)) **
    warp_tile_approximates gC bm bn tm tn wm wn bid wid rCtile
  ensures
    warp_tile_approximates gC bm bn tm tn wm wn bid wid (Chest.chest_comb comb rCtile rAcc)
#pop-options

inline_for_extraction noextract
fn populate_acc_with_zero
  (#et : Type0) {| sc : scalar et, real_like et |}
  (tm tn tk wm wn : szp)
  (accumFrags : array (fragment et FragAcc tm tn tk FragLAcc))
  (#_ : squash (Pulse.Lib.Array.length accumFrags == wm*wn))
requires
  live accumFrags
ensures
  fragarrayAcc_approximates wm wn accumFrags (const _ 0.0R)
