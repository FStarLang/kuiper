module Kuiper.Kernel.GEMM.TensorCore2D.Populate

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc { constraints }
#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 120"

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

module Chest = Kuiper.Chest
module MU = Kuiper.Kernel.GEMM.Util

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.ProofSupport
open Kuiper.Kernel.GEMM.TensorCore2D.FragmentAcc

inline_for_extraction noextract
fn populate_fragments_a
  (#et : Type0)
  {| scalar et, real_like et |}
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (frags : array (fragment et FragA tm tn tk FragLRM))
  (gm : array2 et (rm bm bk))
  (#em : chest2 et bm bk)
  (rm : chest2 real bm bk {em %~ rm})
  (emA : et -> et)
  (mapA : real -> real)
  (#_ : squash (MU.approx1 emA mapA))
  (#f : perm)
  (arow : szlt (bm/(wm*tm)))
  (dotIdx : szlt (bk/tk))
  (#_ : squash (Pulse.Lib.Array.length frags == wm))
preserves
  gpu **
  gm |-> Frac f em
requires
  live frags
ensures
  fragarrayA_approximates wm frags
    (ematrix_subtile (Chest.chest_map mapA rm) (wm*tm) tk arow dotIdx)

inline_for_extraction noextract
fn populate_fragments_b
  (#et : Type0)
  {| scalar et, real_like et |}
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (frags : array (fragment et FragB tm tn tk FragLRM))
  (gm : array2 et (rm bk bn))
  (#em : chest2 et bk bn)
  (rm : chest2 real bk bn {em %~ rm})
  (emB : et -> et)
  (mapB : real -> real)
  (#_ : squash (MU.approx1 emB mapB))
  (#f : perm)
  (bcol : szlt (bn/(wn*tn)))
  (dotIdx : szlt (bk/tk))
  (#_ : squash (Pulse.Lib.Array.length frags == wn))
preserves
  gpu **
  gm |-> Frac f em
requires
  live frags
ensures
  fragarrayB_approximates wn frags
    (ematrix_subtile (Chest.chest_map mapB rm) tk (wn*tn) dotIdx bcol)
