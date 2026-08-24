module Kuiper.Kernel.GEMM.TensorCore2D.KLoop

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
open Kuiper.Kernel.GEMM.TensorCore2D.FragmentAcc
open Kuiper.Kernel.GEMM.TensorCore2D.Subproducts

val tile_offset
  (outer tile : pos)
  (#_ : squash (tile /? outer))
  (block inner global : nat)
  (#_ : squash (global == block * (outer / tile) + inner))
  (_ : unit)
  : Lemma (tile * global == outer * block + inner * tile)

val zero_tile_offset (tile : pos) (i : nat) (_ : unit)
  : Lemma (tile * i == tile * i + 0 * tile)

val nested_subtile_tiled_cell
  (#et : Type0)
  (#rows #cols : nat)
  (em : chest2 et rows cols)
  (outer_rows : pos { outer_rows /? rows })
  (outer_cols : pos { outer_cols /? cols })
  (inner_rows : pos { inner_rows /? outer_rows /\ inner_rows /? rows })
  (inner_cols : pos { inner_cols /? outer_cols /\ inner_cols /? cols })
  (outer_row : natlt (rows / outer_rows))
  (outer_col : natlt (cols / outer_cols))
  (inner_row : natlt (outer_rows / inner_rows))
  (inner_col : natlt (outer_cols / inner_cols))
  (global_row : natlt (rows / inner_rows))
  (global_col : natlt (cols / inner_cols))
  (#_ : squash (inner_rows * global_row ==
                outer_rows * outer_row + inner_row * inner_rows))
  (#_ : squash (inner_cols * global_col ==
                outer_cols * outer_col + inner_col * inner_cols))
  (outer : chest2 et outer_rows outer_cols {
    outer == ematrix_subtile em outer_rows outer_cols outer_row outer_col })
  (_ : unit)
  : Lemma (
      ematrix_subtile outer inner_rows inner_cols inner_row inner_col
      == acc2 (ematrix_tiled em inner_rows inner_cols) global_row global_col)

val partial_matmul_step
  (#rows #shared #cols #tm #tk #tn : nat)
  (z : chest2 real tm tn)
  (a : chest2 (chest2 real tm tk) rows shared)
  (b : chest2 (chest2 real tk tn) shared cols)
  (row : natlt rows)
  (col : natlt cols)
  (vk : natlt shared)
  (rAcc : chest2 real tm tn {
    rAcc == __gmatmul_single z matmul matplus a b row col vk })
  (aTile : chest2 real tm tk { aTile == acc2 a row vk })
  (bTile : chest2 real tk tn { bTile == acc2 b vk col })
  (_ : unit)
  : Lemma (
      rAcc `matplus` matmul aTile bTile
      == __gmatmul_single z matmul matplus a b row col (vk + 1))

// Keep the large tiled invariant opaque in loop VCs: unfolding it activates
// macc_ematrix_tiled's trigger and causes a multi-GB E-matching blowup.  The
// definition is exposed only for concrete indices through kacc_inv_eq.
#push-options "--z3rlimit 100"
[@@"opaque_to_smt"]
inline_for_extraction noextract
let kacc_inv
  (m n k : nat)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bk /?+ k))
  (#_ : squash (wm * tm /?+ m))
  (#_ : squash (wn * tn /?+ n))
  (mrA : chest2 real m k)
  (mrB : chest2 real k n)
  (rAcc0 : chest2 real (wm*tm) (wn*tn))
  (gwRow : natlt (m/(wm*tm)))
  (gwCol : natlt (n/(wn*tn)))
  (vk : nat { vk <= k / bk })
: chest2 real (wm*tm) (wn*tn)
= __gmatmul_single rAcc0 matmul matplus
    (ematrix_tiled mrA (wm*tm) bk) (ematrix_tiled mrB bk (wn*tn)) gwRow gwCol vk
#pop-options

val kacc_inv_eq
  (m n k : nat)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bk /?+ k))
  (#_ : squash (wm * tm /?+ m))
  (#_ : squash (wn * tn /?+ n))
  (mrA : chest2 real m k)
  (mrB : chest2 real k n)
  (rAcc0 : chest2 real (wm*tm) (wn*tn))
  (gwRow : natlt (m/(wm*tm)))
  (gwCol : natlt (n/(wn*tn)))
  (vk : nat { vk <= k / bk })
: Lemma (kacc_inv m n k bm bn bk tm tn tk wm wn mrA mrB rAcc0 gwRow gwCol vk
           == __gmatmul_single rAcc0 matmul matplus
                (ematrix_tiled mrA (wm*tm) bk) (ematrix_tiled mrB bk (wn*tn))
                gwRow gwCol vk)

#push-options "--z3rlimit 100"
inline_for_extraction noextract
fn ktile_advance
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c, real_like et_ab, real_like et_c |}
  // NOTE: m n k are [szp] (sizet), NOT [nat].  kf's dims are sizet; the
  // sz->nat coercion [sizet_to_nat] is GTot, so taking [nat] params here would
  // make kf's stateful call to this fn GHOST (Pulse "Application of a stateful
  // computation cannot have a ghost effect").  subproducts_tc_2d uses szp/szlt
  // for the same reason.  Coercions to nat happen inside slprops / ghost-fn
  // arguments below, where GTot is fine.
  (m n k : szp)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (#_ : squash (wm * tm /?+ m))
  (#_ : squash (wn * tn /?+ n))
  (#_ : squash (valid_frag_et_comb et_ab et_c))
  (aFrags     : array (fragment et_ab FragA tm tn tk FragLRM))
  (bFrags     : array (fragment et_ab FragB tm tn tk FragLRM))
  (accFrags   : array (fragment et_c FragAcc tm tn tk FragLAcc))
  (#_ : squash (Pulse.Lib.Array.length aFrags == wm))
  (#_ : squash (Pulse.Lib.Array.length bFrags == wn))
  (#_ : squash (Pulse.Lib.Array.length accFrags == wm*wn))
  (sA : array2 et_ab (rm bm bk))
  (sB : array2 et_ab (rm bk bn))
  (#eA_sub : chest2 et_ab bm bk)
  (#eB_sub : chest2 et_ab bk bn)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (mrow : szlt (m / bm))
  (mcol : szlt (n / bn))
  (vk : szlt (k / bk))
  (rA_sub : chest2 real bm bk
            { eA_sub %~ rA_sub /\ rA_sub == ematrix_subtile rA bm bk mrow vk })
  (rB_sub : chest2 real bk bn
            { eB_sub %~ rB_sub /\ rB_sub == ematrix_subtile rB bk bn vk mcol })
  (emA emB : et_ab -> et_ab)
  (mapA mapB : real -> real)
  (#_ : squash (MU.approx1 emA mapA))
  (#_ : squash (MU.approx1 emB mapB))
  (rAcc0 : chest2 real (wm*tm) (wn*tn) { rAcc0 == const _ 0.0R })
  (warpRow : szlt (bm/(wm*tm)))
  (warpCol : szlt (bn/(wn*tn)))
  // NOTE: gwRow/gwCol are [enatlt] (ERASED) to match kf's erased locals.  If
  // they were informative [natlt], passing kf's erased gwRow/gwCol here would
  // require a [reveal], which gives this STATEFUL application a ghost effect
  // (Pulse "Application of a stateful computation cannot have a ghost effect").
  (gwRow : enatlt (m/(wm*tm)) { gwRow == mrow * (bm/(wm*tm)) + warpRow })
  (gwCol : enatlt (n/(wn*tn)) { gwCol == mcol * (bn/(wn*tn)) + warpCol })
  (#fA #fB : perm)
  ()
  preserves gpu
  preserves sA |-> Frac fA eA_sub
  preserves sB |-> Frac fB eB_sub
  preserves live aFrags ** live bFrags
  requires
    fragarrayAcc_approximates wm wn accFrags
      (kacc_inv m n k bm bn bk tm tn tk wm wn
        (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
        rAcc0 gwRow gwCol vk)
  ensures
    fragarrayAcc_approximates wm wn accFrags
      (kacc_inv m n k bm bn bk tm tn tk wm wn
        (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
        rAcc0 gwRow gwCol (vk + 1))
#pop-options
