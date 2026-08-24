module Kuiper.Kernel.GEMM.TensorCore2D.KernelLoop

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc { constraints }
#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 10"

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
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
module B = Kuiper.Barrier
module T = Kuiper.Tensor
module FB = Kuiper.Kernel.GEMM.FlipFlopBarrier2
module Chest = Kuiper.Chest
module MU = Kuiper.Kernel.GEMM.Util

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.FragmentAcc
open Kuiper.Kernel.GEMM.TensorCore2D.Subproducts
open Kuiper.Kernel.GEMM.TensorCore2D.KLoop

#push-options "--fuel 1 --ifuel 1 --z3rlimit 10"
inline_for_extraction noextract
fn run_loop
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#m #n #k : szp)
  (#lA : layout2 m k) {| T.ctlayout lA |}
  (gA : array2 et_ab lA)
  (#eA : chest2 et_ab m k)
  (#lB : layout2 k n) {| T.ctlayout lB |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_B))
  (gB : array2 et_ab lB)
  (#eB : chest2 et_ab k n)
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (SZ.fits (m * k)))
  (#_ : squash (SZ.fits (k * n)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_c FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_c))
  (#_ : squash (wm * tm /?+ m))
  (#_ : squash (wn * tn /?+ n))
  (#fA #fB : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (mapA mapB : real -> real)
  (emA emB : et_ab -> et_ab)
  (#_ : squash (MU.approx1 emA mapA))
  (#_ : squash (MU.approx1 emB mapB))
  (nthr : erased nat {
    nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size })
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm * bk + nthr - 1)))
  (#_ : squash (SZ.fits (bk * bn + nthr - 1)))
  (sarA : larray et_ab (bm * bk))
  (sarB : larray et_ab (bk * bn))
  (sA : array2 et_ab (rm bm bk))
  (sB : array2 et_ab (rm bk bn))
  (#_ : squash (
    sA == from_array (rm bm bk) sarA /\
    sB == from_array (rm bk bn) sarB))
  (bid : szlt (m/bm * (n/bn)))
  (tid : szlt nthr)
  (mrow : szlt (m/bm) {
    SZ.v mrow == SZ.v bid / (SZ.v n / SZ.v bn) })
  (mcol : szlt (n/bn) {
    SZ.v mcol == SZ.v bid % (SZ.v n / SZ.v bn) })
  (warpRow : szlt (bm/(wm*tm)))
  (warpCol : szlt (bn/(wn*tn)))
  (gwRow : enatlt (m/(wm*tm)) {
    gwRow == mrow * (bm/(wm*tm)) + warpRow })
  (gwCol : enatlt (n/(wn*tn)) {
    gwCol == mcol * (bn/(wn*tn)) + warpCol })
  (rAcc0 : chest2 real (wm*tm) (wn*tn) {
    rAcc0 == const _ 0.0R })
  (aFrags : array (fragment et_ab FragA tm tn tk FragLRM))
  (bFrags : array (fragment et_ab FragB tm tn tk FragLRM))
  (accFrags : array (fragment et_c FragAcc tm tn tk FragLAcc))
  (#_ : squash (Pulse.Lib.Array.length aFrags == wm))
  (#_ : squash (Pulse.Lib.Array.length bFrags == wn))
  (#_ : squash (Pulse.Lib.Array.length accFrags == wm * wn))
  ()
  preserves
    gpu **
    gA |-> Frac (fA /. (m/bm * (n/bn) * nthr)) eA **
    gB |-> Frac (fB /. (m/bm * (n/bn) * nthr)) eB **
    thread_id nthr tid **
    block_id (m/bm * (n/bn)) bid **
    B.barrier_tok
      (FB.contract eA eB (rm bm bk) (rm bk bn) sarA sarB nthr bid) **
    live aFrags **
    live bFrags
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (eA %~ rA /\ eB %~ rB) **
    fragarrayAcc_approximates wm wn accFrags
      (kacc_inv m n k bm bn bk tm tn tk wm wn
        (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
        rAcc0 gwRow gwCol 0) **
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr) **
    B.barrier_state 0
  ensures
    fragarrayAcc_approximates wm wn accFrags
      (kacc_inv m n k bm bn bk tm tn tk wm wn
        (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
        rAcc0 gwRow gwCol (k / bk)) **
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr) **
    B.barrier_state (2 * (k / bk))
#pop-options
