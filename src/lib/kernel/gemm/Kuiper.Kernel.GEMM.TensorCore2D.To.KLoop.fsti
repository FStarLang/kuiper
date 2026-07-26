module Kuiper.Kernel.GEMM.TensorCore2D.To.KLoop

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15"

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
module B = Kuiper.Barrier
module T = Kuiper.Tensor
module FB = Kuiper.Kernel.GEMM.FlipFlopBarrier2
module MS = Kuiper.Spec.GEMM

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc


open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState
open Kuiper.Kernel.GEMM.TensorCore2D.To.KLoopState

let warp_matmul
  (#m #n : nat)
  (#k : pos)
  (a : chest2 real m k)
  (b : chest2 real k n)
  (rows : pos{rows /?+ m})
  (cols : pos{cols /?+ n})
  (row : natlt (m / rows))
  (col : natlt (n / cols))
  : chest2 real rows cols
= MS.matmul
    (ematrix_subtile a rows k row 0)
    (ematrix_subtile b k cols 0 col)

let kernel_warp_matmul
  (#m #n #k : szp)
  (a : chest2 real m k)
  (b : chest2 real k n)
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (wm * tm /?+ m /\ wn * tn /?+ n))
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (bid : szlt (m / bm * (n / bn)))
  (tid : szlt nthr)
  : chest2 real (wm * tm) (wn * tn)
= warp_matmul a b (wm * tm) (wn * tn)
    (warp_tile_i #m #n bm bn bk tm tn tk wm wn
      nthr bid (tid / warp_size))
    (warp_tile_j #m #n bm bn bk tm tn tk wm wn
      nthr bid (tid / warp_size))

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

inline_for_extraction noextract
fn k_loop_step
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc |}
  {| real_like et_ab, real_like et_acc |}
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
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (#_ : squash (wm * tm /?+ m /\ wn * tn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (SZ.fits (m * k)))
  (#_ : squash (SZ.fits (k * n)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (#_ : squash (eA %~ rA /\ eB %~ rB))
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm * bk + nthr - 1)))
  (#_ : squash (SZ.fits (bk * bn + nthr - 1)))
  (sA : array2 et_ab (rm bm bk))
  (sB : array2 et_ab (rm bk bn))
  (#fA #fB : perm)
  (bid : szlt (m / bm * (n / bn)))
  (tid : szlt nthr)
  (mrow : szlt (m / bm) {
    SZ.v mrow == SZ.v bid / (SZ.v n / SZ.v bn) })
  (mcol : szlt (n / bn) {
    SZ.v mcol == SZ.v bid % (SZ.v n / SZ.v bn) })
  (warpRow : szlt (bm / (wm * tm)) {
    SZ.v warpRow == (SZ.v tid / warp_size) /
      (SZ.v bn / (SZ.v wn * SZ.v tn)) })
  (warpCol : szlt (bn / (wn * tn)) {
    SZ.v warpCol == (SZ.v tid / warp_size) %
      (SZ.v bn / (SZ.v wn * SZ.v tn)) })
  (gwRow : enatlt (m / (wm * tm)) {
    gwRow == mrow * (bm / (wm * tm)) + warpRow})
  (gwCol : enatlt (n / (wn * tn)) {
    gwCol == mcol * (bn / (wn * tn)) + warpCol})
  (rAcc0 : chest2 real (wm * tm) (wn * tn) {
    rAcc0 == const _ 0.0R})
  (aFrags : array (fragment et_ab FragA tm tn tk FragLRM))
  (bFrags : array (fragment et_ab FragB tm tn tk FragLRM))
  (accFrags : array (fragment et_acc FragAcc tm tn tk FragLAcc))
  (#_ : squash (Pulse.Lib.Array.length aFrags == wm))
  (#_ : squash (Pulse.Lib.Array.length bFrags == wn))
  (#_ : squash (Pulse.Lib.Array.length accFrags == wm * wn))
  (v : szle (k / bk))
  (#_ : squash (SZ.v v < k / bk))
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    thread_id nthr tid **
    block_id (m / bm * (n / bn)) bid **
    B.barrier_tok
      (FB.contract eA eB (rm bm bk) (rm bk bn)
        (core sA) (core sB) nthr bid)
  requires
    fragarrayAcc_approximates wm wn accFrags
      (tiled_partial_matmul rAcc0
        (ematrix_tiled rA (wm * tm) bk)
        (ematrix_tiled rB bk (wn * tn))
        gwRow gwCol v) **
    live aFrags **
    live bFrags **
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr) **
    B.barrier_state (barrier_iteration v)
  returns vnext : szle (k / bk)
  ensures
    pure (SZ.v vnext == SZ.v v + 1) **
    fragarrayAcc_approximates wm wn accFrags
      (tiled_partial_matmul rAcc0
        (ematrix_tiled rA (wm * tm) bk)
        (ematrix_tiled rB bk (wn * tn))
        gwRow gwCol (v + 1)) **
    live aFrags **
    live bFrags **
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr) **
    B.barrier_state (barrier_iteration vnext)

inline_for_extraction noextract
fn compute_acc
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc |}
  {| real_like et_ab, real_like et_acc |}
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
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
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
  (#_ : squash (valid_frag_et_dims et_acc FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (wm * tm /?+ m))
  (#_ : squash (wn * tn /?+ n))
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (#_ : squash (eA %~ rA /\ eB %~ rB))
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm * bk + nthr - 1)))
  (#_ : squash (SZ.fits (bk * bn + nthr - 1)))
  (sA : array2 et_ab (rm bm bk))
  (sB : array2 et_ab (rm bk bn))
  (#fA #fB : perm)
  (bid : szlt (m / bm * (n / bn)))
  (tid : szlt nthr)
  (mrow : szlt (m / bm) {
    SZ.v mrow == SZ.v bid / (SZ.v n / SZ.v bn) })
  (mcol : szlt (n / bn) {
    SZ.v mcol == SZ.v bid % (SZ.v n / SZ.v bn) })
  (warpRow : szlt (bm / (wm * tm)) {
    SZ.v warpRow == (SZ.v tid / warp_size) /
      (SZ.v bn / (SZ.v wn * SZ.v tn)) })
  (warpCol : szlt (bn / (wn * tn)) {
    SZ.v warpCol == (SZ.v tid / warp_size) %
      (SZ.v bn / (SZ.v wn * SZ.v tn)) })
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    thread_id nthr tid **
    block_id (m / bm * (n / bn)) bid **
    B.barrier_tok
      (FB.contract eA eB (rm bm bk) (rm bk bn)
        (core sA) (core sB) nthr bid)
  requires
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr) **
    B.barrier_state 0
  returns accFrags : array (fragment et_acc FragAcc tm tn tk FragLAcc)
  ensures
    pure (Pulse.Lib.Array.length accFrags == wm * wn) **
    fragarrayAcc_approximates wm wn accFrags
      (kernel_warp_matmul rA rB
        bm bn bk tm tn tk wm wn nthr bid tid) **
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr) **
    B.barrier_state (tile_barrier_iteration k bk)
