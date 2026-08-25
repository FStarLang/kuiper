module Kuiper.Kernel.GEMM.TensorCore2D.To.KLoopInvariant

open Kuiper

#set-options "--z3rlimit 20 --fuel 1 --ifuel 1"

open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Spec.GEMM
open Kuiper.Tensor.Tiling

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc { constraints }
module KLoop = Kuiper.Kernel.GEMM.TensorCore2D.KLoop

let loop_invariant_lemma
  (m n k : nat)
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (mrow : natlt (m / bm))
  (mcol : natlt (n / bn))
  (warpRow : natlt (bm / (wm * tm)))
  (warpCol : natlt (bn / (wn * tn)))
  (gwRow : natlt (m / (wm * tm)) {
    gwRow == mrow * (bm / (wm * tm)) + warpRow })
  (gwCol : natlt (n / (wn * tn)) {
    gwCol == mcol * (bn / (wn * tn)) + warpCol })
  (vk : natlt (k / bk))
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rAcc0 : chest2 real (wm * tm) (wn * tn) {
    rAcc0 == const _ 0.0R })
  (rAcc : chest2 real (wm * tm) (wn * tn))
  (#_ : squash (wm * tm /?+ m))
  (#_ : squash (wn * tn /?+ n))
  (#_ : squash (
    rAcc == tiled_partial_matmul rAcc0
      (ematrix_tiled rA (wm * tm) bk)
      (ematrix_tiled rB bk (wn * tn))
      gwRow gwCol vk))
  (rA_sub : chest2 real bm bk {
    rA_sub == ematrix_subtile rA bm bk mrow vk })
  (rB_sub : chest2 real bk bn {
    rB_sub == ematrix_subtile rB bk bn vk mcol })
  : Lemma (
      rAcc `matplus`
        matmul (ematrix_subtile rA_sub (wm * tm) bk warpRow 0)
               (ematrix_subtile rB_sub bk (wn * tn) 0 warpCol)
      == tiled_partial_matmul rAcc0
        (ematrix_tiled rA (wm * tm) bk)
        (ematrix_tiled rB bk (wn * tn))
        gwRow gwCol (vk + 1))
=
  KLoop.tile_offset bm (wm * tm) mrow warpRow gwRow ();
  KLoop.tile_offset bn (wn * tn) mcol warpCol gwCol ();
  KLoop.zero_tile_offset bk vk ();
  KLoop.nested_subtile_tiled_cell rA
    bm bk (wm * tm) bk mrow vk warpRow 0 gwRow vk rA_sub ();
  KLoop.nested_subtile_tiled_cell rB
    bk bn bk (wn * tn) vk mcol 0 warpCol vk gwCol rB_sub ();
  KLoop.partial_matmul_step rAcc0
    (ematrix_tiled rA (wm * tm) bk)
    (ematrix_tiled rB bk (wn * tn))
    gwRow gwCol vk rAcc
    (ematrix_subtile rA_sub (wm * tm) bk warpRow 0)
    (ematrix_subtile rB_sub bk (wn * tn) 0 warpCol) ()
