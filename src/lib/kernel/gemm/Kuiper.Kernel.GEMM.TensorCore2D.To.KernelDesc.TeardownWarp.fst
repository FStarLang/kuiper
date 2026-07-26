module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.TeardownWarp

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15 --split_queries always"

open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Kernel.GEMM.Tiled.Common.Vec

module SZ = Kuiper.SizeT

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.TeardownHelpers

ghost
fn gather_warp
  (#et_cd : Type0) {| scalar et_cd, real_like et_cd |}
  (#m #n : szp)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (wm * tm /?+ bm /\ wn * tn /?+ bn))
  (#_ : squash (tm /?+ (wm * tm) /\ tn /?+ (wn * tn)))
  (#_ : squash (SZ.fits ((rm m n).ulen)))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (bid : natlt nblk)
  (wid : natlt (nthr / warp_size))
  (rWarp : chest2 real (wm * tm) (wn * tn))
  requires
    forall+ (lane : natlt warp_size).
      output_lane_approximates
        gD bm bn tm tn wm wn bid (wid * warp_size + lane)
        rWarp
  ensures
    exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
      warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
        (wm * tm) (wn * tn) wid |-> eWarp **
      pure (eWarp %~ rWarp)
{
  assert pure (wid < bm / (wm * tm) * (bn / (wn * tn)));
  assert pure (wid / (bn / (wn * tn)) < bm / (wm * tm));
  assert pure (wid % (bn / (wn * tn)) < bn / (wn * tn));
  Math.Lemmas.lemma_div_exact (wm * tm) tm;
  Math.Lemmas.lemma_div_exact (wn * tn) tn;
  assert pure ((wm * tm) / tm == wm);
  assert pure ((wn * tn) / tn == wn);
  forevery_map
    (fun (lane : natlt warp_size) ->
      output_lane_approximates gD bm bn tm tn wm wn bid
        (wid * warp_size + lane) rWarp)
    (fun (lane : natlt warp_size) ->
      forall+ (mi : natlt wm) (nj : natlt wn).
        exists* (eFrag : chest2 et_cd tm tn).
          own_lane_cells
            (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
            eFrag lane **
          pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj))
    fn lane {
      assert pure ((wid * warp_size + lane) / warp_size == wid);
      assert pure ((wid * warp_size + lane) % warp_size == lane);
      unfold output_lane_approximates gD bm bn tm tn wm wn bid
        (wid * warp_size + lane) rWarp;
      rewrite each ((wid * warp_size + lane) / warp_size) as wid;
      rewrite each ((wid * warp_size + lane) % warp_size) as lane;
    };
  forevery_commute
    (fun (lane : natlt warp_size) (mi : natlt wm) ->
      forall+ (nj : natlt wn).
        exists* (eFrag : chest2 et_cd tm tn).
          own_lane_cells
            (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
            eFrag lane **
          pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj));
  forevery_map
    (fun (mi : natlt wm) ->
      forall+ (lane : natlt warp_size) (nj : natlt wn).
        exists* (eFrag : chest2 et_cd tm tn).
          own_lane_cells
            (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
            eFrag lane **
          pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj))
    (fun (mi : natlt wm) ->
      forall+ (nj : natlt wn) (lane : natlt warp_size).
        exists* (eFrag : chest2 et_cd tm tn).
          own_lane_cells
            (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
            eFrag lane **
          pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj))
    fn mi { forevery_commute _ };
  forevery_map_2
    (fun mi nj ->
      forall+ (lane : natlt warp_size).
        exists* (eFrag : chest2 et_cd tm tn).
          own_lane_cells
            (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
            eFrag lane **
          pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj))
    (fun mi nj ->
      exists* (eFrag : chest2 et_cd tm tn).
        output_fragment gD bm bn tm tn wm wn bid wid mi nj |-> eFrag **
        pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj))
    fn mi nj {
      join_lane_cells_approximates
        (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
        (ematrix_subtile rWarp tm tn mi nj);
    };
  let dWarp =
    warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
      (wm * tm) (wn * tn) wid;
  forevery_map_2
    (fun mi nj ->
      exists* (eFrag : chest2 et_cd tm tn).
        output_fragment gD bm bn tm tn wm wn bid wid mi nj |-> eFrag **
        pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj))
    (fun mi nj ->
      exists* (eFrag : chest2 et_cd tm tn).
        array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj |-> eFrag **
        pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj))
    fn mi nj {
      rewrite each output_fragment gD bm bn tm tn wm wn bid wid mi nj
        as array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj;
    };
  forevery_rw_size2 wm ((wm * tm) / tm) wn ((wn * tn) / tn)
    #(fun (mi : natlt wm) (nj : natlt wn) ->
      exists* (eFrag : chest2 et_cd tm tn).
        array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj |-> eFrag **
        pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj));
  array2_untile_approximates dWarp (SZ.v tm) (SZ.v tn) rWarp;
  rewrite each dWarp as
    warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
      (wm * tm) (wn * tn) wid;
}
