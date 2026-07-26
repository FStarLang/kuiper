module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.TeardownWarp

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Kernel.GEMM.Tiled.Common.Vec

module SZ = Kuiper.SizeT

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

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
