module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.TeardownBlock

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

ghost
fn gather_block
  (#et_cd : Type0) {| scalar et_cd, real_like et_cd |}
  (#m #n : szp)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (wm * tm /?+ bm /\ wn * tn /?+ bn))
  (#_ : squash (SZ.fits ((rm m n).ulen)))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (bid : natlt nblk)
  (rBlock : chest2 real bm bn)
  requires
    forall+ (wid : natlt (nthr / warp_size)).
      exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
        warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
          (wm * tm) (wn * tn) wid |-> eWarp **
        pure (eWarp %~
          ematrix_subtile rBlock (wm * tm) (wn * tn)
            (wid / (bn / (wn * tn)))
            (wid % (bn / (wn * tn))))
  ensures
    exists* (eBlock : chest2 et_cd bm bn).
      block_tile gD (SZ.v bm) (SZ.v bn) bid |-> eBlock **
      pure (eBlock %~ rBlock)
