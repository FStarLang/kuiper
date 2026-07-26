module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.TeardownOutput

#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Kernel.GEMM.Tiled.Common.Vec

module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

ghost
fn gather_output
  (#et_cd : Type0) {| scalar et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (wm * tm /?+ bm /\ wn * tn /?+ bn))
  (#_ : squash (tm /?+ (wm * tm) /\ tn /?+ (wn * tn)))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  ()
  norewrite
  requires
    (forall+ (bid : natlt nblk) (tid : natlt nthr).
      output_lane_approximates gD bm bn tm tn wm wn bid tid
        (ematrix_subtile
          (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn))))) **
    pure (SZ.fits ((rm m n).ulen))
  ensures
    exists* (eD : chest2 et_cd m n).
      gD |-> eD ** pure (eD %~ MS.mmcomb comb_r rC rA rB)
