module Klas.UGS.SwiGLU

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore
open Kuiper.Float.Casts.Base
open Kuiper.EMatrix

open Klas.UGS.Projection
open Kuiper.Kernel.UGS.Epilogue

module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

fn swiglu
  (m k n : szp)
  (#_ : squash (n % pair_group_n == 0))
  (#_ : squash (SZ.fits (2 * n)))
  (#_ : squash (SZ.fits (m * k)))
  (#_ : squash (SZ.fits (k * (2 * n))))
  (#_ : squash (SZ.fits (m * (2 * n))))
  (#_ : squash (m % 64 == 0))
  (#_ : squash (k % 16 == 0))
  (#_ : squash ((m / 64) * ((2 * n) / 64) <= max_blocks))
  (gA : array2 half (rm m k) { is_global gA })
  (gW : array2 half (rm k (2 * n)) { is_global gW })
  (gOut : array2 half (rm m n) { is_global gOut })
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gW)))
  (#eA : chest2 half m k)
  (#eW : chest2 half k (2 * n))
  (#eOut0 : chest2 half m n)
  (#fA #fW : perm)
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gW |-> Frac fW eW)
  requires
    on gpu_loc (gOut |-> eOut0)
  ensures
    (exists* (eOut : chest2 half m n).
      on gpu_loc (gOut |-> eOut) **
      pure (eOut %~ real_epilogue n
        (MS.matmul (to_real_matrix eA) (to_real_matrix eW))))
{
  let gP =
    Kuiper.Tensor.alloc0 #half
      (SZ.mul m (SZ.mul 2sz n))
      (rm m (SZ.mul 2sz n));
  let gPf32 =
    Kuiper.Tensor.alloc0 #float
      (SZ.mul m (SZ.mul 2sz n))
      (rm m (SZ.mul 2sz n));
  projection m k (SZ.mul 2sz n) gA gW gPf32 gP;

  with eC'. assert on gpu_loc (gPf32 |-> eC');
  let eP : chest2 half m (2 * n) =
    chest_map cast_f32_to_f16 eC';
  Kuiper.Tensor.free gPf32;

  launch_kernel_1
    (fun _ ->
      epilogue #m #n #(SZ.mul 2sz n) gP gOut #1.0R
        #eP #eOut0);

  with eOut. assert on gpu_loc (gOut |-> eOut);
  assert pure (
    forall (r : natlt m) (c : natlt n).
      acc2 eOut r c ==
        epilogue_cell
          (acc2 eP r (gate_col_idx n c))
          (acc2 eP r (up_col_idx n c)));
  let rP : chest2 real m (2 * n) =
    MS.matmul (to_real_matrix eA) (to_real_matrix eW);
  epilogue_matrix_approx n eP rP eOut;
  Kuiper.Tensor.free gP #eP;
  assert pure (
    eOut %~ real_epilogue n
      (MS.matmul (to_real_matrix eA) (to_real_matrix eW)));

  ()
}
