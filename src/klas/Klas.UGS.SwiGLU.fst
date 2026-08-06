module Klas.UGS.SwiGLU

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore
open Kuiper.Float.Casts.Base
open Kuiper.EMatrix

open Klas.UGS.Projection
open Kuiper.Kernel.UGS.Epilogue

module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

inline_for_extraction noextract
fn swiglu
  (m k n : szp)
  (#lGate #lUp : layout2 k n)
  {| ctlayout lGate, ctlayout lUp |}
  (#_ : squash (SZ.fits (m * k)))
  (#_ : squash (SZ.fits (k * n)))
  (#_ : squash (SZ.fits (m * n)))
  (#_ : squash (m % 64 == 0))
  (#_ : squash (k % 16 == 0))
  (#_ : squash (n % 64 == 0))
  (#_ : squash ((m / 64) * (n / 64) <= max_blocks))
  (gA : array2 half (rm m k) { is_global gA })
  (gWGate : array2 half lGate { is_global gWGate })
  (gWUp : array2 half lUp { is_global gWUp })
  (gOut : array2 half (rm m n) { is_global gOut })
  (#_ : squash (aligned 16 (core gA)))
  (#eA : chest2 half m k)
  (#eWGate #eWUp : chest2 half k n)
  (#eOut0 : chest2 half m n)
  (#fA #fWGate #fWUp : perm)
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gWGate |-> Frac fWGate eWGate) **
    on gpu_loc (gWUp |-> Frac fWUp eWUp)
  requires
    on gpu_loc (gOut |-> eOut0)
  ensures
    (exists* (eOut : chest2 half m n).
      on gpu_loc (gOut |-> eOut) **
      pure (eOut %~ chest_comb real_swiglu
        (MS.matmul (to_real_matrix eA) (to_real_matrix eWGate))
        (MS.matmul (to_real_matrix eA) (to_real_matrix eWUp))))
{
  let gPGate =
    Kuiper.Tensor.alloc0 #half
      (SZ.mul m n)
      (rm m n);
  let gPUp =
    Kuiper.Tensor.alloc0 #half
      (SZ.mul m n)
      (rm m n);
  let gPf32 =
    Kuiper.Tensor.alloc0 #float
      (SZ.mul m n)
      (rm m n);
  projection m k n gA gWGate gPf32 gPGate;

  with eCGate. assert on gpu_loc (gPf32 |-> eCGate);
  let ePGate : chest2 half m n = chest_map cast_f32_to_f16 eCGate;

  projection m k n gA gWUp gPf32 gPUp;
  with eCUp. assert on gpu_loc (gPf32 |-> eCUp);
  let ePUp : chest2 half m n = chest_map cast_f32_to_f16 eCUp;

  launch_kernel_1
    (fun _ ->
      epilogue #m #n gPGate gPUp gOut #1.0R #1.0R
        #ePGate #ePUp #eOut0);

  with eOut. assert on gpu_loc (gOut |-> eOut);
  assert pure (
    forall (r : natlt m) (c : natlt n).
      acc2 eOut r c ==
        epilogue_cell (acc2 ePGate r c) (acc2 ePUp r c));
  let rGate : chest2 real m n =
    MS.matmul (to_real_matrix eA) (to_real_matrix eWGate);
  let rUp : chest2 real m n =
    MS.matmul (to_real_matrix eA) (to_real_matrix eWUp);
  epilogue_matrix_approx ePGate ePUp rGate rUp eOut;
  Kuiper.Tensor.free gPf32 #eCUp;
  Kuiper.Tensor.free gPGate #ePGate;
  Kuiper.Tensor.free gPUp #ePUp;
  assert pure (
    eOut %~ chest_comb real_swiglu
      (MS.matmul (to_real_matrix eA) (to_real_matrix eWGate))
      (MS.matmul (to_real_matrix eA) (to_real_matrix eWUp)));

  ()
}
