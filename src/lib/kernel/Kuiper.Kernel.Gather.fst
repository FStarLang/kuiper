module Kuiper.Kernel.Gather

#lang-pulse
open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Shareable

open Kuiper.Kernel.TMap

module SZ = Kuiper.SizeT

let vfgather (#et: Type0) (#r : erased nat) (di do : shape r { shape_le di do }) (dim : natlt r)
  (eInp : chest do et) (eIdx : chest di (szlt (do @! dim)))
  (x : abs di) (_: et) (o: et)
  : prop 
  = o == acc eInp (abs_set_at2 di do dim (acc eIdx x) x)

unfold
let gather_frame (#et : Type0) (#r : erased nat) (di do : shape r { shape_le di do }) (cdi: cshape di)
  (dim: szlt r)
  (#lInp: tlayout do) (#lIdx: tlayout di)  {| ctlayout lInp, ctlayout lIdx |}
  (gInp: tensor et lInp {is_global gInp})
  (gIdx: tensor (szlt (do @! (SZ.v dim))) lIdx {is_global gIdx})
  (eInp: chest do et)
  (eIdx: chest di (szlt (do @! (SZ.v dim))))
  (fInp fIdx: perm)
  (fr: perm): slprop =
    (tensor_pts_to gInp #(fInp *. fr) eInp) **
    (tensor_pts_to gIdx #(fIdx *. fr) eIdx)

instance gather_frame_shareable
  (#et : Type0) (#r : erased nat) (di do : shape r { shape_le di do }) (cdi: cshape di)
  (dim: szlt r)
  (#lInp: tlayout do) (#lIdx: tlayout di)  {| ctlayout lInp, ctlayout lIdx|}
  (gInp: tensor et lInp {is_global gInp})
  (gIdx: tensor (szlt (do @! (SZ.v dim))) lIdx {is_global gIdx})
  (eInp: chest do et)
  (eIdx: chest di (szlt (do @! (SZ.v dim))))
  (fInp fIdx: perm):
  shareable (gather_frame di do cdi dim #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx) =
    double_shareable 
    (fun fr -> tensor_pts_to gInp #fr eInp) 
    (fun fr -> tensor_pts_to gIdx #fr eIdx)
    fInp fIdx

inline_for_extraction noextract
fn fgather (#et : Type0) (#r : erased nat) (di do : shape r { shape_le di do }) (cdi: cshape di) (cdo: cshape do)
  (dim: szlt r)
  (#lInp: tlayout do) (#lIdx #lOut: tlayout di)  {| ctlayout lInp, ctlayout lIdx, ctlayout lOut |}
  (gInp: tensor et lInp {is_global gInp})
  (gIdx: tensor (szlt (do @! (SZ.v dim))) lIdx {is_global gIdx})
  (gOut: tensor et lOut {is_global gOut})
  (eInp: chest do et)
  (eIdx: chest di (szlt (do @! (SZ.v dim))))
  (#fInp #fIdx: perm)
  (#fr: perm) (i: conc di) (x : et) 
norewrite
preserves
  gpu **
  (gather_frame di do cdi dim #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx fr)
returns 
  r: et
ensures
  pure (vfgather di do dim eInp eIdx (up i) x r)
{
  let idx = tensor_read gIdx i;
  let idx' = conc_set_at2 cdi cdo dim idx i;
  let inp = tensor_read gInp idx';

  inp
}

inline_for_extraction noextract
fn gather_gpu
  (#et : Type0) (#r : erased nat) (di do : shape r { shape_le di do }) (cdi: cshape di) (cdo: cshape do)
  (dim: szlt r)
  (#lInp: tlayout do) (#lIdx #lOut: tlayout di)  {| ctlayout lInp, ctlayout lIdx, ctlayout lOut |}
  (gInp: tensor et lInp {is_global gInp})
  (gIdx: tensor (szlt (do @! (SZ.v dim))) lIdx {is_global gIdx})
  (gOut: tensor et lOut {is_global gOut})
  (n : sz{SZ.v n == sizeof di /\ n <= max_blocks * max_threads /\ n > 0})
  (eInp: chest do et)
  (eIdx: chest di (szlt (do @! (SZ.v dim))))
  (#fInp #fIdx: perm)
  preserves cpu ** on gpu_loc (gInp |-> Frac fInp eInp) ** on gpu_loc (gIdx |-> Frac fIdx eIdx)
  requires on gpu_loc (live gOut)
  ensures 
      on gpu_loc (gOut |-> gather_chest di do eInp dim eIdx) {
  with eOut. assert on gpu_loc (gOut |-> eOut);
  launch_sync
    (kmap cdi
      (fun fr ->
        (gInp |-> Frac (fInp *. fr) eInp) **
        (gIdx |-> Frac (fIdx *. fr) eIdx))
      #(double_shareable
        (fun fr -> gInp |-> Frac fr eInp)
        (fun fr -> gIdx |-> Frac fr eIdx)
        fInp fIdx)
      (vfgather di do dim eInp eIdx) 
      (fgather di do cdi cdo dim #lInp #lIdx #lOut gInp gIdx gOut eInp eIdx #fInp #fIdx)
      n gOut #eOut #_ #1.0R);
  with eOut'. assert on gpu_loc (gOut |-> eOut');
  assert pure (Kuiper.Chest.equal eOut' (gather_chest di do eInp dim eIdx)); 
}