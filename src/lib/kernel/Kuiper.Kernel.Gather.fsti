module Kuiper.Kernel.Gather

#lang-pulse
open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor

module SZ = Kuiper.SizeT

let gather_chest 
  (#et : Type0)
  (#r : erased nat)
  (di do : shape r { shape_le di do })
  (eInp : chest do et)
  (dim : natlt r)
  (eIdx : chest di (szlt (do @! dim))): chest di et
  = Kuiper.Chest.mk di
      (fun (x : abs di) ->
         let idx = acc eIdx x in
         let x' = abs_set_at2 di do dim idx x in
         acc eInp x')

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
      on gpu_loc (gOut |-> gather_chest di do eInp dim eIdx)