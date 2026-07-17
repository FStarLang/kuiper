module Kuiper.Kernel.Scatter

#lang-pulse
open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor

module SZ = Kuiper.SizeT

let vscatter (#et: Type0) (#r : erased nat) 
  (di do: shape r {shape_le di do}) (dim : natlt r)
  (eInp : chest di et) (eIdx : chest di (szlt (do @! dim))) (eOut : chest do et)
  (i : abs di)
  : prop 
  = acc eOut (abs_set_at2 di do dim (acc eIdx i) i) == acc eInp i

let vscatter_chest
  (#et : Type0)
  (#r : erased nat)
  (di do: shape r {shape_le di do})
  (dim : natlt r)
  (eInp : chest di et) 
  (eIdx : chest di (szlt (do @! dim)))
  (eOut : chest do et): prop
  = chest_foralli (fun i _ -> vscatter di do dim eInp eIdx eOut i) eInp

// Injectivity of the index map, as a refinement on the index chest. The
// scattered cells of `out` are the images of `phi i = abs_set_at2 di do dim
// (acc eIdx i) i`; correctness (no write conflicts) relies on this map being
// injective, which holds iff `eIdx` is injective. PyTorch leaves scatter with
// duplicate indices nondeterministic, so at the trusted C boundary this is an
// assumption: the predicate is erased at extraction, hence requires no runtime
// witness.
let chest_inj
  (#r : erased nat)
  (di do: shape r {shape_le di do})
  (dim : natlt r)
  (eIdx : chest di (szlt (do @! dim))): prop
  = forall (i j : abs di). acc eIdx i == acc eIdx j ==> i == j

inline_for_extraction noextract
fn scatter_gpu
  (#et : Type0) (#r : erased nat) (di do : shape r { shape_le di do }) (cdi: cshape di) (cdo: cshape do)
  (dim: szlt r)
  (#lInp #lIdx: tlayout di) (#lOut: tlayout do) {| ctlayout lInp, ctlayout lIdx, ctlayout lOut |}
  (gInp: tensor et lInp {is_global gInp})
  (gIdx: tensor (szlt (do @! (SZ.v dim))) lIdx {is_global gIdx})
  (gOut: tensor et lOut {is_global gOut})
  (n : sz{SZ.v n == sizeof di /\ n <= max_blocks * max_threads /\ n > 0})
  (eInp: chest di et)
  (eIdx: chest di (szlt (do @! (SZ.v dim))) { chest_inj di do (SZ.v dim) eIdx })
  (#fInp #fIdx: perm)
  preserves cpu ** on gpu_loc (gInp |-> Frac fInp eInp) ** on gpu_loc (gIdx |-> Frac fIdx (eIdx <: chest di (szlt (do @! (SZ.v dim)))))
  requires on gpu_loc (live gOut)
  ensures 
    exists* eOut. 
      on gpu_loc (gOut |-> eOut) **
      pure (vscatter_chest di do dim eInp eIdx eOut)