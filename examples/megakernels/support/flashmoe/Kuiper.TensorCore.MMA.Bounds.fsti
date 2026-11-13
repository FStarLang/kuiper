module Kuiper.TensorCore.MMA.Bounds

open Kuiper
module B = Kuiper.Floating.Bounds
module C = Kuiper.Chest
module MMA = Kuiper.TensorCore.MMA

(* One native m16n8k16 BF16 MMA, not a scalar RN-FMA identity. The operand
   envelopes 32 and 4 make each product's bound exactly representable even
   in BF16. The accumulator envelope is exact in binary32. No accumulation
   order, rounding mode, subnormal preservation or warp ownership is added. *)
val bounded_emma
  (bound:B.bound{bound + 2048 <= B.limit})
  (mc:C.chest2 float 16 8)
  (ma:C.chest2 bf16 16 16)
  (mb:C.chest2 bf16 16 8)
  : Lemma
      (requires C.chest_forall (B.magnitude bound) mc /\
                C.chest_forall (B.magnitude 32) ma /\
                C.chest_forall (B.magnitude 4) mb)
      (ensures C.chest_forall (B.magnitude (bound + 2048)) (MMA.emma mc ma mb))
