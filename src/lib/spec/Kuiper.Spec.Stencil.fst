module Kuiper.Spec.Stencil

open Kuiper
open Kuiper.Chest

let neighborhood_at
  (#et : Type0)
  (#rows #cols : nat)
  (eIn: chest2 et (rows + 2) (cols + 2))
  (i: nat{i > 0 /\ i <= rows})
  (j: nat{j > 0 /\ j <= cols})
  : GTot (seq et)
  =
  seq![
    (acc2 eIn (i - 1) (j - 1)); (acc2 eIn (i - 1) j); (acc2 eIn (i - 1) (j + 1));
    (acc2 eIn i (j - 1)); (acc2 eIn i j); (acc2 eIn i (j + 1));
    (acc2 eIn (i + 1) (j - 1)); (acc2 eIn (i + 1) j); (acc2 eIn (i + 1) (j + 1))
  ]

let stencil_result_at_idx
  (#et:Type0) {| scalar et |}
  (#rows #cols : nat)
  (stencil: natlt 3 -> natlt 3 -> et)
  (eIn: chest2 et (rows + 2) (cols + 2))
  (i: natlt rows)
  (j: natlt cols)
  : GTot et
  =
  let neighbor = neighborhood_at eIn (i + 1) (j + 1)
  in
    ((neighbor @! 0) `mul` (stencil 0 0)) `add` ((neighbor @! 1) `mul` (stencil 0 1)) `add` ((neighbor @! 2) `mul` (stencil 0 2)) `add`
    ((neighbor @! 3) `mul` (stencil 1 0)) `add` ((neighbor @! 4) `mul` (stencil 1 1)) `add` ((neighbor @! 5) `mul` (stencil 1 2)) `add`
    ((neighbor @! 6) `mul` (stencil 2 0)) `add` ((neighbor @! 7) `mul` (stencil 2 1)) `add` ((neighbor @! 8) `mul` (stencil 2 2))

let stencil_result
  (#et:Type0) {| scalar et |}
  (#rows #cols : nat)
  (stencil: natlt 3 -> natlt 3 -> et)
  (eIn: chest2 et (rows + 2) (cols + 2))
  : chest2 et rows cols
  =
  mk2 <| fun i j -> stencil_result_at_idx stencil eIn i j
