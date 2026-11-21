module Kuiper.TensorCore.Model

open Kuiper
open Kuiper.Chest
open Kuiper.Spec.GEMM

val emma
  (#et0 #et1:Type0)
  (#rows #shared #columns:nat)
  (mc:chest2 et1 rows columns)
  (ma:chest2 et0 rows shared)
  (mb:chest2 et0 shared columns)
  : chest2 et1 rows columns

(* Evidence is tied to the chosen scalar interpretation, not just its type. *)
class mma_approximation
  (et0 et1:Type0)
  {| scalar et0, real_like et0, scalar et1, real_like et1 |}
= {
  mma_approx :
    #rows:nat -> #shared:nat -> #columns:nat ->
    (mc:chest2 et1 rows columns) ->
    (ma:chest2 et0 rows shared) ->
    (mb:chest2 et0 shared columns) ->
    (rc:chest2 real rows columns) ->
    (ra:chest2 real rows shared) ->
    (rb:chest2 real shared columns) ->
    Lemma (requires ma %~ ra /\ mb %~ rb /\ mc %~ rc)
          (ensures emma mc ma mb %~ matplus rc (matmul ra rb));
}

instance val mma_half_half : mma_approximation half half
instance val mma_half_float : mma_approximation half float
instance val mma_bf16_float : mma_approximation bf16 float
instance val mma_double_double : mma_approximation double double

let emma_approx_lemma
  (#et0 #et1:Type0)
  {| scalar et0, real_like et0, scalar et1, real_like et1,
     mma_approximation et0 et1 |}
  (#rows #shared #columns:nat)
  (mc:chest2 et1 rows columns)
  (ma:chest2 et0 rows shared)
  (mb:chest2 et0 shared columns)
  (rc:chest2 real rows columns)
  (ra:chest2 real rows shared)
  (rb:chest2 real shared columns)
  : Lemma (requires ma %~ ra /\ mb %~ rb /\ mc %~ rc)
          (ensures emma mc ma mb %~ matplus rc (matmul ra rb))
  = mma_approx mc ma mb rc ra rb
