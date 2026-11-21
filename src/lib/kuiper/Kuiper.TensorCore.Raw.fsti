module Kuiper.TensorCore.Raw

open Kuiper.Chest

(* Representation-level MMA, with no real-approximation assumptions. *)
val emma
  (#et0 #et1:Type)
  (#rows #shared #columns:nat)
  (mc:chest2 et1 rows columns)
  (ma:chest2 et0 rows shared)
  (mb:chest2 et0 shared columns)
  : chest2 et1 rows columns
