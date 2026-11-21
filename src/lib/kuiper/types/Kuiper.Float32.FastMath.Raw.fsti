module Kuiper.Float32.FastMath.Raw

open Kuiper.Floating.Base
open Kuiper.Float32.Raw
module F = Kuiper.Float32.Raw

val sin : t -> t
val cos : t -> t
val log : t -> t
val exp : t -> t
val divide : t -> t -> t
val fma_rn : t -> t -> t -> t

(* Trusted representation behavior of __fsub_rn, without approximation facts. *)
val sub_rn (x y:t) : Tot (result:t{
  match kind x, kind y with
  | NaN, _ | _, NaN -> NaN? (kind result)
  | Infinite, Finite -> bit_eq result x
  | Finite, Infinite -> bit_eq result (F.neg y)
  | Infinite, Infinite ->
      (bit_eq x y ==> NaN? (kind result)) /\
      (~(bit_eq x y) ==> bit_eq result x)
  | Finite, Finite ->
      (x == y ==> result == F.zero) /\
      (bit_eq x y ==> bit_eq result F.zero) /\
      (is_zero x /\ is_zero y /\ ~(bit_eq x y) ==> bit_eq result x)})
