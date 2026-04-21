module Kuiper.Math.Relu

#lang-pulse
open Kuiper

let relu
  (#et:Type) {| scalar et |}
  (x : et)
  : et
  = if x `gt` zero then x else zero

(* must redefine since real comparisons are admitted. *)
let relu_real
  (x : real)
  : real
  = if t2b (x >. zero) then x else zero

val relu_lem
  (#et:Type) {| scalar et, real_like et |}
  (x : et) (r : real)
  : Lemma (requires x %~ r)
          (ensures relu x %~ relu_real r)
          [SMTPat (relu x); SMTPat (x %~ r)]

