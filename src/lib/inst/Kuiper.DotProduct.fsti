module Kuiper.DotProduct

#lang-pulse

open Kuiper
module P = Kuiper.Poly.DotProduct

val dotprod_f32 : P.dotprod_ty f32
val dotprod_f64 : P.dotprod_ty f64
val dotprod_u32 : P.dotprod_ty u32
val dotprod_u64 : P.dotprod_ty u64
