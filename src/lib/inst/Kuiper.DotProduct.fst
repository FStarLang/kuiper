module Kuiper.DotProduct

#lang-pulse

open Kuiper
module P = Kuiper.Poly.DotProduct

let dotprod_f32 : P.dotprod_ty f32 = P.dotprod
let dotprod_f64 : P.dotprod_ty f64 = P.dotprod
let dotprod_u32 : P.dotprod_ty u32 = P.dotprod
let dotprod_u64 : P.dotprod_ty u64 = P.dotprod
