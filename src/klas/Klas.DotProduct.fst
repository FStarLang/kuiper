module Klas.DotProduct

#lang-pulse

open Kuiper
module K = Kuiper.Kernel.DotProduct

let dotprod_f32 : K.dotprod_ty f32 = K.dotprod
let dotprod_f64 : K.dotprod_ty f64 = K.dotprod
let dotprod_u32 : K.dotprod_ty u32 = K.dotprod
let dotprod_u64 : K.dotprod_ty u64 = K.dotprod
