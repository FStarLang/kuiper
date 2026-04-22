module Klas.DotProduct

#lang-pulse

open Kuiper
module K = Kuiper.Kernel.DotProduct

val dotprod_f32 : K.dotprod_ty f32
val dotprod_f64 : K.dotprod_ty f64
val dotprod_u32 : K.dotprod_ty u32
val dotprod_u64 : K.dotprod_ty u64
