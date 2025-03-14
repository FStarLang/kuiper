module Kuiper.HReduce

#lang-pulse

open Kuiper
module P = Kuiper.Poly.HReduce

let reduce_f16_plus : P.reduce_ty f16 = P.reduce
let reduce_f32_plus : P.reduce_ty f32 = P.reduce
let reduce_f64_plus : P.reduce_ty f64 = P.reduce
let reduce_u32_plus : P.reduce_ty u32 = P.reduce
let reduce_u64_plus : P.reduce_ty u64 = P.reduce
