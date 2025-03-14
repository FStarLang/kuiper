module Kuiper.AtomicReduce

#lang-pulse
open Kuiper
module P = Kuiper.Poly.AtomicReduce

let reduce_u32 : P.reduce_ty u32 = P.reduce
let reduce_u64 : P.reduce_ty u64 = P.reduce
let reduce_f32 : P.reduce_ty f32 = P.reduce
let reduce_f64 : P.reduce_ty f64 = P.reduce
