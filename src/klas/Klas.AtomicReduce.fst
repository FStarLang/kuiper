module Klas.AtomicReduce

#lang-pulse
open Kuiper
module K = Kuiper.Kernel.AtomicReduce

let reduce_u32 : K.reduce_ty u32 l1_forward = K.reduce
let reduce_u64 : K.reduce_ty u64 l1_forward = K.reduce
let reduce_f32 : K.reduce_ty f32 l1_forward = K.reduce
let reduce_f64 : K.reduce_ty f64 l1_forward = K.reduce
