module Kuiper.AtomicReduce.F32

#lang-pulse
open Kuiper
open Kuiper.AtomicReduce

[@@CPrologue "__device__"; "KrmlPrivate"]
let kernel : kernel_ty f32 = kernel #f32

let reduce : reduce_ty f32 = reduce kernel
