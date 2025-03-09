module Kuiper.AtomicReduce.F64

#lang-pulse
open Kuiper
open Kuiper.AtomicReduce

[@@CPrologue "__global__"; "KrmlPrivate"]
let kernel : kernel_ty f64 = kernel #f64

let reduce : reduce_ty f64 = reduce kernel
