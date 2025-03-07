module Kuiper.AtomicReduce.U32

#lang-pulse
open Kuiper
open Kuiper.AtomicReduce

[@@CPrologue "__global__"]
let kernel : kernel_ty u32 = kernel #u32

let reduce : reduce_ty u32 = reduce kernel
