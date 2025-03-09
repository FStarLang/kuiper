module Kuiper.AtomicReduce.U64

#lang-pulse
open Kuiper
open Kuiper.AtomicReduce

[@@CPrologue "__global__"; "KrmlPrivate"]
let kernel : kernel_ty u64 = kernel #u64

let reduce : reduce_ty u64 = reduce kernel
