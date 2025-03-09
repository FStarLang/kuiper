module Kuiper.HReduce.U64Plus

#lang-pulse

open Kuiper
open Kuiper.HReduce

[@@CPrologue "__global__"; "KrmlPrivate"]
let k_reduce = d_reduce
let reduce = reduce k_reduce
