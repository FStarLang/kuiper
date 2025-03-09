module Kuiper.HReduce.F16Plus

#lang-pulse

open Kuiper
open Kuiper.HReduce

[@@CPrologue "__global__"; "KrmlPrivate"]
let k_reduce = d_reduce
let reduce = reduce k_reduce
