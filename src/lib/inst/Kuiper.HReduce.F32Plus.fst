module Kuiper.HReduce.F32Plus

#lang-pulse

open Kuiper
open Kuiper.HReduce

[@@CPrologue "__device__"; "KrmlPrivate"]
let k_reduce = d_reduce
let reduce = reduce k_reduce
