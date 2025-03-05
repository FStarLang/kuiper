module Kuiper.HReduceF32Plus

#lang-pulse

open Kuiper
open Kuiper.HReduce

[@@CPrologue "__device__"] let d_reduce = d_reduce
[@@CPrologue "__global__"] let k_reduce = k_reduce
let reduce = reduce k_reduce
