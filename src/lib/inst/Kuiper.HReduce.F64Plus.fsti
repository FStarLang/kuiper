module Kuiper.HReduce.F64Plus

#lang-pulse

open Kuiper
open Kuiper.HReduce

let size = 1024sz (* crutch *)

val reduce : reduce_ty f64
