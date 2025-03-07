module Kuiper.HReduce.F64Plus

#lang-pulse
inline_for_extraction let () = ()

open Kuiper
open Kuiper.HReduce

let size = 1024sz (* crutch *)

val k_reduce : k_reduce_ty f64
val reduce : reduce_ty f64
