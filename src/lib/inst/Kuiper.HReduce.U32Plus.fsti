module Kuiper.HReduce.U32Plus

#lang-pulse
inline_for_extraction let () = ()

open Kuiper
open Kuiper.HReduce

let size = 1024sz (* crutch *)

val k_reduce : k_reduce_ty u32
val reduce : reduce_ty u32
