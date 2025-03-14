module Kuiper.HReduce.U32Plus

#lang-pulse
inline_for_extraction let () = ()

open Kuiper
open Kuiper.HReduce

let size = 1024sz (* crutch *)

val reduce : reduce_ty u32
