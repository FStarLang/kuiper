module Kuiper.HReduce.F16Plus

#lang-pulse
inline_for_extraction let () = ()

open Kuiper
open Kuiper.HReduce

let size = 1024sz (* crutch *)

val reduce : reduce_ty f16
