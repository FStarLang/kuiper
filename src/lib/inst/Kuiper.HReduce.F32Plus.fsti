module Kuiper.HReduce.F32Plus

#lang-pulse

open Kuiper
open Kuiper.HReduce

let size = 1024sz (* crutch *)

inline_for_extraction noextract
val d_reduce : k_reduce_ty f32
val k_reduce : k_reduce_ty f32
val reduce : reduce_ty f32
