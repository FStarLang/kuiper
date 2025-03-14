module Kuiper.HReduce.Inst

#lang-pulse

open Kuiper
module P = Kuiper.HReduce

val reduce_f16_plus : P.reduce_ty f16
val reduce_f32_plus : P.reduce_ty f32
val reduce_f64_plus : P.reduce_ty f64
val reduce_u32_plus : P.reduce_ty u32
val reduce_u64_plus : P.reduce_ty u64
