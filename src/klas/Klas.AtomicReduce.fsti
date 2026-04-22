module Klas.AtomicReduce

#lang-pulse
open Kuiper
open Kuiper.Tensor.Layout.Alg { l1_forward }
module K = Kuiper.Kernel.AtomicReduce

val reduce_u32 : K.reduce_ty u32 l1_forward
val reduce_u64 : K.reduce_ty u64 l1_forward
val reduce_f32 : K.reduce_ty f32 l1_forward
val reduce_f64 : K.reduce_ty f64 l1_forward
