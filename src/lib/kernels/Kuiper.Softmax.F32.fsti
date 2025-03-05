module Kuiper.Softmax.F32

#lang-pulse
open Kuiper
open Kuiper.Softmax

[@@CPrologue "__global__"]
val k_pointwise_exp_f32 : k_pointwise_exp_ty f32
[@@CPrologue "__global__"]
val k_pointwise_div_f32 : k_pointwise_div_ty f32

// val softmax_gpu : softmax_gpu_ty f32
val softmax     : softmax_ty f32
