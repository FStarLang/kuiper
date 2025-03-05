module Kuiper.Softmax.F64

#lang-pulse
open Kuiper
open Kuiper.Softmax

[@@CPrologue "__global__"]
val k_pointwise_exp_f64 : k_pointwise_exp_ty f64
[@@CPrologue "__global__"]
val k_pointwise_div_f64 : k_pointwise_div_ty f64

// val softmax_gpu : softmax_gpu_ty f64
val softmax     : softmax_ty f64
