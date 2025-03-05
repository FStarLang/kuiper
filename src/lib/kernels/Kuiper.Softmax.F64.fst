module Kuiper.Softmax.F64

#lang-pulse
open Kuiper
open Kuiper.Softmax

[@@CPrologue "__global__"]
val k_pointwise_exp_f64 : k_pointwise_exp_ty f64
let k_pointwise_exp_f64 = k_pointwise_exp 

[@@CPrologue "__global__"]
val k_pointwise_div_f64 : k_pointwise_div_ty f64
let k_pointwise_div_f64 = k_pointwise_div

// val softmax_gpu : softmax_gpu_ty f64
// let softmax_gpu = softmax_gpu k_pointwise_exp_f64 k_pointwise_div_f64 Kuiper.HReduceF64Plus.k_reduce

let softmax     = softmax     k_pointwise_exp_f64 k_pointwise_div_f64 Kuiper.HReduceF64Plus.k_reduce
