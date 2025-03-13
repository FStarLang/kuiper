module Kuiper.Softmax.F16

#lang-pulse
open Kuiper
open Kuiper.Softmax

[@@CPrologue "__device__"; "KrmlPrivate"]
val k_pointwise_exp_f16 : k_pointwise_exp_ty f16
let k_pointwise_exp_f16 = k_pointwise_exp

[@@CPrologue "__device__"; "KrmlPrivate"]
val k_pointwise_div_f16 : k_pointwise_div_ty f16
let k_pointwise_div_f16 = k_pointwise_div

// val softmax_gpu : softmax_gpu_ty f16
// let softmax_gpu = softmax_gpu k_pointwise_exp_f16 k_pointwise_div_f16 Kuiper.HReduceF16Plus.k_reduce

let softmax     = softmax     k_pointwise_exp_f16 k_pointwise_div_f16 Kuiper.HReduce.F16Plus.k_reduce
