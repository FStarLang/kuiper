module Kuiper.Softmax.F32

#lang-pulse
open Kuiper
open Kuiper.Softmax

[@@CPrologue "__global__"]
val k_pointwise_exp_f32 : k_pointwise_exp_ty f32
let k_pointwise_exp_f32 = k_pointwise_exp 

[@@CPrologue "__global__"]
val k_pointwise_div_f32 : k_pointwise_div_ty f32
let k_pointwise_div_f32 = k_pointwise_div

// val softmax_gpu : softmax_gpu_ty f32
// let softmax_gpu = softmax_gpu k_pointwise_exp_f32 k_pointwise_div_f32 Kuiper.HReduceF32Plus.k_reduce

let softmax     = softmax     k_pointwise_exp_f32 k_pointwise_div_f32 Kuiper.HReduceF32Plus.k_reduce
