module Kuiper.Softmax.F32

#lang-pulse
open Kuiper
open Kuiper.Softmax

let k_pointwise_exp_f32 = k_pointwise_exp 
let k_pointwise_div_f32 = k_pointwise_div

// let softmax_gpu = softmax_gpu k_pointwise_exp_f32 k_pointwise_div_f32 Kuiper.HReduceF32Plus.k_reduce
let softmax     = softmax     k_pointwise_exp_f32 k_pointwise_div_f32 Kuiper.HReduceF32Plus.k_reduce

