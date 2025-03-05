module Kuiper.Softmax.F64

#lang-pulse
open Kuiper
open Kuiper.Softmax

let k_pointwise_exp_f64 = k_pointwise_exp 
let k_pointwise_div_f64 = k_pointwise_div

// let softmax_gpu = softmax_gpu k_pointwise_exp_f64 k_pointwise_div_f64 Kuiper.HReduceF64Plus.k_reduce
let softmax     = softmax     k_pointwise_exp_f64 k_pointwise_div_f64 Kuiper.HReduceF64Plus.k_reduce

