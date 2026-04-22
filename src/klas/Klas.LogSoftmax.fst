module Klas.LogSoftmax

#lang-pulse
open Kuiper

module K = Kuiper.Kernel.LogSoftmax

let log_softmax_f16 : K.log_softmax_ty f16 = K.log_softmax
let log_softmax_f32 : K.log_softmax_ty f32 = K.log_softmax
let log_softmax_f64 : K.log_softmax_ty f64 = K.log_softmax
