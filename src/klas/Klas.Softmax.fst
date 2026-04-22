module Klas.Softmax

#lang-pulse
open Kuiper

module K = Kuiper.Kernel.Softmax

let softmax_f16 : K.softmax_ty f16 = K.softmax
let softmax_f32 : K.softmax_ty f32 = K.softmax
let softmax_f64 : K.softmax_ty f64 = K.softmax
