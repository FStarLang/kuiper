module Klas.Softmax

#lang-pulse
open Kuiper
module K = Kuiper.Kernel.Softmax

val softmax_n_f16 : K.softmax_ty f16
val softmax_n_f32 : K.softmax_ty f32
val softmax_n_f64 : K.softmax_ty f64
