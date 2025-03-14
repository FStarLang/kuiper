module Kuiper.Softmax.Inst

#lang-pulse
open Kuiper
module P = Kuiper.Softmax

val softmax_f16 : P.softmax_ty f16
val softmax_f32 : P.softmax_ty f32
val softmax_f64 : P.softmax_ty f64
