module Klas.LogSoftmax

#lang-pulse
open Kuiper

module P = Kuiper.Poly.LogSoftmax

let log_softmax_f16 : P.log_softmax_ty f16 = P.log_softmax
let log_softmax_f32 : P.log_softmax_ty f32 = P.log_softmax
let log_softmax_f64 : P.log_softmax_ty f64 = P.log_softmax
