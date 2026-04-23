module Klas.LogSoftmax

#lang-pulse
open Kuiper

module K = Kuiper.Kernel.LogSoftmax

(* dynamic thread number *)
let log_softmax_n_f16 : K.log_softmax_ty f16 = K.log_softmax
let log_softmax_n_f32 : K.log_softmax_ty f32 = K.log_softmax
let log_softmax_n_f64 : K.log_softmax_ty f64 = K.log_softmax

(* full blocks. TODO: would it help to use smin lena 1024? *)
let log_softmax_f16 lena = K.log_softmax #f16 1024sz #lena
let log_softmax_f32 lena = K.log_softmax #f32 1024sz #lena
let log_softmax_f64 lena = K.log_softmax #f64 1024sz #lena
