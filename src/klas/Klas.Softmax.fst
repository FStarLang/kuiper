module Klas.Softmax

#lang-pulse
open Kuiper

module K = Kuiper.Kernel.Softmax

(* GPU-pointer variants, dynamic thread number *)
let softmax_gpu_n_f16 : K.softmax_gpu_ty f16 = K.softmax_gpu
let softmax_gpu_n_f32 : K.softmax_gpu_ty f32 = K.softmax_gpu
let softmax_gpu_n_f64 : K.softmax_gpu_ty f64 = K.softmax_gpu

(* GPU-pointer variants, full blocks *)
let softmax_gpu_f16 lena = K.softmax_gpu #f16 1024sz #lena
let softmax_gpu_f32 lena = K.softmax_gpu #f32 1024sz #lena
let softmax_gpu_f64 lena = K.softmax_gpu #f64 1024sz #lena

(* dynamic thread number *)
let softmax_n_f16 : K.softmax_ty f16 = K.softmax
let softmax_n_f32 : K.softmax_ty f32 = K.softmax
let softmax_n_f64 : K.softmax_ty f64 = K.softmax

(* full blocks. TODO: would it help to use smin lena 1024? *)
let softmax_f16 lena = K.softmax #f16 1024sz #lena
let softmax_f32 lena = K.softmax #f32 1024sz #lena
let softmax_f64 lena = K.softmax #f64 1024sz #lena
