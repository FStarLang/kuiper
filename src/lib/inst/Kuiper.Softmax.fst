module Kuiper.Softmax

#lang-pulse
open Kuiper

module P = Kuiper.Poly.Softmax

(* This clearly works, but fails to compile in some CUDA configs with
   error: identifier "__hdiv" is undefined *)
// let softmax_f16 : P.softmax_ty f16 = P.softmax

let softmax_f32 : P.softmax_ty f32 = P.softmax
let softmax_f64 : P.softmax_ty f64 = P.softmax
