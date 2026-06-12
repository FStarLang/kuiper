module Klas.Elementwise

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.Map
open Kuiper.Tensor.Layout.Alg { l1_forward }

let silu_fw_bf16   lena a #s            = map_gpu silu_step lena a
let neg_fw_bf16    lena a #s            = map_gpu neg_step lena a
let rsqrt_fw_f32   lena a #s            = map_gpu rsqrt_step lena a
let square_fw_f32  lena a #s            = map_gpu square_step lena a
let cos_fw_f32     lena a #s            = map_gpu cos_step lena a
let sin_fw_f32     lena a #s            = map_gpu sin_step lena a

let add_fw_bf16    lena a b #sa #sb #fb = map_gpu2 add_step lena a b
let mul_fw_bf16    lena a b #sa #sb #fb = map_gpu2 mul_step lena a b
let mul_fw_f32     lena a b #sa #sb #fb = map_gpu2 mul_step lena a b

let add_const_fw_f32 c lena a #s        = map_gpu (add_const_step c) lena a
let mul_const_fw_f32 c lena a #s        = map_gpu (mul_const_step c) lena a
