module Kuiper.Example.OnlineSoftmax

#lang-pulse
open Kuiper 
open Kuiper.Array1
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }

module OSMX = Kuiper.Kernel.OnlineSoftmax

let _test (len : szp{len <= max_blocks * max_threads}) =
  OSMX.online_softmax_gpu #f32 1024sz #len #(l1_forward len)

let _testh (len : szp{len <= max_blocks * max_threads}) =
  OSMX.online_softmax_gpu #f16 1024sz #len #(l1_forward len)
