module Klas.Map

#lang-pulse
open Kuiper

module SZ = Kuiper.SizeT
module K = Kuiper.Kernel.Map
module U64 = FStar.UInt64
open Kuiper.Tensor.Layout.Alg

let map_incr =
  K.map_gpu #u64 (fun x -> U64.add_mod x 1uL)
    100sz
    #(l1_forward _)


let map_incr' lena =
  K.map_gpu #u64 (fun x -> U64.add_mod x 1uL)
    lena
    #(l1_forward (SZ.v lena))

