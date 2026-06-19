module Klas.SDPA.LSE

#lang-pulse
open Kuiper
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor
open Kuiper.Index
open Kuiper.Bijection

open Kuiper.Kernel.SDPA.LSE
module SZ = Kuiper.SizeT

let sdpa_lse_naive_bf16
  (n h: szp) 
  (l s: szp) 
  (e ev: szp { SZ.fits (n * h * l * e) /\ SZ.fits (n * h * s * e) /\ SZ.fits (n * h * s * ev) /\ SZ.fits (n * h * l * s) }) = 
  sdpa_lse_naive #bf16 n h l s e ev 
  #(l4_batched_row_major n h l e)
  #(l4_batched_row_major n h s e)
  #(l4_batched_row_major n h s ev)
  #(l4_batched_row_major n h l s)
  #(c_l4_batched_row_major _ h l e)
  #(c_l4_batched_row_major _ h s e)
  #(c_l4_batched_row_major _ h s ev)
  #(c_l4_batched_row_major _ h l s)