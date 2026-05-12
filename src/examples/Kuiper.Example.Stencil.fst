module Kuiper.Example.Stencil

#lang-pulse

open Kuiper
open Kuiper.Tensor.Layout.Alg { l2_row_major, l2_col_major }
open Kuiper.Kernel.Stencil

let stencil3x3_f32_add_rr (#rows #cols : (x:szp{x >= 3})) =
  specialize_host_simple_stencil
    f32
    (fun _ _ -> one)
    l2_row_major l2_row_major
    rows cols

inline_for_extraction noextract
let stencil i j =
  match i, j with
  | 1, 1 -> 8ul
  | _ -> 1ul

let stencil3x3_i32_add_mul2_rc (#rows #cols : (x:szp{x >= 3})) =
  specialize_host_simple_stencil
    u32
    stencil
    l2_row_major l2_col_major
    rows cols
