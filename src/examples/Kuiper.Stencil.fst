module Kuiper.Stencil

#lang-pulse

open Kuiper
open Kuiper.Matrix.Reprs { row_major, col_major }
open Kuiper.Poly.Stencil

let stencil3x3_f32_add_rr (#rows #cols : (x:szp{x >= 3}))
=
  specialize_host_simple_stencil f32 (fun _ _ -> one) row_major row_major 
    #_ #_ #rows #cols

open FStar.UInt32

inline_for_extraction noextract
let stencil i j =
  match i, j with
  | 1, 1 -> 8ul
  | _ -> 1ul

let stencil3x3_i32_add_mul2_rc (#rows #cols : (x:szp{x >= 3}))
=
  specialize_host_simple_stencil u32 stencil row_major col_major 
    #_ #_ #rows #cols
