module Klas.RowScale

#lang-pulse
open Kuiper

module K = Kuiper.Kernel.RowScale
module SZ = Kuiper.SizeT
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg

inline_for_extraction noextract
fn inst (t:Type) {| scalar t|}
  (fla : (len:nat -> layout1 len))
  {| (len:sz -> ctlayout (fla len)) |}
  (flb : (m:nat -> n:nat{SZ.fits (m * n)} -> layout2 m n))
  {| (m:sz -> n:sz{SZ.fits (m * n)} -> ctlayout (flb m n)) |}
  (m n : szp)
  (#_ : squash (m * n <= max_blocks * max_threads))
  (a : array1 t (fla m))
  (b : array2 t (flb m n))
  (#_ : squash (is_global a))
  (#_ : squash (is_global b))
  (#fA : perm)
  (#sa : chest1 t m)
  (#sb : chest2 t m n)
  norewrite
  preserves
    cpu ** on gpu_loc (a |-> Frac fA sa)
  requires
    on gpu_loc (b |-> sb)
  ensures
    on gpu_loc (b |-> K.s_row_scale sa sb)
{
  K.row_scale m n a b;
}

let rowscale_f16_rowmajor = inst f16 l1_forward l2_row_major
let rowscale_f16_colmajor = inst f16 l1_forward l2_col_major
let rowscale_f32_rowmajor = inst f32 l1_forward l2_row_major
let rowscale_f32_colmajor = inst f32 l1_forward l2_col_major
let rowscale_f64_rowmajor = inst f64 l1_forward l2_row_major
let rowscale_f64_colmajor = inst f64 l1_forward l2_col_major
