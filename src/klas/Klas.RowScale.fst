module Klas.RowScale

#lang-pulse
open Kuiper

module Array1 = Kuiper.Array1
module Array2 = Kuiper.Array2
module K = Kuiper.Kernel.RowScale
module SZ = Kuiper.SizeT
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Tensor.Layout.Alg

inline_for_extraction noextract
fn inst (t:Type) {| scalar t|}
  (fla : (len:nat -> Array1.layout len))
  {| (len:sz -> ctlayout (fla len)) |}
  (flb : (m:nat -> n:nat{SZ.fits (m * n)} -> Array2.layout m n))
  {| (m:sz -> n:sz{SZ.fits (m * n)} -> ctlayout (flb m n)) |}
  (m n : sz)
  (#_ : squash (m * n <= max_blocks * max_threads))
  // (#la : Array1.layout m) {| ctlayout la |}
  (a : Array1.t t (fla m))
  // (#lb : Array2.layout m n) {| ctlayout lb |}
  (b : Array2.t t (flb m n))
  (#_ : squash (Array1.is_global a))
  (#_ : squash (Array2.is_global b))
  (#fA : perm)
  (#sa : erased (lseq t m))
  (#sb : ematrix t m n)
  norewrite
  preserves
    cpu ** on gpu_loc (a |-> Frac fA sa)
  requires
    on gpu_loc (b |-> sb)
  ensures
    on gpu_loc (b |-> K.s_row_scale sa sb)
{
  K.row_scale t m n a b;
}

let rowscale_f16_rowmajor = inst f16 l1_forward l2_row_major
let rowscale_f16_colmajor = inst f16 l1_forward l2_col_major
let rowscale_f32_rowmajor = inst f32 l1_forward l2_row_major
let rowscale_f32_colmajor = inst f32 l1_forward l2_col_major
let rowscale_f64_rowmajor = inst f64 l1_forward l2_row_major
let rowscale_f64_colmajor = inst f64 l1_forward l2_col_major
