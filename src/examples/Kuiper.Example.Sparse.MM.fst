
module Kuiper.Example.Sparse.MM

#lang-pulse
open Kuiper
open Kuiper.Sparse
open Kuiper.Spec.GEMM
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major, l2_col_major }
module SZ = Kuiper.SizeT

inline_for_extraction noextract
fn smatrix_sdmm
  (#et : Type0) {| scalar et |}
  (rows shared cols : szp)
  (#lB : layout2 shared cols)
  (#lC : layout2 rows cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared))
  (gB : array2 et lB)
  (gC : array2 et lC)
  #a #b
  requires
    live gC
  preserves
    gpu ** gA |-> a ** gB |-> b
  ensures
    //gC |-> matmul a b
    live gC
{

  let mut i = 0sz;
  unfold smatrix_pts_to gA;

  // FIXME
  array_to_slice gA.elems;
  array_to_slice gA.row_off;
  array_to_slice gA.col_ind;

  while (!i <^ rows)
    invariant live i ** pure (!i <= rows)
    invariant live gC
    decreases (rows - !i)
  {
    let ri = slice_read gA.row_off !i;
    let re = slice_read gA.row_off (!i +^ 1sz);

    let mut j = 0sz;
    while (!j <^ cols)
      invariant live j ** pure (!j <= cols)
      invariant live gC
      decreases (cols - !j)
    {
      let mut dp : et = zero;

      let mut k = ri;

      while (!k <^ re)
        invariant
            live dp ** live k
        decreases (re - !k)
      {
        let x = slice_read gA.elems !k;
        let c = slice_read gA.col_ind !k;

        let jv = !j;
        let y = tensor_read gB ((c <: szlt _), ((jv <: szlt _), ()));

        dp := !dp `add` (x `mul` y);

        k := !k +^ 1sz;
      };

      let iv = !i;
      let jv = !j;
      let dpv = !dp;
      tensor_write gC ((iv <: szlt _), ((jv <: szlt _), ())) dpv;
      j := !j +^ 1sz;
    };
    i := !i +^ 1sz;
  };

  // FIXME
  slice_to_array gA.elems;
  slice_to_array gA.row_off;
  slice_to_array gA.col_ind;

  fold smatrix_pts_to gA a;
}

let _mmsd_u32_rr (rows shared cols : szp { SZ.fits (rows * cols) /\ SZ.fits (shared * cols) }) =
  smatrix_sdmm #u32 #_
  rows shared cols
  #(l2_row_major _ _) #(l2_row_major _ _)

let _mmsd_u32_cc (rows shared cols : szp { SZ.fits (rows * cols) /\ SZ.fits (shared * cols) }) =
  smatrix_sdmm #u32 #_
  rows shared cols
  #(l2_col_major _ _) #(l2_col_major _ _)
