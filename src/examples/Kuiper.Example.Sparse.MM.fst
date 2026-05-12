
module Kuiper.Example.Sparse.MM

#lang-pulse
open Kuiper
open Kuiper.Array2
open Kuiper.Sparse
open Kuiper.Spec.GEMM
open Kuiper.Tensor { ctlayout }
open Kuiper.Tensor.Layout.Alg { l2_row_major, l2_col_major }
module SZ = Kuiper.SizeT
module Array2 = Kuiper.Array2

inline_for_extraction noextract
fn smatrix_sdmm
  (#et : Type0) {| scalar et |}
  (rows shared cols : szp)
  (#lB : layout shared cols)
  (#lC : layout rows cols)
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

  while (!i <^ rows)
    invariant live i ** pure (!i <= rows)
    invariant live gC
    decreases (rows - !i)
  {
    let ri = gpu_array_read gA.row_off !i;
    let re = gpu_array_read gA.row_off (!i +^ 1sz);

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
        let x = gpu_array_read gA.elems !k;
        let c = gpu_array_read gA.col_ind !k;

        let y = Array2.read gB (c, !j);

        dp := !dp `add` (x `mul` y);

        k := !k +^ 1sz;
      };

      Array2.write gC (!i, !j) !dp;
      j := !j +^ 1sz;
    };
    i := !i +^ 1sz;
  };

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
