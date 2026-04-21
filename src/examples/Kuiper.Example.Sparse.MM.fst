
module Kuiper.Example.Sparse.MM

#lang-pulse
open Kuiper
open Kuiper.Sparse
module SZ = FStar.SizeT
open Kuiper.Matrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Spec.GEMM
open Kuiper.Matrix.Reprs

inline_for_extraction noextract
fn smatrix_sdmm
  (#et : Type0) {| scalar et |}
  (rows shared cols : szp)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lB, clayout lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared))
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
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

        let y = gpu_matrix_read gB c !j;

        dp := !dp `add` (x `mul` y);

        k := !k +^ 1sz;
      };

      gpu_matrix_write gC !i !j !dp;
      j := !j +^ 1sz;
    };
    i := !i +^ 1sz;
  };

  fold smatrix_pts_to gA a;
}

let _mmsd_u32_rr (rows shared cols : szp { SZ.fits (rows * cols) /\ SZ.fits (shared * cols) }) =
  smatrix_sdmm #u32 #_
  rows shared cols
  #(row_major _ _) #(row_major _ _)

let _mmsd_u32_cc (rows shared cols : szp { SZ.fits (rows * cols) /\ SZ.fits (shared * cols) }) =
  smatrix_sdmm #u32 #_
  rows shared cols
  #(col_major _ _) #(col_major _ _)
