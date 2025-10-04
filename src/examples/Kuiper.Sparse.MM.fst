
module Kuiper.Sparse.MM

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
  (#erows #eshared #ecols : erased nat)
  (#lB : mlayout eshared ecols)
  (#lC : mlayout erows ecols)
  {| clayout lB, clayout lC |}
  (rows shared cols : szp)
  (gA : smatrix et erows eshared)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et #erows #ecols lC)
  #a #b
  requires
    (exists* c. gpu_matrix_pts_to gC #1.0R c) **
    pure (
      erows == SZ.v rows /\
      eshared == SZ.v shared /\
      ecols == SZ.v cols
    )
  preserves
    gpu ** gA |-> a ** gB |-> b
  ensures
    // gC |-> matmul a b
    (exists* c. gpu_matrix_pts_to gC #1.0R c)
{

  let mut i = 0sz;
  unfold smatrix_pts_to gA;

  while ((!i <^ rows))
    invariant live i
  {
    let mut j = 0sz;
    while((!j <^ cols))
      invariant live j
    {
      with v_i.
        assert i |-> v_i;
      with v_off.
        assert gA.row_off |-> v_off;
      with v_ind.
        assert gA.col_ind |-> v_ind;
      
      let row_cols = hide (slice_row (cast_pos v_off) (cast_pos v_ind) v_i);

      let mut dp : et = zero;
      
      let ri = gpu_array_read gA.row_off !i;
      let re = gpu_array_read gA.row_off (!i +^ 1sz);

      let mut k = ri;

      while ((!k <^ re))
        invariant
          exists* v_k.
            k |-> v_k **
            live dp **
            pure (
              ri <= v_k /\
              (v_k < re ==> SZ.v (v_ind @! v_k) == row_cols @! (v_k - ri)) 
            )
          
      {
        let x = gpu_array_read gA.elems !k;
        let c = gpu_array_read gA.col_ind !k;

        let y = gpu_matrix_read gB c !j;

        dp := !dp `add` (x `mul` y);

        k := !k +^ 1sz;
      };
      admit();
      with c. assert gpu_matrix_pts_to gC #1.0R c;
      gpu_matrix_write #et #erows #ecols #lC #_ gC !i !j dp #c;
    }
  };
  fold smatrix_pts_to gA;
}

let _mmsd_u32 rows shared cols =
  smatrix_sdmm #u32 #_ #_ #_ #_
  #(row_major _ _) #(row_major _ _)
  #(clayout_from_crepr shared cols row_major crepr_row_major)
  #(clayout_from_crepr rows cols row_major crepr_row_major)
  rows shared cols