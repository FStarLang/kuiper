module Kuiper.Example.Sparse.Array.Product

#lang-pulse
open Kuiper
open Kuiper.Sparse
open Kuiper.Sparse.DotProduct
open Kuiper.Sparse.FMA

inline_for_extraction noextract
fn sarray_product_dense
  (#et : eqtype) {| scalar et |}
  (#l : erased nat)
  (a : sarray et l)
  (v : larray et l)
  (#s #t : erased (lseq et l))
  preserves gpu ** a |-> s ** v |-> t
  returns
    dp: et
  ensures
    pure (dp == dprod s t)
{
  unfold sarray_pts_to a s;
  unfold sarray_pts_to' a s;

  with v_elems. assert pts_to_slice a.elems 0 a.nnz v_elems;
  with v_pos.   assert pts_to_slice a.pos 0 a.nnz   v_pos;

  let mut i = 0sz;
  let mut dp : et = zero;

  let pos : erased (lseq nat a.nnz) = cast_pos v_pos;

  while (!i <^ a.nnz)
    invariant
      live dp ** live i **
        pure (
          !i <= a.nnz /\
          !dp = _sparse_dprod v_elems (cast_pos v_pos) t !i
        )
    decreases (a.nnz - !i)
  {
    let p = (a.pos).(!i);
    let x = (a.elems).(!i);
    let y = v.(p);

    dp := fma !dp x y;
    i := !i +^ 1sz;
  };

  sparse_dprod_lemma v_elems pos t;

  fold sarray_pts_to' a s;
  fold sarray_pts_to a s;

  !dp;
}

let sarray_product_dense_u32 #len = sarray_product_dense #u32 #_ #len