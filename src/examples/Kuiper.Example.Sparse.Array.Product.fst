module Kuiper.Example.Sparse.Array.Product

#lang-pulse
open Kuiper
open Kuiper.Sparse
open Kuiper.Sparse.DotProduct
open Kuiper.Sparse.FMA

(* Stored sparse entries approximate reals; structural gaps are real zero. *)
let sarray_approx'
  (#et : Type0) {| scalar et, real_like et |}
  (#l : nat) (a : sarray et l) (s : lseq et l) (rs : lseq real l)
  (elems : lseq et a.nnz) (pos : lseq sz a.nnz) (relems : lseq real a.nnz)
  : slprop
  = a.elems |-> elems ** a.pos |-> pos **
      pure (valid_pos l (cast_pos pos) /\ elems %~ relems /\
            s == seq_unsparse a.nnz l elems (cast_pos pos) /\
            rs == seq_unsparse a.nnz l relems (cast_pos pos))

let sarray_approx
  (#et : Type0) {| scalar et, real_like et |}
  (#l : nat) (a : sarray et l) (s : lseq et l) (rs : lseq real l) : slprop
  = exists* elems pos relems. sarray_approx' a s rs elems pos relems

inline_for_extraction noextract
fn sarray_product_dense
  (#et : eqtype) {| scalar et, real_like et |}
  (#l : erased nat)
  (a : sarray et l)
  (v : larray et l)
  (#s : erased (lseq et l))
  (#rs : erased (lseq real l))
  (#t : erased (lseq et l))
  (#rt : erased (lseq real l))
  preserves gpu ** sarray_approx a s rs ** v |-> t
  requires pure (t %~ rt)
  returns
    dp: et
  ensures
    pure (dp %~ dprod rs rt)
{
  unfold sarray_approx a s rs;

  with v_elems v_pos relems. unfold sarray_approx' a s rs v_elems v_pos relems;

  let mut i = 0sz;
  let mut dp : et = zero;

  let pos : erased (lseq nat a.nnz) = cast_pos v_pos;

  while (!i <^ a.nnz)
    invariant
      live dp ** live i **
        pure (
          !i <= a.nnz /\
          !dp %~ _sparse_dprod relems (cast_pos v_pos) rt !i
        )
    decreases (a.nnz - !i)
  {
    let p = (a.pos).(!i);
    let x = (a.elems).(!i);
    let y = v.(p);

    dp := fma !dp x y;
    i := !i +^ 1sz;
  };

  sparse_dprod_lemma relems pos rt;

  fold sarray_approx' a s rs v_elems v_pos relems;
  fold sarray_approx a s rs;

  !dp;
}

let sarray_product_dense_u32 #len = sarray_product_dense #u32 #_ #_ #len
