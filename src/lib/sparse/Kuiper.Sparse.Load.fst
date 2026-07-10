module Kuiper.Sparse.Load

#lang-pulse

open Kuiper
open Kuiper.Sparse
open Kuiper.Array.Vectorized
open Kuiper.Array2.Vectorized
open Kuiper.EMatrix
open Kuiper.Tensor.Layout.Alg { l2_row_major }
open Kuiper.Array2.Strided { strided_row_major, cell_of_pos }
module T = Kuiper.Tensor
module A = Kuiper.Array1
module M = Kuiper.Array2
module SZ = Kuiper.SizeT


inline_for_extraction noextract
fn gpu_load_cell
  (#et : Type0)
  (#m #n : sz)
  (x : gpu_array et m)
  (i : szlt m)
  (y : gpu_array et n)
  (#f : perm)
  (#s : erased (lseq et n))
  (j : szlt n)
  preserves gpu ** y |-> Frac f s
  requires array_live_cell x i
  ensures  gpu_pts_to_cell x i (s @! j)
{
  unfold array_live_cell x;
  gpu_array_write x i (gpu_array_read y j);
  with t. assert gpu_pts_to_slice x i (i + 1) t;
  assert pure (Seq.equal t seq![s @! j]);
}

inline_for_extraction noextract
fn gpu_array_vec_cpy_device
  (#a : Type u#0) {| sized a, has_vec_cpy a |}
  (#dsz : erased nat)
  (d : gpu_array a dsz) (doff : sz)
  (#_ : squash (aligned' 16 d doff))
  (#ssz : erased nat)
  (s : gpu_array a ssz) (soff : sz)
  (#_ : squash (aligned' 16 s soff))
  (#i #j : erased nat)
  (#f : perm)
  (#v : erased (seq a))
  (#_ : squash (i <= soff /\ soff <= j - chunk a))
  (#_ : squash (len v == j - i))
  preserves gpu
  preserves gpu_pts_to_slice s #f i j v
  requires gpu_live_vec d doff
  requires pure (aligned' 16 s soff)
  ensures  gpu_pts_to_vec' d doff v (soff - i)
{
  unfold gpu_live_vec d;
  with u_. assert gpu_pts_to_vec d doff u_;
  gpu_pts_to_slice_ref d doff (doff + chunk a);
  gpu_pts_to_slice_ref s i j;
  gpu_array_vec_cpy_dd d doff s soff;
  with u. assert gpu_pts_to_vec d doff u;
  assert pure (u `Seq.equal` Seq.slice v (soff - i) (soff - i + chunk a));
}

inline_for_extraction noextract
fn gpu_array_vec_cpy_local
  (#a : Type u#0) {| sized a, has_vec_cpy a |}
  (#dsz : erased nat)
  (d : gpu_array a dsz) (doff : sz)
  (#_ : squash (aligned' 16 d doff))
  (#ssz : erased nat)
  (s : larray a ssz) (soff : sz { soff + chunk a <= ssz})
  (#f : perm)
  (#v : erased (lseq a ssz))
  preserves gpu
  preserves s |-> Frac f v
  requires gpu_live_vec d doff
  ensures  gpu_pts_to_vec' d doff v soff
{
  unfold gpu_live_vec d;
  with u_. assert gpu_pts_to_vec d doff u_;
  gpu_pts_to_slice_ref d doff (doff + chunk a);
  gpu_array_vec_cpy_hd d doff s soff;
  with u. assert gpu_pts_to_vec d doff u;
  assert pure (u `Seq.equal` Seq.slice v soff (soff + chunk a));
}

inline_for_extraction noextract
fn matrix_vec_store
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : M.layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : M.array2 et l)
  (i : szlt rows)
  (j : sz { j + chunk et <= cols })
  (#n : erased nat)
  (arr : larray et n)
  (#f : perm)
  (#s : erased (lseq et n))
  (k : sz { k + chunk et <= n })
  preserves gpu
  preserves arr |-> Frac f s
  requires  matrix_live_vec gm i j
  requires  pure (aligned' 16 (M.core gm) (cell_of_pos l i j))
  ensures   matrix_pts_to_vec_slice gm i j s k
{
  unfold matrix_live_vec gm;
  with v. assert matrix_pts_to_vec gm i j v;
  unfold matrix_pts_to_vec gm;

  strided.pf i j;
  let offset : sz = strided.offset +^ strided.stride *^ i +^ j;

  forevery_map #(natlt (chunk et))
    (fun x -> M.pts_to_cell gm ((SZ.v i <: natlt rows), (j + x)) (v @! x))
    (fun x -> gpu_pts_to_cell (M.core gm) (offset + x) (v @! x))
    fn x {
      let j' : natlt cols = j + x;
      assert rewrites_to j' (j + x);
      let i' : natlt rows = SZ.v i;
      assert rewrites_to i' (SZ.v i);

      M.pts_to_cell_eq gm (i', j') 1.0R (v @! x);
      strided.pf i' j';

      rewrite M.pts_to_cell gm (i', j') (v @! x)
      as gpu_pts_to_cell (M.core gm) (offset + x) (v @! x);
    };

  forevery_rw_size (chunk et) ((offset + chunk et) - offset);
  strided.pf i (j + chunk et - 1);
  gpu_array_unslice_1' (M.core gm) offset (offset + chunk et);
  fold gpu_live_vec (M.core gm) offset;

  gpu_array_vec_cpy_local (M.core gm) offset arr k;

  gpu_array_slice_1' (M.core gm) offset (offset + chunk et);
  forevery_rw_size ((offset + chunk et) - offset) (chunk et);
  forevery_map #(natlt (chunk et))
    (fun x -> gpu_pts_to_cell (M.core gm) (offset + x) (Seq.slice s k (k + chunk et) @! x))
    (fun x -> M.pts_to_cell gm ((SZ.v i <: natlt rows), (j + x)) (seq_chunk s k @! x))
    fn x {
      let j' : natlt cols = j + x;
      assert rewrites_to j' (j + x);
      let i' : natlt rows = SZ.v i;
      assert rewrites_to i' (SZ.v i);

      M.pts_to_cell_eq gm (i', j') 1.0R (Seq.slice s k (k + chunk et) @! x);
      strided.pf i' j';

      assert pure (Seq.slice s k (k + chunk et) @! x == seq_chunk s k @! x);
      rewrite gpu_pts_to_cell (M.core gm) (offset + x) (Seq.slice s k (k + chunk et) @! x)
      as M.pts_to_cell gm (i', j') (seq_chunk s k @! x);
    };
  fold matrix_pts_to_vec gm i j (seq_chunk s k);
  fold matrix_pts_to_vec_slice gm i j s k;
}

open Kuiper.Seq.Common { seq_blit }
let seq_blit'
  (#a:Type)
  (#n1 : nat)
  (s1 : lseq a n1) (off1 : natlt n1)
  (#n2 : nat)
  (s2 : lseq a n2) (off2 : nat)
  (cnt : nat { cnt /? n1 /\ cnt /? off1 /\ cnt /? n2 /\ cnt /? off2 } )
  : lseq a n1
=
  lemma_divides_leq cnt n1 off1;
  lemma_divides_leq cnt n2 off2;
  if off2 < n2
    then seq_blit s1 off1 s2 off2 cnt
    else s1


ghost
fn lower_cont
  (#et : Type u#0)
  (#sz : pos)
  (#l : A.layout sz) {| cl : cont_layout l |}
  (a : A.array1 et l)
  (#f : perm)
  (#s : lseq et sz)
  requires a |-> Frac f s
  ensures  gpu_pts_to_slice (A.core a) #f cl.offset (cl. offset + sz) s
{
  A.explode a;
  forevery_map #(natlt sz)
    (fun i -> Cell a i |-> Frac f (s @! i))
    (fun i -> Cell (A.core a) (cl.offset + i <: nat) |-> Frac f (s @! i))
    fn i {
      cl.pf i;
      A.pts_to_cell_eq a i f (s @! i);
      rewrite  Cell a i |-> Frac f (s @! i)
      as Cell (A.core a) (cl.offset + i <: nat) |-> Frac f (s @! i);
    };
  cl.pf (sz - 1);
  forevery_rw_size sz ((cl.offset + sz) - cl.offset);
  gpu_array_unslice_1' (A.core a) cl.offset (cl.offset + sz);
}

ghost
fn raise_cont
  (#et : Type u#0)
  (#sz : pos)
  (#l : A.layout sz) {| cl : cont_layout l |}
  (a : A.array1 et l)
  (#f : perm)
  (#s : lseq et sz)
  requires pure (fits (A.layout_size l))
  requires gpu_pts_to_slice (A.core a) #f cl.offset (cl. offset + sz) s
  ensures  a |-> Frac f s
{
  cl.pf (sz - 1);
  gpu_array_slice_1' (A.core a) cl.offset (cl.offset + sz);
  forevery_rw_size ((cl.offset + sz) - cl.offset) sz;
  forevery_map #(natlt sz)
    (fun i -> Cell (A.core a) (cl.offset + i <: nat) |-> Frac f (s @! i))
    (fun i -> Cell a i |-> Frac f (s @! i))
    fn i {
      cl.pf i;
      A.pts_to_cell_eq a i f (s @! i);
      rewrite Cell (A.core a) (cl.offset + i <: nat) |-> Frac f (s @! i)
      as  Cell a i |-> Frac f (s @! i);
    };
  A.implode a;
}

let aligned_cont_offset
  (#et : Type u#0) {| sized et, has_vec_cpy et |}
  (#sz : erased nat)
  (#l : A.layout sz) {| cl : cont_layout l |}
  (a : A.array1 et l)
  (off : erased nat { chunk et /? off })
: Lemma
  (requires
    aligned 16 (A.core a) /\
    aligned_cont_layout (chunk et) cl
  )
  (ensures aligned' 16 (A.core a) (cl.offset + off))
= ()

#push-options "--split_queries always --z3rlimit 10"
noextract
fn array_vec_cpy
  (#et : Type u#0) {| sized et, has_vec_cpy et |}
  (#dst_sz : erased nat { 0 < dst_sz /\ chunk et /? dst_sz })
  (#dst_l : A.layout dst_sz) {| T.ctlayout dst_l, dst_cl : cont_layout dst_l |}
  (dst_arr : A.array1 et dst_l)
  (dst_off : szlt dst_sz { chunk et /? dst_off })
  (#src_sz : szp { chunk et /? src_sz })
  (#src_l : A.layout src_sz) {| T.ctlayout src_l, src_cl : cont_layout src_l |}
  (src_arr : A.array1 et src_l)
  (src_off : sz { chunk et /? src_off })
  (#f : perm)
  (#ds : erased (lseq et dst_sz))
  (#ss : erased (lseq et src_sz))
  preserves gpu
  requires  dst_arr |-> ds
  requires  pure (aligned 16 (A.core dst_arr))
  requires  pure (aligned_cont_layout (chunk et) dst_cl)
  preserves src_arr |-> Frac f ss
  requires  pure (aligned 16 (A.core src_arr))
  requires  pure (aligned_cont_layout (chunk et) src_cl)
  ensures   dst_arr |-> seq_blit' ds dst_off ss src_off (chunk et)
{
  if (src_off <^ src_sz)
  {
    lower_cont dst_arr;
    lower_cont src_arr #f;
    
    dst_cl.pf dst_off;
    src_cl.pf src_off;
    
    aligned_cont_offset dst_arr (SZ.v dst_off);
    aligned_cont_offset src_arr (SZ.v src_off);

    gpu_array_vec_cpy_dd
      (A.core dst_arr) (dst_cl.offset +^ dst_off)
      (A.core src_arr) (src_cl.offset +^ src_off);

    raise_cont dst_arr;
    raise_cont src_arr #f;
  }
  else
  {
    ();
  }
}
#pop-options