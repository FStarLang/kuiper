module Kuiper.Sparse.LoadStore

#lang-pulse

open Kuiper
open Kuiper.Sparse
open Kuiper.Array.Vectorized
open Kuiper.Array2.Vectorized
open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
open Kuiper.Array2.Strided
module T = Kuiper.Tensor
module SZ = Kuiper.SizeT

let load_array_vec_bounds
  (n m i : nat)
  (nthr ch : pos)
  (tid : natlt nthr)
  : Lemma
      (requires nthr * ch /? n /\ i + n <= m)
      (ensures forall (k : natlt (n / nthr / ch)).
        i + (k * nthr + tid) * ch <= m - ch)
  = introduce forall (k : natlt (n / nthr / ch)).
      i + (k * nthr + tid) * ch <= m - ch
    with ()

inline_for_extraction noextract
fn array_vec_cpy_device
  (#a : Type u#0) {| sized a, has_vec_cpy a |}
  (#dsz : erased nat)
  (d : larray a dsz) (doff : sz)
  (#_ : squash (aligned' 16 d doff))
  (#ssz : erased nat)
  (s : larray a ssz) (soff : sz)
  (#_ : squash (aligned' 16 s soff))
  (#i #j : erased nat)
  (#f : perm)
  (#v : erased (seq a))
  (#_ : squash (i <= soff /\ soff <= j - chunk a))
  (#_ : squash (len v == j - i))
  preserves gpu
  preserves pts_to_slice s #f i j v
  requires live_vec d doff
  requires pure (aligned' 16 s soff)
  ensures  pts_to_vec' d doff v (soff - i)
{
  unfold live_vec d;
  with u_. assert pts_to_vec d doff u_;
  pts_to_slice_ref d doff (doff + chunk a);
  pts_to_slice_ref s i j;
  array_vec_cpy d doff s soff;
  with u. assert pts_to_vec d doff u;
  assert pure (u `Seq.equal` Seq.slice v (soff - i) (soff - i + chunk a));
}

#push-options "--z3rlimit 30"
inline_for_extraction noextract
fn load_array_vec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n : sz)
  (x : larray et n)
  (#_ : squash (aligned 16 x))
  (#m : sz)
  (y : larray et m)
  (#f : perm)
  (#s : erased (lseq et m))
  (i : szle m)
  (#_ : squash (aligned' 16 y i))
  (nthr : sz)
  (tid : szlt nthr)
  (#_ : squash (nthr * chunk et /? n))
  (#_ : squash (i + n <= m))
  preserves gpu ** y |-> Frac f s
  requires thread_live_chunks x nthr tid
  ensures thread_pts_to_chunks x s i nthr tid
{
  unfold thread_live_chunks x nthr tid;

  forevery_rw_size (n / (nthr * (chunk et))) (n /^ nthr /^ chunk et);

  load_array_vec_bounds n m i nthr (chunk et) tid;
  foreach (n /^ nthr /^ chunk et)
  (fun k -> live_vec x ((k * nthr + tid) * chunk et))
  (fun k ->
    pts_to_vec' x ((k * nthr + tid) * chunk et)
      s (i + (k * nthr + tid) * chunk et)
  )
  #(gpu ** y |-> Frac f s)
  fn k {
    // This thread's k-th chunk ends at or before n: k < n / nthr / chunk et and
    // tid < nthr give (k * nthr + tid) * chunk et < n, and nthr * chunk et
    // divides n closes the gap to the end of the chunk. Spelled out because as
    // a single query the nonlinear chain times out.
    assert pure ((k * nthr + tid) * chunk et < n);
    assert pure ((k * nthr + tid) * chunk et + chunk et <= n);
    array_vec_cpy_device
      x ((k *^ nthr +^ tid) *^ chunk et)
      y (i +^ ((k *^ nthr +^ tid) *^ chunk et));
  };

  FStar.Math.Lemmas.division_multiplication_lemma n nthr (chunk et);
  forevery_rw_size (n /^ nthr /^ chunk et) (n / (nthr * (chunk et)));

  fold thread_pts_to_chunks x s i nthr tid;
}
#pop-options

inline_for_extraction noextract
fn load_cell
  (#et : Type0)
  (#m #n : sz)
  (x : larray et m)
  (i : szlt m)
  (y : larray et n)
  (#f : perm)
  (#s : erased (lseq et n))
  (j : szlt n)
  preserves gpu ** y |-> Frac f s
  requires array_live_cell x i
  ensures  Cell (x <: array et) (SZ.v i) |-> Seq.index s j
{
  unfold array_live_cell x;
  slice_write x i (Pulse.Lib.Array.(y.(j)));
  with t. assert pts_to_slice x i (i + 1) t;
  assert pure (Seq.equal t seq![Seq.index s j]);
}

inline_for_extraction noextract
fn load2_array
  (#et1 #et2 : Type0)
  (#dsz : sz)
  (dst1 : larray et1 dsz)
  (dst2 : larray et2 dsz)
  (to : szle dsz)
  (#ssz : sz)
  (src1 : larray et1 ssz)
  (src2 : larray et2 ssz)
  (#f : perm)
  (#s1 : erased (lseq et1 ssz))
  (#s2 : erased (lseq et2 ssz))
  (i : szle ssz)
  (nthr : sz)
  (tid : szlt nthr)
  (#_ : squash (i + to <= ssz))
  preserves gpu ** src1 |-> Frac f s1 ** src2 |-> Frac f s2
  requires thread_slice_live dst1 0 to nthr tid
  requires thread_slice_live dst2 0 to nthr tid
  requires pure (fits (dsz + nthr))
  ensures thread_slice_pts_to dst1 0 to s1 i nthr tid
  ensures thread_slice_pts_to dst2 0 to s2 i nthr tid
{
  unfold thread_slice_live dst1 0 to nthr tid;
  unfold thread_slice_live dst2 0 to nthr tid;

  forevery_zip #(natlt ((to - 0 - tid) `divup` nthr))
    (fun k -> array_live_cell dst1 (0 + k * nthr + tid))
    _;
  forevery_rw_size
    ((to - 0 - tid) `divup` nthr)
    ((to +^ (nthr -^ 1sz) -^ tid) /^ nthr);

  foreach ((to +^ (nthr -^ 1sz) -^ tid) /^ nthr)
    (fun k ->
      array_live_cell dst1 (0 + k * nthr + tid) **
      array_live_cell dst2 (0 + k * nthr + tid)
    )
    (fun k ->
      pts_to_cell dst1 (k * nthr + tid) (s1 @! i + k * nthr + tid) **
      pts_to_cell dst2 (k * nthr + tid) (s2 @! i + k * nthr + tid)
    )
    #(gpu ** src1 |-> Frac f s1 ** src2 |-> Frac f s2)
    fn k {
      rewrite each (0 + k * nthr + tid) as (k * nthr + tid);
      load_cell dst1 (k *^ nthr +^ tid) src1 (i +^ k *^ nthr +^ tid);
      load_cell dst2 (k *^ nthr +^ tid) src2 (i +^ k *^ nthr +^ tid);
    };

  forevery_rw_size
    ((to +^ (nthr -^ 1sz) -^ tid) /^ nthr)
    ((to - 0 - tid) `divup` nthr);
  forevery_unzip _ _;

  forevery_ext #(natlt ((to - 0 - tid) `divup` nthr))
    (fun x ->
      pts_to_cell dst1 (x * nthr + tid) (s1 @! i + x * nthr + tid)
    )
    (fun x ->
      pts_to_cell dst1 (0 + x * nthr + tid) (s1 @! i + x * nthr + tid)
    );
  fold thread_slice_pts_to dst1 0 to s1 i nthr tid;

  forevery_ext #(natlt ((to - 0 - tid) `divup` nthr))
    (fun x ->
      pts_to_cell dst2 (x * nthr + tid) (s2 @! i + x * nthr + tid)
    )
    (fun x ->
      pts_to_cell dst2 (0 + x * nthr + tid) (s2 @! i + x * nthr + tid)
    );
  fold thread_slice_pts_to dst2 0 to s2 i nthr tid;
}

inline_for_extraction noextract
fn array_vec_cpy_local
  (#a : Type u#0) {| sized a, has_vec_cpy a |}
  (#dsz : erased nat)
  (d : larray a dsz) (doff : sz)
  (#_ : squash (aligned' 16 d doff))
  (#ssz : erased nat)
  (s : larray a ssz) (soff : sz { soff + chunk a <= ssz})
  (#_ : squash (aligned' 16 s soff))
  (#f : perm)
  (#v : erased (lseq a ssz))
  preserves gpu
  preserves s |-> Frac f v
  requires live_vec d doff
  ensures  pts_to_vec' d doff v soff
{
  unfold live_vec d;
  with u_. assert pts_to_vec d doff u_;
  pts_to_slice_ref d doff (doff + chunk a);
  array_vec_cpy d doff s soff;
  with u. assert pts_to_vec d doff u;
  assert pure (u `Seq.equal` Seq.slice v soff (soff + chunk a));
}

inline_for_extraction noextract
fn matrix_vec_store
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : layout2 rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : szlt rows)
  (j : sz { j + chunk et <= cols })
  (#n : erased nat)
  (arr : larray et n)
  (#f : perm)
  (#s : erased (lseq et n))
  (k : sz { k + chunk et <= n })
  preserves gpu
  preserves arr |-> Frac f s
  requires  pure (aligned' 16 arr k)
  requires  matrix_live_vec gm i j
  requires  pure (aligned' 16 (core gm) (cell_of_pos l i j))
  requires  pure (aligned_strided_row_major (chunk et) strided)
  ensures   matrix_pts_to_vec_slice gm i j s k
{
  unfold matrix_live_vec gm;
  with v. assert matrix_pts_to_vec gm i j v;
  unfold matrix_pts_to_vec gm i j _;

  strided.pf i j;
  let offset : sz = strided.offset +^ strided.stride *^ i +^ j;

  forevery_map #(natlt (chunk et))
    (fun x -> Cell gm (idx2 (i <: natlt rows) (j + x <: natlt cols)) |-> (Seq.index v x))
    (fun x -> Cell (core gm <: array et) (offset + x <: nat) |-> (Seq.index v x))
    fn x {
      let j' : natlt cols = j + x;
      assert rewrites_to j' (j + x);
      let i' : natlt rows = SZ.v i;
      assert rewrites_to i' (SZ.v i);

      tensor_pts_to_cell_eq gm (idx2 i' j') 1.0R (Seq.index v x);
      strided.pf i' j';

      rewrite Cell gm (idx2 i' j') |-> Seq.index v x
      as Cell (core gm <: array et) (offset + x <: nat) |-> Seq.index v x;
    };

  forevery_rw_size (chunk et) ((offset + chunk et) - offset);
  strided.pf i (j + chunk et - 1);
  cells_to_nonempty_slice (core gm) offset (offset + chunk et);
  fold live_vec (core gm) offset;

  array_vec_cpy_local (core gm) offset arr k;

  slice_to_cells (core gm) offset (offset + chunk et);

  forevery_rw_size ((offset + chunk et) - offset) (chunk et);
  forevery_map #(natlt (chunk et))
    (fun x ->
      pts_to_cell
        (core gm)
        (offset + x)
        (Seq.index (Seq.slice s k (k + chunk et)) x)
    )
    (fun x ->
      tensor_pts_to_cell gm (idx2 i (j + x)) (Seq.index (seq_chunk s k) x)
    )
    fn x {
      let j' : natlt cols = j + x;
      assert rewrites_to j' (j + x);
      let i' : natlt rows = SZ.v i;
      assert rewrites_to i' (SZ.v i);

      tensor_pts_to_cell_eq gm (idx2 i' j') 1.0R (Seq.index (Seq.slice s k (k + chunk et)) x);
      strided.pf i' j';

      // assert pure (Seq.slice s k (k + chunk et) @! x == seq_chunk s k @! x);
      rewrite
        pts_to_cell
          (core gm)
          (offset + x)
          (Seq.index (Seq.slice s k (k + chunk et)) x)
      as tensor_pts_to_cell gm (idx2 i' j') (Seq.index (seq_chunk s k) x);
    };
  fold matrix_pts_to_vec gm i j (seq_chunk s k);
  fold matrix_pts_to_vec_slice gm i j s k;
}
inline_for_extraction noextract
fn matrix_vec_write_in_bounds
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : szp { fits (rows * cols) /\ chunk et /? cols })
  (#l : layout2 rows cols) {| ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : szlt rows)
  (j : sz { chunk et /? j })
  (#f : perm)
  (#n : erased nat)
  (arr : larray et n)
  (#s : erased (lseq et n))
  (k : sz { k + chunk et <= n })
  preserves gpu
  preserves arr |-> Frac f s
  requires  pure (aligned' 16 arr k)
  requires  matrix_live_vec_in_bounds gm i j
  requires  pure (aligned 16 (core gm))
  requires  pure (aligned_strided_row_major (chunk et) strided)
  ensures   matrix_pts_to_vec_slice_in_bounds gm i j s k
{
  if (j <^ cols) {
    unfold_matrix_live_vec_in_bounds gm i j;

    aligned_cell_strided_row_major gm i j;
    // lemma_divides_leq (chunk et) cols j;
    matrix_vec_store gm i j arr k;

    fold_matrix_pts_to_vec_slice_in_bounds gm i j s k;
  }
  else
  {
    unfold_matrix_live_vec_not_in_bounds gm i j;
    fold_matrix_pts_to_vec_slice_not_in_bounds gm i j s k;
  }
}

inline_for_extraction noextract
let offset_chunk_
  (et : Type0) {| sized et, has_vec_cpy et |}
  (j : sz { chunk et /? j })
  (k nthr : sz)
: Pure sz
  (requires fits (offset_chunk et j k nthr))
  (ensures fun r -> v r == offset_chunk et j k nthr)
=
  j +^ (k *^ nthr) *^ chunk et

inline_for_extraction noextract
fn matrix_store_tile_vec_underspec
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : szp { fits (rows * cols) /\ chunk et /? cols })
  (#l : layout2 rows cols) {| ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : szlt rows)
  (j : sz { chunk et /? j })
  (#n : sz { chunk et /? n })
  (arr : larray et n)
  (#f : perm)
  (#s : erased (lseq et n))
  (nthr : sz)
  preserves gpu
  preserves arr |-> Frac f s
  requires  pure (aligned 16 arr)
  requires  thread_live_tile_vec gm i j n nthr
  requires  pure (aligned 16 (core gm))
  requires  pure (aligned_strided_row_major (chunk et) strided)
  requires  pure (fits (j + n * nthr))
  ensures   thread_pts_to_tile_vec_underspec gm i j s nthr
{
  unfold thread_live_tile_vec gm;

  forevery_rw_size (n / chunk et) (n /^ chunk et);

  foreach (n /^ chunk et)
    (fun k -> thread_live_vec gm i j n nthr k)
    (fun k -> thread_pts_to_vec_underspec gm i j s nthr k)
    #(gpu ** arr |-> Frac f s)
    fn k {
      assert pure (k * chunk et <= n);
      assert pure (k * chunk et * nthr <= n * nthr);
      assert pure (j + k * chunk et * nthr <= j + n * nthr);
      assert pure (offset_chunk et j k nthr <= j + n * nthr);
      FStar.SizeT.fits_lte (offset_chunk et j k nthr) (j + n * nthr);
      rewrite each offset_chunk et j k nthr
      as v (offset_chunk_ et j k nthr);
      matrix_vec_write_in_bounds gm
        i (offset_chunk_ et j k nthr)
        arr (k *^ chunk et);
      rewrite each v (offset_chunk_ et j k nthr)
      as offset_chunk et j k nthr;
    };

  forevery_rw_size (n /^ chunk et) (n / chunk et);
  fold thread_pts_to_tile_vec_underspec gm i j s nthr;
}

inline_for_extraction noextract
fn matrix_store_tile_vec
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : szp { fits (rows * cols) /\ chunk et /? cols })
  (#l : layout2 rows cols) {| ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : szlt rows)
  (j : sz { chunk et /? j })
  (em : chest2 et rows cols)
  (n : sz { chunk et /? n })
  (arr : larray et n)
  (#f : perm)
  (#s : erased (lseq et n))
  (nthr : sz)
  preserves gpu
  preserves arr |-> Frac f s
  requires  pure (aligned 16 arr)
  requires  pure (is_ematrix_tile em i j s nthr)
  requires  thread_live_tile_vec gm i j n nthr
  requires  pure (aligned 16 (core gm))
  requires  pure (aligned_strided_row_major (chunk et) strided)
  requires  pure (fits (j + n * nthr))
  ensures   thread_pts_to_tile_vec gm i j em n nthr
{
  matrix_store_tile_vec_underspec gm i j arr nthr;
  fold_thread_pts_to_tile_vec gm i j s em nthr;
}

#push-options "--z3rlimit 10"
inline_for_extraction noextract
fn array_cpy_chunk
  (#et : Type u#0) {| sized et, has_vec_cpy et |}
  (dst_arr : larray et (chunk et))
  (#src_sz : szp { chunk et /? src_sz })
  (#src_l : layout1 src_sz) {| T.ctlayout src_l, src_cl : cont_layout src_l |}
  (src_arr : array1 et src_l)
  (src_off : szlt src_sz { chunk et /? src_off })
  (#f : perm)
  (#ss : erased (lseq et src_sz))
  preserves gpu
  requires  live dst_arr
  requires  pure (aligned 16 dst_arr)
  preserves src_arr |-> Frac f (seq_to_chest1 ss)
  requires  pure (aligned 16 (core src_arr))
  requires  pure (aligned_cont_layout (chunk et) src_cl)
  ensures   dst_arr |-> Seq.slice ss src_off (src_off + chunk et)
{
  lower_cont src_arr #f;

  src_cl.pf src_off;

  aligned_cont_offset src_arr (SZ.v src_off);

  Pulse.Lib.Array.PtsTo.pts_to_len dst_arr;
  array_vec_cpy
    dst_arr 0sz
    (core src_arr) (src_cl.offset +^ src_off);

  raise_cont src_arr #f;

  with s. assert dst_arr |-> s;
  assert pure (Seq.equal s (Seq.slice ss src_off (src_off + chunk et)));

  with ss'. assert src_arr |-> Frac f ss';
  assert pure (ss' `equal` seq_to_chest1 ss);
}
#pop-options