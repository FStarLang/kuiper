module Kuiper.Sparse.Load

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
  array_unslice_1' (core gm) offset (offset + chunk et);
  fold live_vec (core gm) offset;

  array_vec_cpy_local (core gm) offset arr k;

  array_slice_1' (core gm) offset (offset + chunk et);
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

// TODO  mover estas definiciones
let chest_blit
  (#a:Type)
  (#n1 : nat)
  (s1 : chest1 a n1) (off1 : nat)
  (#n2 : nat)
  (s2 : chest1 a n2) (off2 : nat)
  (cnt : nat{off1 + cnt <= n1 /\ off2 + cnt <= n2})
  : chest1 a n1
=
  mk1 fun i ->
    if i < off1 || off1 + cnt <= i
      then acc1 s1 i
      else acc1 s2 (off2 + i - off1)

let chest1_blit'
  (#a:Type)
  (#n1 : nat)
  (s1 : chest1 a n1) (off1 : natlt n1)
  (#n2 : nat)
  (s2 : chest1 a n2) (off2 : nat)
  (cnt : nat { cnt /? n1 /\ cnt /? off1 /\ cnt /? n2 /\ cnt /? off2 } )
  : chest1 a n1
=
  lemma_divides_leq cnt n1 off1;
  lemma_divides_leq cnt n2 off2;
  if off2 < n2
    then chest_blit s1 off1 s2 off2 cnt
    else s1

open Kuiper.Shape { abs_bring_forward_bij }
open Kuiper.Bijection

let prod_unit_bij (a : Type) : (a & unit =~ a) =
{
  ff = (fun (x, ()) -> x);
  gg = (fun x -> (x, ()));

  ff_gg = ez;
  gg_ff = ez;
}

ghost
fn forevery_abs1_iso
  (#n : nat)
  (p : abs (n @| INil) -> slprop)
  requires forall+ (i : abs (n @| INil)). p i
  ensures  forall+ (i : natlt n). p (idx1 i)
{
  forevery_iso (abs_bring_forward_bij 0 (n @| INil)) _;
  rewrite each ((n @| INil) @! 0) as n;
  rewrite each abs (modulo_i 0 (n @| INil)) as unit;
  forevery_iso (prod_unit_bij (natlt n)) _;
  forevery_ext _ (fun i -> p (idx1 i));
}

ghost
fn forevery_abs1_iso_back
  (#n : nat)
  (p : abs (n @| INil) -> slprop)
  requires forall+ (i : natlt n). p (idx1 i)
  ensures  forall+ (i : abs (n @| INil)). p i
{
  forevery_iso (bij_sym (prod_unit_bij (natlt n))) _;
  rewrite each n as ((n @| INil) @! 0);
  rewrite each unit as abs (modulo_i 0 (n @| INil));
  forevery_iso (bij_sym (abs_bring_forward_bij 0 (n @| INil))) _;
  forevery_ext _ p;
}

ghost
fn lower_cont
  (#et : Type u#0)
  (#sz : pos)
  (#l : layout1 sz) {| cl : cont_layout l |}
  (a : array1 et l)
  (#f : perm)
  (#s : chest1 et sz)
  requires a |-> Frac f s
  ensures  pts_to_slice (core a) #f cl.offset (cl. offset + sz) (chest1_to_seq s)
{
  tensor_explode a;
  let s' = chest1_to_seq s;

  forevery_abs1_iso _;

  forevery_map #(natlt sz)
    (fun i -> Cell a (idx1 i) |-> Frac f (acc s (idx1 i)))
    (fun i -> Cell (core a <: array et) (cl.offset + i <: nat) |-> Frac f (Seq.index s' i))
    fn i {
      cl.pf i;
      tensor_pts_to_cell_eq a (idx1 i) f (acc s (idx1 i));
      rewrite  Cell a (idx1 i) |-> Frac f (acc s (idx1 i))
      as pts_to_cell (core a) #f (cl.offset + i) (Seq.index s' i);
    };
  cl.pf (sz - 1);
  forevery_rw_size sz ((cl.offset + sz) - cl.offset);
  array_unslice_1' (core a) cl.offset (cl.offset + sz);
}

ghost
fn raise_cont
  (#et : Type u#0)
  (#sz : pos)
  (#l : layout1 sz) {| cl : cont_layout l |}
  (a : array1 et l)
  (#f : perm)
  (#s : lseq et sz)
  requires pure (fits (tlayout_ulen l))
  requires pts_to_slice (core a) #f cl.offset (cl. offset + sz) s
  ensures  a |-> Frac f (seq_to_chest1 s)
{
  let s' = seq_to_chest1 s;
  cl.pf (sz - 1);
  array_slice_1' (core a) cl.offset (cl.offset + sz);
  forevery_rw_size ((cl.offset + sz) - cl.offset) sz;
  forevery_map #(natlt sz)
    (fun i ->
      Cell (core a <: array et) (cl.offset + i <: nat) |-> Frac f (Seq.index s i))
    (fun i -> Cell a (idx1 i) |-> Frac f (acc s' (idx1 i)))
    fn i {
      cl.pf i;
      tensor_pts_to_cell_eq a (idx1 i) f (Seq.index s i);
      rewrite Cell (core a <: array et) (cl.offset + i <: nat) |-> Frac f (Seq.index s i)
      as  Cell a (idx1 i) |-> Frac f (acc s' (idx1 i));
    };
  forevery_abs1_iso_back #sz
    (fun i -> tensor_pts_to_cell a #f i (acc s' i));
  tensor_implode a;
}

let aligned_cont_offset
  (#et : Type u#0) {| sized et, has_vec_cpy et |}
  (#sz : erased nat)
  (#l : layout1 sz) {| cl : cont_layout l |}
  (a : array1 et l)
  (off : erased nat { chunk et /? off })
: Lemma
  (requires
    aligned 16 (core a) /\
    aligned_cont_layout (chunk et) cl
  )
  (ensures aligned' 16 (core a) (cl.offset + off))
= ()

#push-options "--split_queries always --z3rlimit 15"
noextract
fn array_vec_cpy'
  (#et : Type u#0) {| sized et, has_vec_cpy et |}
  (#dst_sz : erased nat { 0 < dst_sz /\ chunk et /? dst_sz })
  (#dst_l : layout1 dst_sz) {| T.ctlayout dst_l, dst_cl : cont_layout dst_l |}
  (dst_arr : array1 et dst_l)
  (dst_off : szlt dst_sz { chunk et /? dst_off })
  (#src_sz : szp { chunk et /? src_sz })
  (#src_l : layout1 src_sz) {| T.ctlayout src_l, src_cl : cont_layout src_l |}
  (src_arr : array1 et src_l)
  (src_off : sz { chunk et /? src_off })
  (#f : perm)
  (#ds : chest1 et dst_sz)
  (#ss : chest1 et src_sz)
  preserves gpu
  requires  dst_arr |-> ds
  requires  pure (aligned 16 (core dst_arr))
  requires  pure (aligned_cont_layout (chunk et) dst_cl)
  preserves src_arr |-> Frac f ss
  requires  pure (aligned 16 (core src_arr))
  requires  pure (aligned_cont_layout (chunk et) src_cl)
  ensures   dst_arr |-> chest1_blit' ds dst_off ss src_off (chunk et)
{
  if (src_off <^ src_sz)
  {
    lower_cont dst_arr;
    lower_cont src_arr #f;

    dst_cl.pf dst_off;
    src_cl.pf src_off;

    aligned_cont_offset dst_arr (SZ.v dst_off);
    aligned_cont_offset src_arr (SZ.v src_off);

    array_vec_cpy
      (core dst_arr) (dst_cl.offset +^ dst_off)
      (core src_arr) (src_cl.offset +^ src_off);

    raise_cont dst_arr;
    raise_cont src_arr #f;

    with ds'. assert dst_arr |-> ds';
    assert pure (
      ds' `equal` (chest1_blit' ds dst_off ss src_off (chunk et))
    );

    with ss'. assert src_arr |-> Frac f ss';
    assert pure (ss' `equal` ss);
  }
  else
  {
    assert pure (
      ds `equal` (chest1_blit' ds dst_off ss src_off (chunk et))
    );
  }
}
#pop-options

#push-options "--z3rlimit 10"
noextract
fn array_vec_cpy_dh
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