module Kuiper.Array2.Vectorized

(* Vectorized read for Array2 (Tensor-backed) layouts. *)

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.Chest

open Kuiper.Tensor { array2, layout2, idx2 }
open Kuiper.Array2.Strided
open Kuiper.TensorRO { vtlayout_of_tlayout }
module RO = Kuiper.TensorRO
module T = Kuiper.Tensor
module SZ = Kuiper.SizeT

let strided_row_major_contiguous
  (#rows #cols : erased nat)
  (l : layout2 rows cols) {| d : strided_row_major (vtlayout_of_tlayout l) |}
  (i : natlt rows)
  (j1 j2 : natlt cols)
  : Lemma (cell_of_pos l i j2 - cell_of_pos l i j1 == j2 - j1)
  = d.pf i j1; d.pf i j2

let all_but_window l j k : natlt l -> prop =
  fun i -> i < j \/ i >= j + k

let get_slice_inv
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols) {| strided : strided_row_major (vtlayout_of_tlayout l) |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : natlt (cols - chunk et + 1))
  (f : perm)
  (em : chest2 et rows cols)
  (k : nat {k <= chunk et})
  : slprop
  =
  pts_to_slice (T.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + k)
      (Seq.init_ghost k (fun x -> acc2 em i (j + x))) **
  (forall+ (x : natlt cols {all_but_window cols j k x}).
    pts_to_cell (T.core gm) #f (cell_of_pos l i x) (acc2 em i x))

ghost
fn __get_slice_step
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols) {| T.ctlayout l, strided : strided_row_major (vtlayout_of_tlayout l) |}
  (gm : array2 et l)
  (i : nat {i < rows})
  (j : nat {j < cols - chunk et + 1})
  (k : nat {k < chunk et})
  (#f : perm)
  (#em : chest2 et rows cols)
  requires get_slice_inv gm i j f em k
  ensures  get_slice_inv gm i j f em (k + 1)
{
  unfold get_slice_inv gm i j f em k;
  forevery_remove' #(natlt cols)
    (fun x -> all_but_window cols j k x)
    (fun x -> pts_to_cell (T.core gm) #f (cell_of_pos l i x) (acc2 em i x))
    (j + k);
  forevery_refine_ext
    (fun (x : natlt cols) -> all_but_window cols j (k + 1) x)
    (fun (x : natlt cols) -> pts_to_cell (T.core gm) #f (cell_of_pos l i x) (acc2 em i x));

  assert pts_to_cell (T.core gm) #f (cell_of_pos l i (j + k)) (acc2 em i (j + k));

  strided_row_major_contiguous l i j (j + k);
  assert pure (j + k < cols);
  assert pure (cell_of_pos l i (j + k) == cell_of_pos l i j + k);

  rewrite
    pts_to_cell (T.core gm) #f (cell_of_pos l i (j + k)) (acc2 em i (j + k))
  as
    pts_to_cell (T.core gm) #f (cell_of_pos l i j + k) (acc2 em i (j + k));

  slice_concat (T.core gm) #f _ (cell_of_pos l i j + k) _;

  assert pure (Seq.equal
      (Seq.init_ghost k (fun x -> acc2 em i (j + x)) `Seq.append` seq![acc2 em i (j + k)])
      (Seq.init_ghost (k + 1) (fun x -> acc2 em i (j + x))));

  fold get_slice_inv gm i j f em (k + 1);

  ();
}

ghost
fn rec __get_slice
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : layout2 rows cols) {| T.ctlayout l, strided : strided_row_major (vtlayout_of_tlayout l) |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : natlt (cols - chunk et + 1))
  (k : nat {k <= chunk et})
  (#f : perm)
  (#em : chest2 et rows cols)
  requires get_slice_inv gm i j f em k
  ensures  get_slice_inv gm i j f em (chunk et)
  decreases (chunk et - k)
{
  let eq = k = chunk et;
  if (eq) {
    rewrite each k as (chunk et);
    ();
  } else {
    __get_slice_step gm i j k;
    __get_slice gm i j (k + 1);
  }
}

ghost
fn get_slice
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : layout2 rows cols) {| T.ctlayout l, strided : strided_row_major (vtlayout_of_tlayout l) |}
  (gm : array2 et l)
  (i : nat{i < rows})
  (j : nat{j < cols - chunk et + 1})
  (#f : perm)
  (#em : chest2 et rows cols)
  requires  gm |-> Frac f em
  ensures
    pts_to_slice (T.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + chunk et)
      (Seq.init_ghost (chunk et) (fun x -> acc2 em i (j + x))) **
    (forall+ (x : natlt cols{all_but_window cols j (chunk et) x}).
      pts_to_cell (T.core gm) #f (cell_of_pos l i x) (acc2 em i x)) **
    (forall+ (r : natlt rows { ~ (eq2 #(natlt rows) r i) } ) (c : natlt cols).
      pts_to_cell (T.core gm) #f (cell_of_pos l r c) (acc2 em r c))
{
  T.tensor_ilower2 gm;
  // Convert pts_to_cell to pts_to_cell
  forevery_map_2
    (fun (r : natlt rows) (c : natlt cols) ->
      T.tensor_pts_to_cell gm #f (idx2 r c) (acc2 em r c))
    (fun (r : natlt rows) (c : natlt cols) ->
      pts_to_cell (T.core gm) #f (cell_of_pos l r c) (acc2 em r c))
    fn r c {
      T.tensor_pts_to_cell_eq gm (idx2 r c) f (acc2 em r c);
      rewrite
        T.tensor_pts_to_cell gm #f (idx2 r c) (acc2 em r c)
      as
        pts_to_cell (T.core gm) #f (cell_of_pos l r c) (acc2 em r c);
    };

  // Extract row i
  forevery_remove #(natlt rows) _ i;

  // Extract cell j and create empty slice
  forevery_remove #(natlt cols) _ j;
  gpu_slice_split' (T.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + 0) (cell_of_pos l i j + 1);
  assert pure (seq![] `Seq.equal` Seq.init_ghost 0 (fun x -> acc2 em i (j + x)));
  assert pts_to_slice (T.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + 0)
    (Seq.init_ghost 0 (fun x -> acc2 em i (j + x)));
  assert pure (Seq.equal (Kuiper.Seq.Common.seq_drop (cell_of_pos l i j + 0 - cell_of_pos l i j)
          seq![acc2 em i j]) seq![acc2 em i j]);
  assert pts_to_slice (T.core gm) #f (cell_of_pos l i j + 0) (cell_of_pos l i j + 1)
    seq![acc2 em i j];
  rewrite pts_to_slice (T.core gm) #f (cell_of_pos l i j + 0) (cell_of_pos l i j + 1)
    seq![acc2 em i j]
    as pts_to_slice (T.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + 1)
    seq![acc2 em i j];
  forevery_insert #(natlt cols) #(fun x -> ~(eq2 #(natlt cols) x j))
    (fun x ->
      pts_to_slice (T.core gm) #f
      (cell_of_pos l i x)
      (cell_of_pos l i x + 1)
      seq![acc2 em i x]) j;
  forevery_unrefine #(natlt cols) _;

  forevery_refine_ext
    (fun (x : natlt cols) -> all_but_window cols j 0 x)
    _;
  fold get_slice_inv gm i j f em 0;
  __get_slice gm i j 0;
  unfold get_slice_inv gm i j f em (chunk et);

  ();
}

#push-options "--z3rlimit 20"
ghost
fn __unget_slice_step
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols) {| T.ctlayout l, strided : strided_row_major (vtlayout_of_tlayout l) |}
  (gm : array2 et l)
  (i : nat {i < rows})
  (j : nat {j < cols - chunk et + 1})
  (k : nat {k < chunk et})
  (#f : perm)
  (#em : chest2 et rows cols)
  requires get_slice_inv gm i j f em (k + 1)
  ensures  get_slice_inv gm i j f em k
{
  unfold get_slice_inv gm i j f em (k + 1);
  assert pure (Seq.equal
      (Seq.init_ghost k (fun x -> acc2 em i (j + x)) `Seq.append` seq![acc2 em i (j + k)])
      (Seq.init_ghost (k + 1) (fun x -> acc2 em i (j + x))));
  slice_split (T.core gm) #f #(Seq.init_ghost k (fun x -> acc2 em i (j + x))) #(seq![acc2 em i (j + k)]) _ (cell_of_pos l i j + k) _;
  strided_row_major_contiguous l i j (j + k);
  assert pure (j + k < cols);
  assert pure (cell_of_pos l i (j + k) == cell_of_pos l i j + k);
  rewrite
    pts_to_cell (T.core gm) #f (cell_of_pos l i j + k) (acc2 em i (j + k))
  as
    pts_to_cell (T.core gm) #f (cell_of_pos l i (j + k)) (acc2 em i (j + k));
  assert pts_to_cell (T.core gm) #f (cell_of_pos l i (j + k)) (acc2 em i (j + k));
  forevery_insert #(natlt cols)
    (fun x -> pts_to_cell (T.core gm) #f (cell_of_pos l i x) (acc2 em i x))
    (j + k);

  forevery_refine_ext
    (fun (x : natlt cols) -> all_but_window cols j k x)
    (fun (x : natlt cols) -> pts_to_cell (T.core gm) #f (cell_of_pos l i x) (acc2 em i x));

  fold get_slice_inv gm i j f em k;

  ();
}
#pop-options

ghost
fn rec __unget_slice
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : layout2 rows cols) {| T.ctlayout l, strided : strided_row_major (vtlayout_of_tlayout l) |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : natlt (cols - chunk et + 1))
  (k : nat {k <= chunk et})
  (#f : perm)
  (#em : chest2 et rows cols)
  requires get_slice_inv gm i j f em (chunk et)
  ensures  get_slice_inv gm i j f em k
  decreases (chunk et - k)
{
  let eq = k = chunk et;
  if (eq) {
    rewrite each (chunk et <: nat) as k;
    ();
  } else {
    __unget_slice gm i j (k + 1);
    __unget_slice_step gm i j k;
  }
}

#push-options "--z3rlimit 160"
ghost
fn unget_slice
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : layout2 rows cols) {| T.ctlayout l, strided : strided_row_major (vtlayout_of_tlayout l) |}
  (gm : array2 et l)
  (i : nat{i < rows})
  (j : nat{j < cols - chunk et + 1})
  (#f : perm)
  (#em : chest2 et rows cols)
  requires
    pts_to_slice (T.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + chunk et)
      (Seq.init_ghost (chunk et) (fun x -> acc2 em i (j + x))) **
    (forall+ (x : natlt cols{all_but_window cols j (chunk et) x}).
      pts_to_cell (T.core gm) #f (cell_of_pos l i x) (acc2 em i x)) **
    (forall+ (r : natlt rows { ~ (eq2 #(natlt rows) r i) } ) (c : natlt cols).
      pts_to_cell (T.core gm) #f (cell_of_pos l r c) (acc2 em r c))
  ensures  gm |-> Frac f em
{
  fold get_slice_inv gm i j f em (chunk et);
  __unget_slice gm i j 0;
  unfold get_slice_inv gm i j f em 0;
  forevery_unrefine #(natlt cols) _;
  with s0 . assert pts_to_slice (T.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j) s0;

  // Merge the empty slice back into j
  forevery_remove #(natlt cols) _ j;
  slice_concat (T.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j) _;
  with s1 . assert pts_to_slice (T.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + 1) s1;
  assert pure (Seq.equal s1 seq![acc2 em i j]);
  assert pts_to_slice (T.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + 1) seq![acc2 em i j];
  forevery_insert #(natlt cols) #(fun x -> ~(eq2 #(natlt cols) x j)) (fun x ->
        pts_to_slice (T.core gm) #f
        (cell_of_pos l i x)
        (cell_of_pos l i x + 1)
        seq![acc2 em i x] ) j;
  forevery_unrefine #(natlt cols) _;

  let phi = (fun (r: natlt rows) (c: natlt (reveal #nat cols)) ->
            pts_to_slice #et
              (T.core gm)
              #f
              (cell_of_pos #(reveal #nat rows) #(reveal #nat cols) l r c)
              (cell_of_pos #(reveal #nat rows) #(reveal #nat cols) l r c + 1)
              (cons #et
                  (acc2 #et #(reveal #nat rows) #(reveal #nat cols) em r c)
                  (empty #et)));
  let p = (fun (r: natlt rows) ->
          forall+ (c: natlt (reveal #nat cols)). phi r c);
  forevery_ext
    #(natlt cols)
    _
    (phi i);
  rewrite
    forall+ (x: natlt (reveal #nat cols)).
      phi i x
    as p i;
  forevery_ext_2
    #(r:
      natlt (reveal #nat rows) {~(eq2 #(natlt (reveal #nat rows)) r i)})
    _
    phi;
  forevery_ext
    #(r:
      natlt (reveal #nat rows) {~(eq2 #(natlt (reveal #nat rows)) r i)})
    _
    (fun (r:
      natlt (reveal #nat rows) {~(eq2 #(natlt (reveal #nat rows)) r i)}) -> p r);
  forevery_insert p _;
  forevery_unrefine _;
  forevery_ext _ (fun x -> forall+ y . phi x y);
  forevery_ext_2 _ (fun r c ->
            pts_to_slice #et
              (T.core gm)
              #f
              (cell_of_pos #(reveal #nat rows) #(reveal #nat cols) l r c)
              (cell_of_pos #(reveal #nat rows) #(reveal #nat cols) l r c + 1)
              (cons #et
                  (acc2 #et #(reveal #nat rows) #(reveal #nat cols) em r c)
                  (empty #et)));

  // Convert back from pts_to_cell to M.pts_to_cell
  forevery_map_2
    (fun (r : natlt rows) (c : natlt cols) ->
      pts_to_cell (T.core gm) #f (cell_of_pos l r c) (acc2 em r c))
    (fun (r : natlt rows) (c : natlt cols) ->
      T.tensor_pts_to_cell gm #f (idx2 r c) (acc2 em r c))
    fn r c {
      T.tensor_pts_to_cell_eq gm (idx2 r c) f (acc2 em r c);
      rewrite
        pts_to_cell (T.core gm) #f (cell_of_pos l r c) (acc2 em r c)
      as
        T.tensor_pts_to_cell gm #f (idx2 r c) (acc2 em r c);
    };
  T.tensor_iraise2 gm;
}
#pop-options

#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn array2_vec_read
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : layout2 rows cols) {| T.ctlayout l, strided : strided_row_major (vtlayout_of_tlayout l) |}
  (gm : array2 et l)
  (i : szlt rows)
  (j : szlt (cols - chunk et + 1))
  (#f : perm)
  (#em : chest2 et rows cols)
  (arr : array et)
  (#s : erased (seq et))
  preserves gpu
  preserves gm |-> Frac f em
  requires  pure (aligned' 16 (T.core gm) (cell_of_pos l i j))
  requires  pure (aligned 16 arr)
  requires  arr |-> s
  requires  pure (Pulse.Lib.Array.length arr == chunk et)
  ensures   arr |-> Seq.init_ghost (chunk et) (fun x -> acc2 em i (j + x))
{
  Pulse.Lib.Array.pts_to_len arr;

  get_slice gm i j;

  strided.pf i j;
  strided.pf i (j + chunk et - 1);

  let offset = strided.offset +^ strided.stride *^ i +^ j;

  with s0.
    assert pts_to_slice (T.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + chunk et) s0;
  array_vec_cpy arr 0sz (T.core gm) offset;

  with ds1. assert pts_to arr ds1;

  unget_slice gm i j;

  assert pure (Seq.equal ds1 (Seq.init_ghost (chunk et) (fun x -> acc2 em i (j + x))));

  ();
}
#pop-options

(* ---------------------------------------------------------------- *)
(* Cell-level <-> slice-level bridge for a run of cells in one row.  *)
(* ---------------------------------------------------------------- *)

let row_cells_inv
  (#et : Type0) {| sized et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (f : perm)
  (i : natlt rows)
  (j : nat)
  (w : pos { j + w <= cols })
  (v : seq et { Seq.length v == w })
  (k : natle w)
  : slprop
= pts_to_slice (T.core gm) #f
    (cell_of_pos l i j) (cell_of_pos l i j + k) (Seq.slice v 0 k) **
  (forall+ (x : natlt w { x >= k }).
    pts_to_cell (T.core gm) #f
      (cell_of_pos l i ((j + x) <: natlt cols)) (Seq.index v x))

ghost
fn __row_cells_open
  (#et : Type0) {| sized et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols) {| T.ctlayout l |}
  (gm : array2 et l)
  (f : perm)
  (i : natlt rows)
  (j : nat)
  (w : pos { j + w <= cols })
  (v : seq et { Seq.length v == w })
  requires row_cells gm f i j w v
  ensures
    forall+ (x : natlt w).
      pts_to_cell (T.core gm) #f
        (cell_of_pos l i ((j + x) <: natlt cols)) (Seq.index v x)
{
  unfold row_cells gm f i j w v;
  forevery_map
    #(natlt w)
    (fun x -> T.tensor_pts_to_cell gm #f (idx2 i ((j + x) <: natlt cols)) (Seq.index v x))
    (fun x -> pts_to_cell (T.core gm) #f
                (cell_of_pos l i ((j + x) <: natlt cols)) (Seq.index v x))
    fn x {
      T.tensor_pts_to_cell_eq gm (idx2 i ((j + x) <: natlt cols)) f (Seq.index v x);
      rewrite
        T.tensor_pts_to_cell gm #f (idx2 i ((j + x) <: natlt cols)) (Seq.index v x)
      as
        pts_to_cell (T.core gm) #f
          (cell_of_pos l i ((j + x) <: natlt cols)) (Seq.index v x);
    };
}

ghost
fn __row_cells_close
  (#et : Type0) {| sized et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols) {| T.ctlayout l |}
  (gm : array2 et l)
  (f : perm)
  (i : natlt rows)
  (j : nat)
  (w : pos { j + w <= cols })
  (v : seq et { Seq.length v == w })
  requires
    forall+ (x : natlt w).
      pts_to_cell (T.core gm) #f
        (cell_of_pos l i ((j + x) <: natlt cols)) (Seq.index v x)
  ensures row_cells gm f i j w v
{
  forevery_map
    #(natlt w)
    (fun x -> pts_to_cell (T.core gm) #f
                (cell_of_pos l i ((j + x) <: natlt cols)) (Seq.index v x))
    (fun x -> T.tensor_pts_to_cell gm #f (idx2 i ((j + x) <: natlt cols)) (Seq.index v x))
    fn x {
      T.tensor_pts_to_cell_eq gm (idx2 i ((j + x) <: natlt cols)) f (Seq.index v x);
      rewrite
        pts_to_cell (T.core gm) #f
          (cell_of_pos l i ((j + x) <: natlt cols)) (Seq.index v x)
      as
        T.tensor_pts_to_cell gm #f (idx2 i ((j + x) <: natlt cols)) (Seq.index v x);
    };
  fold row_cells gm f i j w v;
}

#push-options "--z3rlimit 40"
ghost
fn __row_cells_step
  (#et : Type0) {| sized et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols) {| strided : strided_row_major (vtlayout_of_tlayout l) |}
  (gm : array2 et l)
  (f : perm)
  (i : natlt rows)
  (j : nat)
  (w : pos { j + w <= cols })
  (v : seq et { Seq.length v == w })
  (k : nat { k < w })
  requires row_cells_inv gm f i j w v k
  ensures  row_cells_inv gm f i j w v (k + 1)
{
  unfold row_cells_inv gm f i j w v k;
  forevery_remove'
    #(natlt w)
    (fun x -> x >= k)
    (fun x -> pts_to_cell (T.core gm) #f
                (cell_of_pos l i ((j + x) <: natlt cols)) (Seq.index v x))
    k;
  forevery_refine_ext
    (fun (x : natlt w) -> x >= k + 1)
    (fun (x : natlt w) -> pts_to_cell (T.core gm) #f
                (cell_of_pos l i ((j + x) <: natlt cols)) (Seq.index v x));
  strided_row_major_contiguous l i j ((j + k) <: natlt cols);
  assert pure (cell_of_pos l i ((j + k) <: natlt cols) == cell_of_pos l i j + k);
  rewrite
    pts_to_cell (T.core gm) #f
      (cell_of_pos l i ((j + k) <: natlt cols)) (Seq.index v k)
  as
    pts_to_cell (T.core gm) #f (cell_of_pos l i j + k) (Seq.index v k);
  slice_concat (T.core gm) #f _ (cell_of_pos l i j + k) _;
  assert pure (Seq.equal
    (Seq.slice v 0 k `Seq.append` seq![Seq.index v k])
    (Seq.slice v 0 (k + 1)));
  fold row_cells_inv gm f i j w v (k + 1);
}
#pop-options

ghost
fn rec __row_cells_up
  (#et : Type0) {| sized et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols) {| strided : strided_row_major (vtlayout_of_tlayout l) |}
  (gm : array2 et l)
  (f : perm)
  (i : natlt rows)
  (j : nat)
  (w : pos { j + w <= cols })
  (v : seq et { Seq.length v == w })
  (k : natle w)
  requires row_cells_inv gm f i j w v k
  ensures  row_cells_inv gm f i j w v w
  decreases (w - k)
{
  let eq = k = w;
  if (eq) {
    rewrite (row_cells_inv gm f i j w v k)
        as  (row_cells_inv gm f i j w v w);
    ();
  } else {
    __row_cells_step gm f i j w v k;
    __row_cells_up gm f i j w v (k + 1);
  }
}

#push-options "--z3rlimit 40"
ghost
fn __row_cells_unstep
  (#et : Type0) {| sized et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols) {| strided : strided_row_major (vtlayout_of_tlayout l) |}
  (gm : array2 et l)
  (f : perm)
  (i : natlt rows)
  (j : nat)
  (w : pos { j + w <= cols })
  (v : seq et { Seq.length v == w })
  (k : nat { k < w })
  requires row_cells_inv gm f i j w v (k + 1)
  ensures  row_cells_inv gm f i j w v k
{
  unfold row_cells_inv gm f i j w v (k + 1);
  assert pure (Seq.equal
    (Seq.slice v 0 k `Seq.append` seq![Seq.index v k])
    (Seq.slice v 0 (k + 1)));
  slice_split (T.core gm) #f
    #(Seq.slice v 0 k) #(seq![Seq.index v k])
    _ (cell_of_pos l i j + k) _;
  strided_row_major_contiguous l i j ((j + k) <: natlt cols);
  assert pure (cell_of_pos l i ((j + k) <: natlt cols) == cell_of_pos l i j + k);
  rewrite
    pts_to_cell (T.core gm) #f (cell_of_pos l i j + k) (Seq.index v k)
  as
    pts_to_cell (T.core gm) #f
      (cell_of_pos l i ((j + k) <: natlt cols)) (Seq.index v k);
  forevery_insert
    #(natlt w)
    #(fun (x : natlt w) -> x >= k + 1)
    (fun (x : natlt w) -> pts_to_cell (T.core gm) #f
                (cell_of_pos l i ((j + x) <: natlt cols)) (Seq.index v x))
    k;
  forevery_refine_ext
    (fun (x : natlt w) -> x >= k)
    (fun (x : natlt w) -> pts_to_cell (T.core gm) #f
                (cell_of_pos l i ((j + x) <: natlt cols)) (Seq.index v x));
  fold row_cells_inv gm f i j w v k;
}
#pop-options

ghost
fn rec __row_cells_down
  (#et : Type0) {| sized et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols) {| strided : strided_row_major (vtlayout_of_tlayout l) |}
  (gm : array2 et l)
  (f : perm)
  (i : natlt rows)
  (j : nat)
  (w : pos { j + w <= cols })
  (v : seq et { Seq.length v == w })
  (k : natle w)
  requires row_cells_inv gm f i j w v w
  ensures  row_cells_inv gm f i j w v k
  decreases (w - k)
{
  let eq = k = w;
  if (eq) {
    rewrite (row_cells_inv gm f i j w v w)
        as  (row_cells_inv gm f i j w v k);
    ();
  } else {
    __row_cells_down gm f i j w v (k + 1);
    __row_cells_unstep gm f i j w v k;
  }
}

#push-options "--z3rlimit 40"
ghost
fn row_cells_to_slice
  (#et : Type0) {| sized et |}
  (#rows #cols : erased nat)
  (#l : layout2 rows cols) {| T.ctlayout l, strided : strided_row_major (vtlayout_of_tlayout l) |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat)
  (w : pos { j + w <= cols })
  (#f : perm)
  (v : seq et { Seq.length v == w })
  requires row_cells gm f i j w v
  ensures
    pts_to_slice (T.core gm) #f
      (cell_of_pos l i j) (cell_of_pos l i j + w) v
{
  __row_cells_open gm f i j w v;
  forevery_remove #(natlt w) _ 0;
  rewrite each ((j + 0) <: natlt cols) as (j <: natlt cols);
  gpu_slice_split' (T.core gm) #f
    (cell_of_pos l i j) (cell_of_pos l i j + 0) (cell_of_pos l i j + 1);
  assert pure (Seq.equal (Seq.slice v 0 0) seq![]);
  assert pts_to_slice (T.core gm) #f
    (cell_of_pos l i j) (cell_of_pos l i j + 0) (Seq.slice v 0 0);
  assert pure (Seq.equal
    (Kuiper.Seq.Common.seq_drop (cell_of_pos l i j + 0 - cell_of_pos l i j)
      seq![Seq.index v 0])
    seq![Seq.index v 0]);
  rewrite
    pts_to_slice (T.core gm) #f
      (cell_of_pos l i j + 0) (cell_of_pos l i j + 1) seq![Seq.index v 0]
  as
    pts_to_cell (T.core gm) #f
      (cell_of_pos l i ((j + 0) <: natlt cols)) (Seq.index v 0);
  forevery_insert
    #(natlt w)
    #(fun (x : natlt w) -> ~(eq2 #(natlt w) x 0))
    (fun (x : natlt w) -> pts_to_cell (T.core gm) #f
                (cell_of_pos l i ((j + x) <: natlt cols)) (Seq.index v x))
    0;
  forevery_refine_ext
    (fun (x : natlt w) -> x >= 0)
    (fun (x : natlt w) -> pts_to_cell (T.core gm) #f
                (cell_of_pos l i ((j + x) <: natlt cols)) (Seq.index v x));
  fold row_cells_inv gm f i j w v 0;
  __row_cells_up gm f i j w v 0;
  unfold row_cells_inv gm f i j w v w;
  forevery_elim_empty
    #(x : natlt w { x >= w })
    (fun (x : natlt w { x >= w }) -> pts_to_cell (T.core gm) #f
                (cell_of_pos l i ((j + x) <: natlt cols)) (Seq.index v x));
  assert pure (Seq.equal (Seq.slice v 0 w) v);
}
#pop-options

#push-options "--z3rlimit 40"
ghost
fn row_slice_to_cells
  (#et : Type0) {| sized et |}
  (#rows #cols : erased nat)
  (#l : layout2 rows cols) {| T.ctlayout l, strided : strided_row_major (vtlayout_of_tlayout l) |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat)
  (w : pos { j + w <= cols })
  (#f : perm)
  (v : seq et { Seq.length v == w })
  requires
    pts_to_slice (T.core gm) #f
      (cell_of_pos l i j) (cell_of_pos l i j + w) v
  ensures row_cells gm f i j w v
{
  assert pure (Seq.equal (Seq.slice v 0 w) v);
  rewrite
    pts_to_slice (T.core gm) #f
      (cell_of_pos l i j) (cell_of_pos l i j + w) v
  as
    pts_to_slice (T.core gm) #f
      (cell_of_pos l i j) (cell_of_pos l i j + w) (Seq.slice v 0 w);
  forevery_intro_empty
    #(x : natlt w { x >= w })
    (fun (x : natlt w { x >= w }) -> pts_to_cell (T.core gm) #f
                (cell_of_pos l i ((j + x) <: natlt cols)) (Seq.index v x));
  fold row_cells_inv gm f i j w v w;
  __row_cells_down gm f i j w v 0;
  unfold row_cells_inv gm f i j w v 0;
  assert pure (Seq.equal (Seq.slice v 0 0) seq![]);
  forevery_remove' #(natlt w) (fun (x : natlt w) -> x >= 0) _ 0;
  rewrite
    pts_to_cell (T.core gm) #f
      (cell_of_pos l i ((j + 0) <: natlt cols)) (Seq.index v 0)
  as
    pts_to_slice (T.core gm) #f
      (cell_of_pos l i j + 0) (cell_of_pos l i j + 1) seq![Seq.index v 0];
  slice_concat (T.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + 0) _;
  assert pure (Seq.equal (Seq.slice v 0 0 `Seq.append` seq![Seq.index v 0])
                         seq![Seq.index v 0]);
  rewrite
    pts_to_slice (T.core gm) #f
      (cell_of_pos l i j) (cell_of_pos l i j + 1) seq![Seq.index v 0]
  as
    pts_to_cell (T.core gm) #f
      (cell_of_pos l i ((j + 0) <: natlt cols)) (Seq.index v 0);
  forevery_insert
    #(natlt w)
    #(fun (x : natlt w) -> x >= 0 /\ ~(eq2 #(natlt w) x 0))
    (fun (x : natlt w) -> pts_to_cell (T.core gm) #f
                (cell_of_pos l i ((j + x) <: natlt cols)) (Seq.index v x))
    0;
  forevery_unrefine
    (fun (x : natlt w) -> pts_to_cell (T.core gm) #f
                (cell_of_pos l i ((j + x) <: natlt cols)) (Seq.index v x));
  __row_cells_close gm f i j w v;
}
#pop-options

#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn array2_vec_write_cells
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : layout2 rows cols) {| T.ctlayout l, strided : strided_row_major (vtlayout_of_tlayout l) |}
  (gm : array2 et l)
  (i : szlt rows)
  (j : szlt (cols - chunk et + 1))
  (arr : array et)
  (#f : perm)
  (old nv : erased (seq et))
  (#_ : squash (Seq.length old == chunk et /\ Seq.length nv == chunk et))
  ()
  preserves gpu
  preserves arr |-> Frac f nv
  requires  row_cells gm 1.0R i j (chunk et) (reveal old)
  requires  pure (aligned' 16 (T.core gm) (cell_of_pos l i j))
  requires  pure (aligned 16 arr)
  ensures   row_cells gm 1.0R i j (chunk et) (reveal nv)
{
  Pulse.Lib.Array.pts_to_len arr;

  row_cells_to_slice gm (SZ.v i) (SZ.v j) (chunk et) (reveal old);

  strided.pf i j;
  let offset = strided.offset +^ strided.stride *^ i +^ j;

  array_vec_cpy (T.core gm) offset arr 0sz;

  with s'.
    assert pts_to_slice (T.core gm)
      (cell_of_pos l i j) (cell_of_pos l i j + chunk et) s';
  assert pure (Seq.equal s' (reveal nv));
  rewrite
    pts_to_slice (T.core gm)
      (cell_of_pos l i j) (cell_of_pos l i j + chunk et) s'
  as
    pts_to_slice (T.core gm)
      (cell_of_pos l i j) (cell_of_pos l i j + chunk et) (reveal nv);

  row_slice_to_cells gm (SZ.v i) (SZ.v j) (chunk et) (reveal nv);
}
#pop-options

#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn array2_vec_write
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : layout2 rows cols) {| T.ctlayout l, strided : strided_row_major (vtlayout_of_tlayout l) |}
  (gm : array2 et l)
  (i : szlt rows)
  (j : szlt (cols - chunk et + 1))
  (#em : chest2 et rows cols)
  (arr : array et)
  (#f : perm)
  (nv : erased (seq et))
  (#_ : squash (Seq.length nv == chunk et))
  ()
  preserves gpu
  preserves arr |-> Frac f nv
  requires  gm |-> em
  requires  pure (aligned' 16 (T.core gm) (cell_of_pos l i j))
  requires  pure (aligned 16 arr)
  ensures   gm |-> chest2_row_blit em i j (chunk et) (reveal nv)
{
  Pulse.Lib.Array.pts_to_len arr;

  let em' : chest2 et rows cols =
    chest2_row_blit em (SZ.v i) (SZ.v j) (chunk et) (reveal nv);

  get_slice gm (SZ.v i) (SZ.v j);

  strided.pf i j;
  let offset = strided.offset +^ strided.stride *^ i +^ j;

  array_vec_cpy (T.core gm) offset arr 0sz;

  with s'.
    assert pts_to_slice (T.core gm)
      (cell_of_pos l i j) (cell_of_pos l i j + chunk et) s';
  assert pure (Seq.equal s'
    (Seq.init_ghost (chunk et) (fun x -> acc2 em' (SZ.v i) (SZ.v j + x))));
  rewrite
    pts_to_slice (T.core gm)
      (cell_of_pos l i j) (cell_of_pos l i j + chunk et) s'
  as
    pts_to_slice (T.core gm)
      (cell_of_pos l i j) (cell_of_pos l i j + chunk et)
      (Seq.init_ghost (chunk et) (fun x -> acc2 em' (SZ.v i) (SZ.v j + x)));

  forevery_map
    #(x : natlt cols { all_but_window cols (SZ.v j) (chunk et) x })
    (fun x -> pts_to_cell (T.core gm) (cell_of_pos l i x) (acc2 em i x))
    (fun x -> pts_to_cell (T.core gm) (cell_of_pos l i x) (acc2 em' i x))
    fn x {
      rewrite
        pts_to_cell (T.core gm) (cell_of_pos l i x) (acc2 em i x)
      as
        pts_to_cell (T.core gm) (cell_of_pos l i x) (acc2 em' i x);
    };
  forevery_map_2
    #(r : natlt rows { ~ (eq2 #(natlt rows) r (SZ.v i)) }) #(natlt cols)
    (fun r c -> pts_to_cell (T.core gm) (cell_of_pos l r c) (acc2 em r c))
    (fun r c -> pts_to_cell (T.core gm) (cell_of_pos l r c) (acc2 em' r c))
    fn r c {
      rewrite
        pts_to_cell (T.core gm) (cell_of_pos l r c) (acc2 em r c)
      as
        pts_to_cell (T.core gm) (cell_of_pos l r c) (acc2 em' r c);
    };

  unget_slice gm (SZ.v i) (SZ.v j) #1.0R #em';
}
#pop-options

let vstrided_row_run
  (#rows #cols : nat)
  (l : RO.vlayout2 rows cols) {| d : strided_row_major l |}
  (i : natlt rows)
  (j : nat)
  (w : nat { j + w <= cols })
  : Lemma (forall (x : natlt w).
             vcell_of_pos l i (j + x) == d.offset + d.stride * i + j + x)
  = introduce forall (x : natlt w).
                vcell_of_pos l i (j + x) == d.offset + d.stride * i + j + x
    with d.pf i (j + x)

inline_for_extraction noextract
fn roarray2_vec_read
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : RO.vlayout2 rows cols) {| strided : strided_row_major l |}
  (gm : RO.roarray2 et l)
  (i : szlt rows)
  (j : szlt (cols - chunk et + 1))
  (#f : perm)
  (#em : chest2 et rows cols)
  (arr : array et)
  (#s : erased (seq et))
  preserves gpu
  preserves gm |-> Frac f em
  requires  pure (aligned' 16 (RO.core gm) (vcell_of_pos l i j))
  requires  pure (aligned 16 arr)
  requires  arr |-> s
  requires  pure (Pulse.Lib.Array.length arr == chunk et)
  ensures   arr |-> Seq.init_ghost (chunk et) (fun x -> acc2 em i (j + x))
{
  Pulse.Lib.Array.pts_to_len arr;
  RO.tensor_to_raw gm;
  with v. assert RO.core gm |-> Frac f v;

  strided.pf i j;
  strided.pf i (j + chunk et - 1);
  vstrided_row_run l i j (chunk et);
  assert pure (vcell_of_pos l i (j + chunk et - 1) < RO.vtlayout_ulen l);
  let offset = strided.offset +^ strided.stride *^ i +^ j;

  array_to_slice (RO.core gm);
  array_to_slice arr;
  array_vec_cpy arr 0sz (RO.core gm) offset;
  with ds1. assert pts_to_slice arr 0 (chunk et) ds1;
  slice_to_array arr;
  slice_to_array (RO.core gm);
  RO.raw_to_tensor gm em;

  assert pure (Seq.equal ds1 (Seq.init_ghost (chunk et) (fun x -> acc2 em i (j + x))));
  ();
}
