module Kuiper.Array2.Vectorized

(* Vectorized read for Array2, analogous to Kuiper.Matrix.Vectorized.
   Ported from Kuiper.Matrix.Vectorized with Array2 types. *)

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.EMatrix

open Kuiper.Array2 { array2, layout }
open Kuiper.Array2.Strided
module M = Kuiper.Array2
module T = Kuiper.Tensor

let strided_row_major_contiguous
  (#rows #cols : erased nat)
  (l : layout rows cols) {| d : strided_row_major l |}
  (i : natlt rows)
  (j1 j2 : natlt cols)
  : Lemma (cell_of_pos l i j2 - cell_of_pos l i j1 == j2 - j1)
  = d.pf i j1; d.pf i j2

let all_but_window l j k : natlt l -> prop =
  fun i -> i < j \/ i >= j + k

let get_slice_inv
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout rows cols) {| strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : natlt (cols - chunk et + 1))
  (f : perm)
  (em : ematrix et rows cols)
  (k : nat {k <= chunk et})
  : slprop
  =
  gpu_pts_to_slice (M.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + k)
      (Seq.init_ghost k (fun x -> macc em i (j + x))) **
  (forall+ (x : natlt cols {all_but_window cols j k x}).
    gpu_pts_to_cell (M.core gm) #f (cell_of_pos l i x) (macc em i x))

ghost
fn __get_slice_step
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : nat {i < rows})
  (j : nat {j < cols - chunk et + 1})
  (k : nat {k < chunk et})
  (#f : perm)
  (#em : ematrix et rows cols)
  requires get_slice_inv gm i j f em k
  ensures  get_slice_inv gm i j f em (k + 1)
{
  unfold get_slice_inv gm i j f em k;
  forevery_remove' #(natlt cols)
    (fun x -> all_but_window cols j k x)
    (fun x -> gpu_pts_to_cell (M.core gm) #f (cell_of_pos l i x) (macc em i x))
    (j + k);
  forevery_refine_ext
    (fun (x : natlt cols) -> all_but_window cols j (k + 1) x)
    (fun (x : natlt cols) -> gpu_pts_to_cell (M.core gm) #f (cell_of_pos l i x) (macc em i x));

  assert gpu_pts_to_cell (M.core gm) #f (cell_of_pos l i (j + k)) (macc em i (j + k));

  strided_row_major_contiguous l i j (j + k);
  assert pure (j + k < cols);
  assert pure (cell_of_pos l i (j + k) == cell_of_pos l i j + k);

  rewrite
    gpu_pts_to_cell (M.core gm) #f (cell_of_pos l i (j + k)) (macc em i (j + k))
  as
    gpu_pts_to_cell (M.core gm) #f (cell_of_pos l i j + k) (macc em i (j + k));

  gpu_slice_concat (M.core gm) #f _ (cell_of_pos l i j + k) _;

  assert pure (Seq.equal
      (Seq.init_ghost k (fun x -> macc em i (j + x)) `Seq.append` seq![macc em i (j + k)])
      (Seq.init_ghost (k + 1) (fun x -> macc em i (j + x))));

  fold get_slice_inv gm i j f em (k + 1);

  ();
}

ghost
fn rec __get_slice
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : natlt (cols - chunk et + 1))
  (k : nat {k <= chunk et})
  (#f : perm)
  (#em : ematrix et rows cols)
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
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : nat{i < rows})
  (j : nat{j < cols - chunk et + 1})
  (#f : perm)
  (#em : ematrix et rows cols)
  requires  gm |-> Frac f em
  ensures
    gpu_pts_to_slice (M.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + chunk et)
      (Seq.init_ghost (chunk et) (fun x -> macc em i (j + x))) **
    (forall+ (x : natlt cols{all_but_window cols j (chunk et) x}).
      gpu_pts_to_cell (M.core gm) #f (cell_of_pos l i x) (macc em i x)) **
    (forall+ (r : natlt rows { ~ (eq2 #(natlt rows) r i) } ) (c : natlt cols).
      gpu_pts_to_cell (M.core gm) #f (cell_of_pos l r c) (macc em r c))
{
  M.ilower gm;
  // Convert pts_to_cell to gpu_pts_to_cell
  forevery_map_2
    (fun (r : natlt rows) (c : natlt cols) ->
      M.pts_to_cell gm #f (r, c) (macc em r c))
    (fun (r : natlt rows) (c : natlt cols) ->
      gpu_pts_to_cell (M.core gm) #f (cell_of_pos l r c) (macc em r c))
    fn r c {
      M.pts_to_cell_eq gm (r, c) f (macc em r c);
      rewrite
        M.pts_to_cell gm #f (r, c) (macc em r c)
      as
        gpu_pts_to_cell (M.core gm) #f (cell_of_pos l r c) (macc em r c);
    };

  // Extract row i
  forevery_remove #(natlt rows) _ i;

  // Extract cell j and create empty slice
  forevery_remove #(natlt cols) _ j;
  gpu_slice_split' (M.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + 0) (cell_of_pos l i j + 1);
  assert pure (seq![] `Seq.equal` Seq.init_ghost 0 (fun x -> macc em i (j + x)));
  assert gpu_pts_to_slice (M.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + 0)
    (Seq.init_ghost 0 (fun x -> macc em i (j + x)));
  assert pure (Seq.equal (Kuiper.Seq.Common.seq_drop (cell_of_pos l i j + 0 - cell_of_pos l i j)
          seq![macc em i j]) seq![macc em i j]);
  assert gpu_pts_to_slice (M.core gm) #f (cell_of_pos l i j + 0) (cell_of_pos l i j + 1)
    seq![macc em i j];
  rewrite gpu_pts_to_slice (M.core gm) #f (cell_of_pos l i j + 0) (cell_of_pos l i j + 1)
    seq![macc em i j]
    as gpu_pts_to_slice (M.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + 1)
    seq![macc em i j];
  forevery_insert #(natlt cols) #(fun x -> ~(eq2 #(natlt cols) x j))
    (fun x ->
      gpu_pts_to_slice (M.core gm) #f
      (cell_of_pos l i x)
      (cell_of_pos l i x + 1)
      seq![macc em i x]) j;
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
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : nat {i < rows})
  (j : nat {j < cols - chunk et + 1})
  (k : nat {k < chunk et})
  (#f : perm)
  (#em : ematrix et rows cols)
  requires get_slice_inv gm i j f em (k + 1)
  ensures  get_slice_inv gm i j f em k
{
  unfold get_slice_inv gm i j f em (k + 1);
  assert pure (Seq.equal
      (Seq.init_ghost k (fun x -> macc em i (j + x)) `Seq.append` seq![macc em i (j + k)])
      (Seq.init_ghost (k + 1) (fun x -> macc em i (j + x))));
  gpu_slice_split (M.core gm) #f #(Seq.init_ghost k (fun x -> macc em i (j + x))) #(seq![macc em i (j + k)]) _ (cell_of_pos l i j + k) _;
  strided_row_major_contiguous l i j (j + k);
  assert pure (j + k < cols);
  assert pure (cell_of_pos l i (j + k) == cell_of_pos l i j + k);
  rewrite
    gpu_pts_to_cell (M.core gm) #f (cell_of_pos l i j + k) (macc em i (j + k))
  as
    gpu_pts_to_cell (M.core gm) #f (cell_of_pos l i (j + k)) (macc em i (j + k));
  assert gpu_pts_to_cell (M.core gm) #f (cell_of_pos l i (j + k)) (macc em i (j + k));
  forevery_insert #(natlt cols)
    (fun x -> gpu_pts_to_cell (M.core gm) #f (cell_of_pos l i x) (macc em i x))
    (j + k);

  forevery_refine_ext
    (fun (x : natlt cols) -> all_but_window cols j k x)
    (fun (x : natlt cols) -> gpu_pts_to_cell (M.core gm) #f (cell_of_pos l i x) (macc em i x));

  fold get_slice_inv gm i j f em k;

  ();
}
#pop-options

ghost
fn rec __unget_slice
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : natlt (cols - chunk et + 1))
  (k : nat {k <= chunk et})
  (#f : perm)
  (#em : ematrix et rows cols)
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
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : nat{i < rows})
  (j : nat{j < cols - chunk et + 1})
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    gpu_pts_to_slice (M.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + chunk et)
      (Seq.init_ghost (chunk et) (fun x -> macc em i (j + x))) **
    (forall+ (x : natlt cols{all_but_window cols j (chunk et) x}).
      gpu_pts_to_cell (M.core gm) #f (cell_of_pos l i x) (macc em i x)) **
    (forall+ (r : natlt rows { ~ (eq2 #(natlt rows) r i) } ) (c : natlt cols).
      gpu_pts_to_cell (M.core gm) #f (cell_of_pos l r c) (macc em r c))
  ensures  gm |-> Frac f em
{
  fold get_slice_inv gm i j f em (chunk et);
  __unget_slice gm i j 0;
  unfold get_slice_inv gm i j f em 0;
  forevery_unrefine #(natlt cols) _;
  with s0 . assert  gpu_pts_to_slice (M.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j) s0;

  // Merge the empty slice back into j
  forevery_remove #(natlt cols) _ j;
  gpu_slice_concat (M.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j) _;
  with s1 . assert (gpu_pts_to_slice (M.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + 1) s1);
  assert pure (Seq.equal s1 seq![macc em i j]);
  assert (gpu_pts_to_slice (M.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + 1) seq![macc em i j]);
  forevery_insert #(natlt cols) #(fun x -> ~(eq2 #(natlt cols) x j)) (fun x ->
        gpu_pts_to_slice (M.core gm) #f
        (cell_of_pos l i x)
        (cell_of_pos l i x + 1)
        seq![macc em i x] ) j;
  forevery_unrefine #(natlt cols) _;

  let phi = (fun (r: natlt rows) (c: natlt (reveal #nat cols)) ->
            gpu_pts_to_slice #et
              #(M.layout_size #(reveal #nat rows) #(reveal #nat cols) l)
              (M.core #et #rows #cols #l gm)
              #f
              (cell_of_pos #(reveal #nat rows) #(reveal #nat cols) l r c)
              (cell_of_pos #(reveal #nat rows) #(reveal #nat cols) l r c + 1)
              (cons #et
                  (macc #et #(reveal #nat rows) #(reveal #nat cols) em r c)
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
            gpu_pts_to_slice #et
              #(M.layout_size #(reveal #nat rows) #(reveal #nat cols) l)
              (M.core #et #rows #cols #l gm)
              #f
              (cell_of_pos #(reveal #nat rows) #(reveal #nat cols) l r c)
              (cell_of_pos #(reveal #nat rows) #(reveal #nat cols) l r c + 1)
              (cons #et
                  (macc #et #(reveal #nat rows) #(reveal #nat cols) em r c)
                  (empty #et)));

  // Convert back from gpu_pts_to_cell to M.pts_to_cell
  forevery_map_2
    (fun (r : natlt rows) (c : natlt cols) ->
      gpu_pts_to_cell (M.core gm) #f (cell_of_pos l r c) (macc em r c))
    (fun (r : natlt rows) (c : natlt cols) ->
      M.pts_to_cell gm #f (r, c) (macc em r c))
    fn r c {
      M.pts_to_cell_eq gm (r, c) f (macc em r c);
      rewrite
        gpu_pts_to_cell (M.core gm) #f (cell_of_pos l r c) (macc em r c)
      as
        M.pts_to_cell gm #f (r, c) (macc em r c);
    };
  M.iraise gm;
}
#pop-options

#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn array2_vec_read
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : szlt rows)
  (j : szlt (cols - chunk et + 1))
  (#f : perm)
  (#em : ematrix et rows cols)
  (arr : array et)
  (#s : erased (seq et))
  preserves gpu
  preserves gm |-> Frac f em
  requires  pure (aligned' 16 (M.core gm) (cell_of_pos l i j))
  requires  arr |-> s
  requires  pure (Pulse.Lib.Array.length arr == chunk et)
  ensures   arr |-> Seq.init_ghost (chunk et) (fun x -> macc em i (j + x))
{
  Pulse.Lib.Array.pts_to_len arr;

  get_slice gm i j;

  strided.pf i j;
  strided.pf i (j + chunk et - 1);

  let offset = strided.offset +^ strided.stride *^ i +^ j;

  with s0.
    assert gpu_pts_to_slice (M.core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + chunk et) s0;
  gpu_array_vec_cpy_dh arr 0sz (M.core gm) offset;

  with ds1. assert pts_to arr ds1;

  unget_slice gm i j;

  assert pure (Seq.equal ds1 (Seq.init_ghost (chunk et) (fun x -> macc em i (j + x))));

  ();
}
#pop-options
