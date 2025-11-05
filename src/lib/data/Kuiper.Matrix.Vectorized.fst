module Kuiper.Matrix.Vectorized

#lang-pulse

friend Kuiper.Matrix

open Kuiper.Matrix.Common

open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.Matrix
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type

let strided_row_major_contiguous
  (#rows #cols : erased nat)
  (l : mlayout rows cols) {| d : strided_row_major l |}
  (i : natlt rows)
  (j1 j2 : natlt cols)
  : Lemma (cell_of_pos l i j2 - cell_of_pos l i j1 == j2 - j1)
  = d.pf i j1; d.pf i j2

let all_but_window l j k : natlt l -> prop =
  fun i -> i < j \/ i >= j + k

let get_slice_inv
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : mlayout rows cols) {| strided : strided_row_major l |}
  (gm : gpu_matrix et l)
  (i : natlt rows)
  (j : natlt (cols - chunk et + 1))
  (f : perm)
  (em : ematrix et rows cols)
  (k : nat {k <= chunk et})
  : slprop
  =
  gpu_pts_to_slice (core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + k)
      (Seq.init_ghost k (fun x -> macc em i (j + x))) **
  (forall+ (x : natlt cols {all_but_window cols j k x}).
    gpu_pts_to_cell (core gm) #f (cell_of_pos l i x) (macc em i x))

ghost
fn __get_slice_step
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : mlayout rows cols) {| clayout l, strided : strided_row_major l |}
  (gm : gpu_matrix et l)
  (i : nat {i < rows})
  (j : nat {j < cols - chunk et + 1}) // using natlt gives terrible issues below, see https://github.com/FStarLang/pulse/issues/495
  (k : nat {k < chunk et})
  (#f : perm)
  (#em : ematrix et rows cols)
  requires get_slice_inv gm i j f em k
  ensures  get_slice_inv gm i j f em (k + 1)
{
  unfold get_slice_inv gm i j f em k;
  forevery_remove' #(natlt cols)
    (fun x -> all_but_window cols j k x)
    (fun x -> gpu_pts_to_cell (core gm) #f (cell_of_pos l i x) (macc em i x))
    (j + k);
  forevery_refine_ext
    (fun (x : natlt cols) -> all_but_window cols j (k + 1) x)
    (fun (x : natlt cols) -> gpu_pts_to_cell (core gm) #f (cell_of_pos l i x) (macc em i x));

  assert gpu_pts_to_cell (core gm) #f (cell_of_pos l i (j + k)) (macc em i (j + k));

  strided_row_major_contiguous l i j (j + k);
  assert pure (j + k < cols);
  assert pure (cell_of_pos l i (j + k) == cell_of_pos l i j + k);

  rewrite
    gpu_pts_to_cell (core gm) #f (cell_of_pos l i (j + k)) (macc em i (j + k))
  as
    gpu_pts_to_cell (core gm) #f (cell_of_pos l i j + k) (macc em i (j + k));

  gpu_slice_concat (core gm) #f _ (cell_of_pos l i j + k) _;

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
  (#l : mlayout rows cols) {| clayout l, strided : strided_row_major l |}
  (gm : gpu_matrix et l)
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
  (#l : mlayout rows cols) {| clayout l, strided : strided_row_major l |}
  (gm : gpu_matrix et l)
  (i : nat{i < rows})
  (j : nat{j < cols - chunk et + 1})
  (#f : perm)
  (#em : ematrix et rows cols)
  requires  gm |-> Frac f em
  ensures
    // The slice
    gpu_pts_to_slice (core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + chunk et)
      (Seq.init_ghost (chunk et) (fun x -> macc em i (j + x))) **
    // Rest of this row
    (forall+ (x : natlt cols{all_but_window cols j (chunk et) x}).
      gpu_pts_to_cell (core gm) #f (cell_of_pos l i x) (macc em i x)) **
    // All other rows
    (forall+ (r : natlt rows { ~ (eq2 #(natlt rows) r i) } ) (c : natlt cols).
      gpu_pts_to_cell (core gm) #f (cell_of_pos l r c) (macc em r c))
{
  // View matrix as its set of cells
  gpu_matrix_iconcr gm;
  let p = core gm;
  assert rewrites_to p (core gm);
  // Extract permission to the row we're interested in
  forevery_remove #(natlt rows) _ i;

  gpu_slice_empty_intro (core gm) (cell_of_pos l i j) #f;
  assert pure (seq![] `Seq.equal` Seq.init_ghost 0 (fun x -> macc em i (j + x)));
  assert gpu_pts_to_slice (core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + 0)
    (Seq.init_ghost 0 (fun x -> macc em i (j + x)));
  forevery_refine_ext
    (fun (x : natlt cols) -> all_but_window cols j 0 x)
    _;
  fold get_slice_inv gm i j f em 0;
  __get_slice gm i j 0;
  unfold get_slice_inv gm i j f em (chunk et);

  ();
}

ghost
fn unget_slice
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| clayout l, strided : strided_row_major l |}
  (gm : gpu_matrix et l)
  (i : szlt rows)
  (j : szlt (cols - chunk et + 1))
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    // The slice
    gpu_pts_to_slice (core gm) #f (cell_of_pos l i j) (cell_of_pos l i j + chunk et)
      (Seq.init_ghost (chunk et) (fun x -> macc em i (j + x))) **
    // Rest of this row
    (forall+ (x : natlt cols{all_but_window cols j (chunk et) x}).
      gpu_pts_to_cell (core gm) #f (cell_of_pos l i x) (macc em i x)) **
    // All other rows
    (forall+ (r : natlt rows { ~ (eq2 #(natlt rows) r i) } ) (c : natlt cols).
      gpu_pts_to_cell (core gm) #f (cell_of_pos l r c) (macc em r c))
  ensures  gm |-> Frac f em
{
  (* the way back of the above... every step is invertible, this should work out fine. *)
  admit();
}

#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn gpu_matrix_vec_read
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| clayout l, strided : strided_row_major l |}
  (gm : gpu_matrix et l)
  (i : szlt rows)
  (j : szlt (cols - chunk et + 1))
  (#f : perm)
  (#em : ematrix et rows cols)
  (arr : array et)
  (#s : erased (seq et))
  preserves gpu
  preserves gm |-> Frac f em
  requires  pure (aligned' 16 (core gm) (cell_of_pos l i j))
  requires  arr |-> s
  requires  pure (Pulse.Lib.Array.length arr >= chunk et)
  ensures   arr |-> Seq.init_ghost (chunk et) (fun x -> macc em i (j + x))
{
  Pulse.Lib.Array.pts_to_len arr;

  get_slice gm i j;

  strided.pf i j;
  strided.pf i (j + chunk et - 1);

  let offset = strided.offset +^ strided.stride *^ i +^ j;

  gpu_array_vec_cpy_dh arr 0sz (core gm) offset;

  unget_slice gm i j;

  with s. assert pts_to arr s;
  assert pure (Seq.equal s (Seq.init_ghost (chunk et) (fun x -> macc em i (j + x))));

  ();
}

inline_for_extraction noextract
fn gpu_matrix_vec_read'
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| clayout l, strided_row_major l |}
  (gm : gpu_matrix et l)
  (i : szlt rows)
  (j : szlt (cols - chunk et + 1))
  (#f : perm)
  (#em : ematrix et rows cols)
  (arr : array et)
  (#s : erased (seq et))
  preserves gpu
  preserves gm |-> Frac f em
  requires  arr |-> s
  requires  pure (Pulse.Lib.Array.length arr >= chunk et)
  ensures   exists* (s': lseq et (chunk et)). arr |-> s' **
    pure (forall x. Seq.index s' x == macc em i (j + x))
{
  gpu_matrix_vec_read gm i j arr;
}

inline_for_extraction noextract
fn gpu_matrix_vec_cp_async
  (#et:Type0) {| sized et, has_vec_cpy et|}
  (#rows #cols : erased nat)
  (#lsrc #ldst : mlayout rows cols)
  {| clayout lsrc, strided : strided_row_major lsrc |}
  (src : gpu_matrix et lsrc)
  (dst : gpu_matrix et ldst)
  (i : szlt rows)
  (j : szlt (cols - chunk et + 1))
  (#f : perm)
  (#em : ematrix et rows cols)
  (arr : array et)
  (#s : erased (lseq et (chunk et)))
  preserves gpu
  preserves src |-> Frac f em
  requires
    forall+ (k : natlt (chunk et)).
      gpu_matrix_pts_to_cell dst i (j + k) (s @! k)
  ensures
    forall+ (k : natlt (chunk et)).
      gpu_matrix_pts_to_cell dst i (j + k) (macc em i (j + k))
{
  get_slice src i j;

  strided.pf i j;
  strided.pf i (j + chunk et - 1);

  let offset = strided.offset +^ strided.stride *^ i +^ j;

  // TODO create slice from cells

  gpu_array_vec_cpy_async dst 0sz src 0sz offset;
  admit();

  //unget_slice gm i j;

  //with s. assert pts_to arr s;
  //assert pure (Seq.equal s (Seq.init_ghost (chunk et) (fun x -> macc em i (j + x))));

  ();
}
#pop-options
