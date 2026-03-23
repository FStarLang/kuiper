module Kuiper.Matrix.Slice
#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix
open Kuiper.FArray
open Pulse.Lib.Trade
module A = Kuiper.VArray
module SZ = Kuiper.SizeT
module FA = Kuiper.FArray

(* ============ DEFINITIONS ============ *)

let col_farray
  (#et : Type) (#rows #cols : erased nat) (#l : mlayout rows cols)
  (gm : gpu_matrix et l) (j : enatlt cols)
  : farray et (col_flayout l j)
  = FA.from_array (col_flayout l j) (Kuiper.Matrix.core gm)

let row_farray
  (#et : Type) (#rows #cols : erased nat) (#l : mlayout rows cols)
  (gm : gpu_matrix et l) (i : enatlt rows)
  : farray et (row_flayout l i)
  = FA.from_array (row_flayout l i) (Kuiper.Matrix.core gm)

(* ============ CELL EQUIVALENCE ============ *)

let col_cell_eq
  (#et : Type) (#rows #cols : nat) (#l : mlayout rows cols)
  (gm : gpu_matrix et l) (j : natlt cols)
  (i : natlt rows) (f : perm) (v : et)
  : Lemma (
      farray_pts_to_cell (col_farray gm j) #f i v
      ==
      gpu_matrix_pts_to_cell gm #f i j v
    )
  = farray_pts_to_cell_eq (col_farray gm j) i f v;
    gpu_matrix_pts_to_cell_eq gm i j f v

let row_cell_eq
  (#et : Type) (#rows #cols : nat) (#l : mlayout rows cols)
  (gm : gpu_matrix et l) (i : natlt rows)
  (j : natlt cols) (f : perm) (v : et)
  : Lemma (
      farray_pts_to_cell (row_farray gm i) #f j v
      ==
      gpu_matrix_pts_to_cell gm #f i j v
    )
  = farray_pts_to_cell_eq (row_farray gm i) j f v;
    gpu_matrix_pts_to_cell_eq gm i j f v

(* ============ COLUMN EXTRACTION ============ *)

ghost
fn gpu_matrix_extract_col
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (j : natlt cols)
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  ensures
    factored
      (col_farray gm j |-> Frac f (ematrix_col em j))
      (gm |-> Frac f em)
{
  gpu_matrix_pts_to_ref gm;
  gpu_matrix_explode gm;

  (* For each row, separate column j *)
  ghost
  fn extract_col_from_row (i : natlt rows)
    norewrite
    requires
      forall+ (c : natlt cols).
        gpu_matrix_pts_to_cell gm #f i c (macc em i c)
    ensures
      gpu_matrix_pts_to_cell gm #f i j (macc em i j) **
      (forall+ (x : natlt cols { ~(eq2 #(natlt cols) x j) }).
        gpu_matrix_pts_to_cell gm #f i x (macc em i x))
  { forevery_remove _ j };
  forevery_map _ _ extract_col_from_row;
  forevery_unzip
    (fun (i : natlt rows) -> gpu_matrix_pts_to_cell gm #f i j (macc em i j))
    (fun (i : natlt rows) ->
      forall+ (x : natlt cols { ~(eq2 #(natlt cols) x j) }).
        gpu_matrix_pts_to_cell gm #f i x (macc em i x));

  (* Convert matrix cells → farray cells *)
  ghost
  fn to_col_cell (i : natlt rows)
    requires gpu_matrix_pts_to_cell gm #f i j (macc em i j)
    ensures  farray_pts_to_cell (col_farray gm j) #f i (macc em i j)
  {
    col_cell_eq gm j i f (macc em i j);
    rewrite gpu_matrix_pts_to_cell gm #f i j (macc em i j)
         as farray_pts_to_cell (col_farray gm j) #f i (macc em i j);
  };
  forevery_map
    (fun (i : natlt rows) -> gpu_matrix_pts_to_cell gm #f i j (macc em i j))
    (fun (i : natlt rows) -> farray_pts_to_cell (col_farray gm j) #f i (macc em i j))
    to_col_cell;

  (* Implode into farray ownership *)
  forevery_ext
    (fun (i : natlt rows) -> farray_pts_to_cell (col_farray gm j) #f i (macc em i j))
    (fun (i : natlt rows) -> farray_pts_to_cell (col_farray gm j) #f i (Seq.index (ematrix_col em j) i));
  farray_implode (col_farray gm j) #f #(ematrix_col em j);

  (* Build trade *)
  ghost
  fn restore_trade ()
    norewrite
    requires
      forall+ (i : natlt rows) (x : natlt cols { ~(eq2 #(natlt cols) x j) }).
        gpu_matrix_pts_to_cell gm #f i x (macc em i x)
    requires
      col_farray gm j |-> Frac f (ematrix_col em j)
    ensures
      gm |-> Frac f em
  {
    farray_explode (col_farray gm j);

    ghost
    fn from_col_cell (i : natlt rows)
      requires farray_pts_to_cell (col_farray gm j) #f i (Seq.index (ematrix_col em j) i)
      ensures  gpu_matrix_pts_to_cell gm #f i j (macc em i j)
    {
      col_cell_eq gm j i f (macc em i j);
      rewrite farray_pts_to_cell (col_farray gm j) #f i (Seq.index (ematrix_col em j) i)
           as gpu_matrix_pts_to_cell gm #f i j (macc em i j);
    };
    forevery_map _ _ from_col_cell;

    forevery_zip
      (fun (i : natlt rows) -> gpu_matrix_pts_to_cell gm #f i j (macc em i j))
      (fun (i : natlt rows) ->
        forall+ (x : natlt cols { ~(eq2 #(natlt cols) x j) }).
          gpu_matrix_pts_to_cell gm #f i x (macc em i x));

    ghost
    fn insert_col (i : natlt rows)
      norewrite
      requires
        gpu_matrix_pts_to_cell gm #f i j (macc em i j) **
        (forall+ (x : natlt cols { ~(eq2 #(natlt cols) x j) }).
          gpu_matrix_pts_to_cell gm #f i x (macc em i x))
      ensures
        forall+ (c : natlt cols). gpu_matrix_pts_to_cell gm #f i c (macc em i c)
    {
      forevery_insert
        #(natlt cols) #(fun (x : natlt cols) -> ~(eq2 #(natlt cols) x j))
        (fun (c : natlt cols) -> gpu_matrix_pts_to_cell gm #f i c (macc em i c))
        j;
      forevery_unrefine _;
    };
    forevery_map _ _ insert_col;

    gpu_matrix_implode gm;
  };

  fold farray_pts_to (col_farray gm j) #f (ematrix_col em j);
  Pulse.Lib.Trade.intro_trade _ _ _ restore_trade;
}

ghost
fn gpu_matrix_restore_col
  (#et:Type0) (#rows #cols : nat) (#l : mlayout rows cols)
  (gm : gpu_matrix et l) (j : natlt cols)
  (#em : ematrix et rows cols) (#f : perm)
  requires
    factored (col_farray gm j |-> Frac f (ematrix_col em j)) (gm |-> Frac f em)
  ensures
    gm |-> Frac f em
{
  unfold factored _ _;
  ambig_trade_elim ();
}

(* ============ ROW EXTRACTION ============ *)

ghost
fn gpu_matrix_extract_row
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (i : natlt rows)
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  ensures
    factored
      (row_farray gm i |-> Frac f (ematrix_row em i))
      (gm |-> Frac f em)
{
  gpu_matrix_pts_to_ref gm;
  gpu_matrix_explode gm;

  (* Commute to iterate columns-first *)
  forevery_commute
    (fun (r : natlt rows) (c : natlt cols) ->
      gpu_matrix_pts_to_cell gm #f r c (macc em r c));

  (* For each column, separate row i *)
  ghost
  fn extract_row_from_col (j : natlt cols)
    norewrite
    requires
      forall+ (r : natlt rows).
        gpu_matrix_pts_to_cell gm #f r j (macc em r j)
    ensures
      gpu_matrix_pts_to_cell gm #f i j (macc em i j) **
      (forall+ (x : natlt rows { ~(eq2 #(natlt rows) x i) }).
        gpu_matrix_pts_to_cell gm #f x j (macc em x j))
  { forevery_remove _ i };
  forevery_map _ _ extract_row_from_col;
  forevery_unzip
    (fun (j : natlt cols) -> gpu_matrix_pts_to_cell gm #f i j (macc em i j))
    (fun (j : natlt cols) ->
      forall+ (x : natlt rows { ~(eq2 #(natlt rows) x i) }).
        gpu_matrix_pts_to_cell gm #f x j (macc em x j));

  (* Convert matrix cells → farray cells *)
  ghost
  fn to_row_cell (j : natlt cols)
    requires gpu_matrix_pts_to_cell gm #f i j (macc em i j)
    ensures  farray_pts_to_cell (row_farray gm i) #f j (macc em i j)
  {
    row_cell_eq gm i j f (macc em i j);
    rewrite gpu_matrix_pts_to_cell gm #f i j (macc em i j)
         as farray_pts_to_cell (row_farray gm i) #f j (macc em i j);
  };
  forevery_map
    (fun (j : natlt cols) -> gpu_matrix_pts_to_cell gm #f i j (macc em i j))
    (fun (j : natlt cols) -> farray_pts_to_cell (row_farray gm i) #f j (macc em i j))
    to_row_cell;

  (* Implode into farray ownership *)
  forevery_ext
    (fun (j : natlt cols) -> farray_pts_to_cell (row_farray gm i) #f j (macc em i j))
    (fun (j : natlt cols) -> farray_pts_to_cell (row_farray gm i) #f j (Seq.index (ematrix_row em i) j));
  farray_implode (row_farray gm i) #f #(ematrix_row em i);

  (* Build trade *)
  ghost
  fn restore_trade ()
    norewrite
    requires
      forall+ (j : natlt cols) (x : natlt rows { ~(eq2 #(natlt rows) x i) }).
        gpu_matrix_pts_to_cell gm #f x j (macc em x j)
    requires
      row_farray gm i |-> Frac f (ematrix_row em i)
    ensures
      gm |-> Frac f em
  {
    farray_explode (row_farray gm i);

    ghost
    fn from_row_cell (j : natlt cols)
      requires farray_pts_to_cell (row_farray gm i) #f j (Seq.index (ematrix_row em i) j)
      ensures  gpu_matrix_pts_to_cell gm #f i j (macc em i j)
    {
      row_cell_eq gm i j f (macc em i j);
      rewrite farray_pts_to_cell (row_farray gm i) #f j (Seq.index (ematrix_row em i) j)
           as gpu_matrix_pts_to_cell gm #f i j (macc em i j);
    };
    forevery_map _ _ from_row_cell;

    forevery_zip
      (fun (j : natlt cols) -> gpu_matrix_pts_to_cell gm #f i j (macc em i j))
      (fun (j : natlt cols) ->
        forall+ (x : natlt rows { ~(eq2 #(natlt rows) x i) }).
          gpu_matrix_pts_to_cell gm #f x j (macc em x j));

    ghost
    fn insert_row (j : natlt cols)
      norewrite
      requires
        gpu_matrix_pts_to_cell gm #f i j (macc em i j) **
        (forall+ (x : natlt rows { ~(eq2 #(natlt rows) x i) }).
          gpu_matrix_pts_to_cell gm #f x j (macc em x j))
      ensures
        forall+ (r : natlt rows). gpu_matrix_pts_to_cell gm #f r j (macc em r j)
    {
      forevery_insert
        #(natlt rows) #(fun (x : natlt rows) -> ~(eq2 #(natlt rows) x i))
        (fun (r : natlt rows) -> gpu_matrix_pts_to_cell gm #f r j (macc em r j))
        i;
      forevery_unrefine _;
    };
    forevery_map _ _ insert_row;

    forevery_commute
      (fun (c : natlt cols) (r : natlt rows) ->
        gpu_matrix_pts_to_cell gm #f r c (macc em r c));

    gpu_matrix_implode gm;
  };

  fold farray_pts_to (row_farray gm i) #f (ematrix_row em i);
  Pulse.Lib.Trade.intro_trade _ _ _ restore_trade;
}

ghost
fn gpu_matrix_restore_row
  (#et:Type0) (#rows #cols : nat) (#l : mlayout rows cols)
  (gm : gpu_matrix et l) (i : natlt rows)
  (#em : ematrix et rows cols) (#f : perm)
  requires
    factored (row_farray gm i |-> Frac f (ematrix_row em i)) (gm |-> Frac f em)
  ensures
    gm |-> Frac f em
{
  unfold factored _ _;
  ambig_trade_elim ();
}
