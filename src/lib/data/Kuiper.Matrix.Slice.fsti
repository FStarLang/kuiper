module Kuiper.Matrix.Slice
#lang-pulse

(* Extracting rows and columns from a gpu_matrix as farrays. *)

open Kuiper
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix
open Kuiper.FArray
open Pulse.Lib.Trade
module SZ = Kuiper.SizeT

(* ============ COLUMN ============ *)

let col_flayout (#rows #cols : nat) (l : mlayout rows cols) (j : natlt cols)
  : flayout rows
  = {
      flen = l.len;
      fmap = {
        f = (fun (i : natlt rows) -> l.map.f (i, j));
        is_inj = (fun x y -> l.map.is_inj (x, j) (y, j));
      };
    }

inline_for_extraction noextract
instance cflayout_col
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (cl : clayout l)
  (j : szlt cols)
  : cflayout (col_flayout l j)
  = { cf_len = cl.m_len; cf_sz = cl.m_rows; cf_to = (fun i -> cl.c_to i j); }

let ematrix_col (#et:Type) (#rows #cols : nat)
  (em : ematrix et rows cols) (j : natlt cols)
  : GTot (lseq et rows)
  = Seq.init_ghost rows (fun (i : natlt rows) -> macc em i j)

inline_for_extraction noextract
val col_farray
  (#et : Type) (#rows #cols : erased nat) (#l : mlayout rows cols)
  (gm : gpu_matrix et l) (j : enatlt cols)
  : farray et (col_flayout l j)

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

ghost
fn gpu_matrix_restore_col
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (j : natlt cols)
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    factored
      (col_farray gm j |-> Frac f (ematrix_col em j))
      (gm |-> Frac f em)
  ensures
    gm |-> Frac f em

(* ============ ROW ============ *)

let row_flayout (#rows #cols : nat) (l : mlayout rows cols) (i : natlt rows)
  : flayout cols
  = {
      flen = l.len;
      fmap = {
        f = (fun (j : natlt cols) -> l.map.f (i, j));
        is_inj = (fun x y -> l.map.is_inj (i, x) (i, y));
      };
    }

inline_for_extraction noextract
instance cflayout_row
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (cl : clayout l)
  (i : szlt rows)
  : cflayout (row_flayout l i)
  = { cf_len = cl.m_len; cf_sz = cl.m_cols; cf_to = (fun j -> cl.c_to i j); }

let ematrix_row (#et:Type) (#rows #cols : nat)
  (em : ematrix et rows cols) (i : natlt rows)
  : GTot (lseq et cols)
  = Seq.init_ghost cols (fun (j : natlt cols) -> macc em i j)

inline_for_extraction noextract
val row_farray
  (#et : Type) (#rows #cols : erased nat) (#l : mlayout rows cols)
  (gm : gpu_matrix et l) (i : enatlt rows)
  : farray et (row_flayout l i)

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

ghost
fn gpu_matrix_restore_row
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (i : natlt rows)
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    factored
      (row_farray gm i |-> Frac f (ematrix_row em i))
      (gm |-> Frac f em)
  ensures
    gm |-> Frac f em
