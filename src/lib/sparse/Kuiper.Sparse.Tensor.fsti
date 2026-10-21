module Kuiper.Sparse.Tensor

#lang-pulse

open Kuiper
open Kuiper.Sparse.Common
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Array2.Strided { strided_row_major, aligned_strided_row_major }
open FStar.Tactics.Typeclasses { no_method }
open Kuiper.Array.Vectorized
open Kuiper.Seq.Common { (@!) }
open Kuiper.Tensor.Layout.Slice
open Kuiper.Array2.Strided { cell_of_pos }

module SZ = FStar.SizeT
module Chest = Kuiper.Chest

val aligned_cell_strided_row_major
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos { chunk et /? cols })
  (#l : layout2 rows cols) {| strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : natlt cols { chunk et /? j })
: Lemma
  (requires
    aligned 16 (core gm) /\
    aligned_strided_row_major (chunk et) strided
  )
  (ensures aligned' 16 (core gm) (cell_of_pos l i j))

let array_cell_of_pos (#n : nat)
  (l : layout1 n) (i : natlt n) : GTot (natlt (tlayout_ulen l)) =
  l.imap.f (i, ())

inline_for_extraction noextract
class cont_layout (#n : erased nat) (l : layout1 n) = {
  [@@@no_method]
  offset : sz;
  [@@@no_method]
  pf : i:natlt n -> squash (array_cell_of_pos l i == offset + i);
}

let aligned_cont_layout
  (#n : erased nat)
  (#l : layout1 n)
  (k : pos)
  (cl : cont_layout l)
: prop = k /?+ cl.offset

inline_for_extraction noextract
instance val cont_layout_l1_forward (#n : erased nat) : cont_layout (l1_forward n)

inline_for_extraction noextract
instance val cont_layout_strided_row_major
  (#rows #cols : erased nat)
  (l : layout2 rows cols { 0 < cols }) {| srm : strided_row_major l |}
  (#_: squash (fits (tlayout_ulen l)))
  (i : natlt rows)
  {| conc_i : concrete_sz i |}
  : cont_layout #cols (tlayout_slice l 0 i)

val aligned_cont_strided_row_major
  (#rows #cols : erased nat { 0 < cols })
  (l : layout2 rows cols) {| srm : strided_row_major l |}
  (#_: squash (fits (tlayout_ulen l)))
  (k : pos)
  (i : szlt rows)
: Lemma
  (requires aligned_strided_row_major k srm)
  (ensures aligned_cont_layout k (cont_layout_strided_row_major l i))

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

val aligned_cont_offset
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

open Pulse.Lib.Trade { (@==>) }

noextract
val chest2_upd_row
  (#et : Type0)
  (#rows #cols : erased nat)
  (em : chest2 et rows cols)
  (i : natlt rows)
  (new_row : chest1 et cols)
  : chest2 et rows cols

inline_for_extraction noextract
let tensor_row
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  (a : array2 et l)
  (i : erased nat { i < rows })
  : array1 et #cols (tlayout_slice l 0 i)
= sliceof a 0 i

ghost
fn tensor_extract_row
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (a : array2 et l)
  (i : natlt rows)
  (#f : perm)
  (#s : chest2 et rows cols)
  requires
    a |-> Frac f s
  ensures
    tensor_row a i |-> Frac f (chest2_row s i) **
    (forall* (s' : chest1 et cols).
      tensor_row a i |-> Frac f s' @==>
      a |-> Frac f (chest2_upd_row s i s'))

ghost
fn tensor_extract_row_ro
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (a : array2 et l)
  (i : natlt rows)
  (#f : perm)
  (#s : chest2 et rows cols)
  requires
    a |-> Frac f s
  ensures
    factored
      (tensor_row a i |-> Frac f (chest2_row s i))
      (a |-> Frac f s)

ghost
fn tensor_restore_row
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (a : array2 et l)
  (i : natlt rows)
  (#f : perm)
  (#s : chest2 et rows cols)
  requires
    factored
      (tensor_row a i |-> Frac f (chest_slice 0 i s))
      (a |-> Frac f s)
  ensures
    a |-> Frac f s
