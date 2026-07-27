module Kuiper.Array2.Strided.Slice

#lang-pulse

open Kuiper
open Kuiper.Injection
open Kuiper.Shape
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Layout.Slice
open Kuiper.Array2.Strided
open Kuiper.Divides
open Kuiper.Matrix.Casts
module SZ = Kuiper.SizeT
open FStar.Tactics.Typeclasses { no_method }

(* A rank-3 "row-major-like" strided characterization: cell (p,i,j) is an affine
   function offset + pstride*p + rstride*i + j. This lets us slice out a page and
   obtain a rank-2 strided_row_major layout at a runtime page. *)



(* cell reduction: slicing page 0 of dim 0 exposes cell (p,i,j) as l3.imap.f (idx3 p i j) *)
#push-options "--fuel 4 --ifuel 4 --z3rlimit 40"
let slice3_cell_lemma
  (batch rows cols : nat)
  (l3 : layout3 batch rows cols)
  (page : natlt batch)
  (i : natlt rows) (j : natlt cols)
  : Lemma (cell_of_pos (tlayout_slice l3 0 page) i j == l3.imap.f (idx3 page i j))
  = ()
#pop-options

(* Per-page slice: given a rank-3 strided characterization, page p's slice is a
   rank-2 strided_row_major layout with offset = offset3 + pstride3*p, stride = rstride3. *)
#push-options "--fuel 4 --ifuel 4 --z3rlimit 40"
inline_for_extraction noextract
instance slice_of_3
  (batch rows cols : erased nat)
  (l3 : layout3 batch rows cols)
  {| s3 : strided_row_major_3 l3 |}
  (page : erased (natlt batch))
  {| cpage : concrete_sz page |}
  (_ : squash (SZ.fits (s3.offset3 + s3.pstride3 * page)))
  : strided_row_major #rows #cols (tlayout_slice l3 0 page) =
{
  offset = s3.offset3 +^ s3.pstride3 *^ concr' cpage;
  stride = s3.rstride3;
  pf = (fun i j ->
          slice3_cell_lemma batch rows cols l3 page i j;
          s3.pf3 page i j);
}
#pop-options

let lemma_slice_of_3_offset
  (batch rows cols : erased nat)
  (l3 : layout3 batch rows cols)
  {| s3 : strided_row_major_3 l3 |}
  (page : natlt batch)
  {| cpage : concrete_sz page |}
  (sqf : squash (SZ.fits (s3.offset3 + s3.pstride3 * page)))
  : Lemma (SZ.v (slice_of_3 batch rows cols l3 #s3 page ()).offset
             == s3.offset3 + s3.pstride3 * page)
  = ()

let lemma_slice_of_3_stride
  (batch rows cols : erased nat)
  (l3 : layout3 batch rows cols)
  {| s3 : strided_row_major_3 l3 |}
  (page : natlt batch)
  {| cpage : concrete_sz page |}
  (sqf : squash (SZ.fits (s3.offset3 + s3.pstride3 * page)))
  : Lemma (SZ.v (slice_of_3 batch rows cols l3 #s3 page ()).stride == SZ.v s3.rstride3)
  = ()

let lemma_aligned_slice_of_3
  (batch rows cols : erased nat)
  (l3 : layout3 batch rows cols)
  {| s3 : strided_row_major_3 l3 |}
  (page : natlt batch)
  {| cpage : concrete_sz page |}
  (#sqf : squash (SZ.fits (s3.offset3 + s3.pstride3 * page)))
  (k : pos)
  : Lemma (requires k /?+ s3.rstride3 /\ k /?+ s3.offset3 /\ k /?+ s3.pstride3)
          (ensures aligned_strided_row_major k
                     (slice_of_3 batch rows cols l3 #s3 page ()))
  = lemma_slice_of_3_offset batch rows cols l3 #s3 page #cpage sqf;
    lemma_slice_of_3_stride batch rows cols l3 #s3 page #cpage sqf;
    lemma_divides_product_l k s3.pstride3 page;
    lemma_divides_sum k s3.offset3 (s3.pstride3 * page)

(* Instance for the concrete rank-3 row-major layout: cell (p,i,j) = p*(rows*cols) + i*cols + j.
   This makes the batched kernel genuinely callable at batch>1. *)
#push-options "--fuel 4 --ifuel 4 --z3rlimit 40"
let l3_batched_cell_lemma
  (batch rows cols : nat)
  (p : natlt batch) (i : natlt rows) (j : natlt cols)
  : Lemma ((l3_batched_row_major batch rows cols).imap.f (idx3 p i j)
             == p * (rows * cols) + cols * i + j)
  = ()
#pop-options

#push-options "--fuel 4 --ifuel 4 --z3rlimit 40"
instance strided_row_major_3_l3_batched
  (batch : erased nat) (rows : erased nat) (cols : erased nat)
  (#_ : squash (cols > 0))
  {| crows : concrete_sz rows, ccols : concrete_sz cols |}
  (sqf : squash (SZ.fits (rows * cols)))
  : strided_row_major_3 (l3_batched_row_major batch rows cols) =
{
  offset3 = 0sz;
  pstride3 = concr' crows *^ concr' ccols;
  rstride3 = concr' ccols;
  pf3 = (fun p i j -> l3_batched_cell_lemma batch rows cols p i j);
}
#pop-options

(* Instance for the singleton (batch=1) cast, so the rank-2 entry can be derived
   from the batched one at batch=1: cell (0,i,j) == cell_of_pos l i j. *)
#push-options "--fuel 4 --ifuel 4 --z3rlimit 60"
let l2_to_l3n_page0_bridge
  (#a #b : nat) (l : layout2 a b) {| cl : ctlayout l |}
  (i : natlt a) (j : natlt b)
  : Lemma ((l2_to_l3n #a #b #l).imap.f (idx3 0 i j) == cell_of_pos l i j)
  = ()
#pop-options

#push-options "--fuel 4 --ifuel 4 --z3rlimit 60"
inline_for_extraction noextract
instance strided_row_major_3_l2_to_l3n
  (#a #b : nat) (#l : layout2 a b) {| cl : ctlayout l |}
  {| str : strided_row_major l |}
  : strided_row_major_3 (l2_to_l3n #a #b #l) =
{
  offset3 = str.offset;
  pstride3 = 0sz;
  rstride3 = str.stride;
  pf3 = (fun p i j -> l2_to_l3n_page0_bridge l i j; str.pf i j);
}
#pop-options

#push-options "--fuel 4 --ifuel 4 --z3rlimit 60"
let lemma_l2_to_l3n_fields
  (#a #b : nat) (#l : layout2 a b) {| cl : ctlayout l |}
  {| str : strided_row_major l |}
  : Lemma ((strided_row_major_3_l2_to_l3n #a #b #l #cl #str).offset3 == str.offset /\
           (strided_row_major_3_l2_to_l3n #a #b #l #cl #str).pstride3 == 0sz /\
           (strided_row_major_3_l2_to_l3n #a #b #l #cl #str).rstride3 == str.stride)
  = ()
#pop-options
