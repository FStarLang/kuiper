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
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.Divides
open Kuiper.Matrix.Casts
module SZ = Kuiper.SizeT
open FStar.Tactics.Typeclasses { no_method }

#push-options "--fuel 4 --ifuel 4 --z3rlimit 40"

(* A rank-3 "row-major-like" strided characterization: cell (p,i,j) is an affine
   function offset + pstride*p + rstride*i + j. This lets us slice out a page and
   obtain a rank-2 strided_row_major (vtlayout_of_tlayout layout) at a runtime page. *)
inline_for_extraction noextract
class strided_row_major_3 (#batch #rows #cols : erased nat) (l3 : layout3 batch rows cols) = {
  [@@@no_method]
  offset3 : sz;
  [@@@no_method]
  pstride3 : sz;
  [@@@no_method]
  rstride3 : sz;
  [@@@no_method]
  pf3 : p:natlt batch -> i:natlt rows -> j:natlt cols ->
          squash (l3.imap.f (idx3 p i j)
                    == offset3 + pstride3 * p + rstride3 * i + j);
}

(* Per-page slice: given a rank-3 strided characterization, page p's slice is a
   rank-2 strided_row_major (vtlayout_of_tlayout layout) with offset = offset3 + pstride3*p, stride = rstride3. *)
inline_for_extraction noextract
instance val slice_of_3
  (batch rows cols : erased nat)
  (l3 : layout3 batch rows cols)
  {| s3 : strided_row_major_3 l3 |}
  (page : erased (natlt batch))
  {| cpage : concrete_sz page |}
  (_ : squash (SZ.fits (s3.offset3 + s3.pstride3 * page)))
  : strided_row_major #rows #cols (vtlayout_of_tlayout (tlayout_slice l3 0 page))

val lemma_slice_of_3_offset
  (batch rows cols : erased nat)
  (l3 : layout3 batch rows cols)
  {| s3 : strided_row_major_3 l3 |}
  (page : natlt batch)
  {| cpage : concrete_sz page |}
  (sqf : squash (SZ.fits (s3.offset3 + s3.pstride3 * page)))
  : Lemma (SZ.v (slice_of_3 batch rows cols l3 #s3 page ()).offset
             == s3.offset3 + s3.pstride3 * page)

val lemma_slice_of_3_stride
  (batch rows cols : erased nat)
  (l3 : layout3 batch rows cols)
  {| s3 : strided_row_major_3 l3 |}
  (page : natlt batch)
  {| cpage : concrete_sz page |}
  (sqf : squash (SZ.fits (s3.offset3 + s3.pstride3 * page)))
  : Lemma (SZ.v (slice_of_3 batch rows cols l3 #s3 page ()).stride == SZ.v s3.rstride3)

val lemma_aligned_slice_of_3
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

(* Instance for the concrete rank-3 row-major layout: cell (p,i,j) = p*(rows*cols) + i*cols + j.
   This makes the batched kernel genuinely callable at batch>1. *)
inline_for_extraction noextract
instance val strided_row_major_3_l3_batched
  (batch : erased nat) (rows : erased nat) (cols : erased nat)
  (#_ : squash (cols > 0))
  {| crows : concrete_sz rows, ccols : concrete_sz cols |}
  (sqf : squash (SZ.fits (rows * cols)))
  : strided_row_major_3 (l3_batched_row_major batch rows cols)

(* Instance for the singleton (batch=1) cast, so the rank-2 entry can be derived
   from the batched one at batch=1: cell (0,i,j) == cell_of_pos l i j. *)
inline_for_extraction noextract
instance val strided_row_major_3_l2_to_l3n
  (#a #b : nat) (#l : layout2 a b) {| cl : ctlayout l |}
  {| str : strided_row_major (vtlayout_of_tlayout l) |}
  : strided_row_major_3 (l2_to_l3n #a #b #l)

(* Reveal lemma: the [batch = 1] cast instance's characterization fields are the
   underlying rank-2 stride/offset (with a zero page stride).  The instance is
   [inline_for_extraction noextract], so these projections are opaque to SMT at a
   call site; this lemma exposes them so batch-one alignment obligations reduce
   to the rank-2 [aligned_strided_row_major] hypothesis. *)
val lemma_l2_to_l3n_fields
  (#a #b : nat) (#l : layout2 a b) {| cl : ctlayout l |}
  {| str : strided_row_major (vtlayout_of_tlayout l) |}
  : Lemma ((strided_row_major_3_l2_to_l3n #a #b #l #cl #str).offset3 == str.offset /\
           (strided_row_major_3_l2_to_l3n #a #b #l #cl #str).pstride3 == 0sz /\
           (strided_row_major_3_l2_to_l3n #a #b #l #cl #str).rstride3 == str.stride)
#pop-options
