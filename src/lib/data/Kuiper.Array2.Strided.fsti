module Kuiper.Array2.Strided

(* Strided layout properties for Array2 (Tensor-backed) layouts. *)

#lang-pulse

open Kuiper
open Kuiper.Injection
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Tiling { subtile_layout }
module SZ = Kuiper.SizeT
open FStar.Tactics.Typeclasses { no_method }

let cell_of_pos (#rows #cols : nat)
  (l : layout2 rows cols) (i : natlt rows) (j : natlt cols) : GTot nat =
  l.imap.f (idx2 i j)

inline_for_extraction noextract
class strided_row_major (#rows #cols : erased nat) (l : layout2 rows cols) = {
  [@@@no_method]
  offset : sz;
  [@@@no_method]
  stride : szp;
  [@@@no_method]
  pf : i:natlt rows -> j:natlt cols ->
         squash (cell_of_pos l i j == offset + stride * i + j);
}

let aligned_strided_row_major
  (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  (n : pos)
  (srm : strided_row_major l)
  : prop =
  n /?+ srm.stride /\ n /?+ srm.offset

inline_for_extraction noextract
class strided_col_major (#rows #cols : erased nat) (l : layout2 rows cols) = {
  [@@@no_method]
  offset : sz;
  [@@@no_method]
  stride : szp;
  [@@@no_method]
  pf : i:natlt rows -> j:natlt cols ->
         squash (cell_of_pos l i j == offset + stride * j + i);
}

let aligned_strided_col_major
  (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  (n : pos)
  (srm : strided_col_major l)
  : prop =
  n /?+ srm.stride /\ n /?+ srm.offset

(* Instance for l2_row_major *)
inline_for_extraction noextract
instance val strided_row_major_l2_row_major (#rows #cols : erased nat)
  (#_ : squash (cols > 0))
  {| d : concrete_sz cols |}
  : strided_row_major (l2_row_major rows cols)

val lemma_aligned_strided_row_major_l2_row_major (#rows #cols : erased nat)
  (#_ : squash (cols > 0))
  {| d : concrete_sz cols |}
  (n : pos)
  : Lemma (requires n /?+ cols)
          (ensures aligned_strided_row_major n (strided_row_major_l2_row_major #rows #cols #_ #d))

(* Instance for l2_col_major *)
inline_for_extraction noextract
instance val strided_col_major_l2_col_major (#rows #cols : erased nat)
  (#_ : squash (rows > 0))
  {| d : concrete_sz rows |}
  : strided_col_major (l2_col_major rows cols)

(* Instance for subtile_layout *)
inline_for_extraction noextract
instance val strided_row_major_subtile (#rows #cols : erased nat)
  (l : layout2 rows cols)
  (#_ : squash (SZ.fits (l.ulen)))
  {| sub : strided_row_major l |}
  (trows : erased int {0 < trows /\ trows /?+ rows})
  (tcols : erased int {0 < tcols /\ tcols /?+ cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : strided_row_major (subtile_layout l trows tcols tr tc)

val lemma_subtile_strided_row_major_offset
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  {| sub : strided_row_major l |}
  (trows : erased int {0 < trows /\ trows /?+ rows})
  (tcols : erased int {0 < tcols /\ tcols /?+ cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : Lemma (requires SZ.fits (l.ulen))
          (ensures
            SZ.v (strided_row_major_subtile l trows tcols tr tc).offset
            ==
            sub.offset + sub.stride * (tr * trows) + tc * tcols)
          [SMTPat (strided_row_major_subtile l trows tcols tr tc).offset]

val lemma_subtile_strided_row_major_stride
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  {| sub : strided_row_major l |}
  (trows : erased int {0 < trows /\ trows /?+ rows})
  (tcols : erased int {0 < tcols /\ tcols /?+ cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : Lemma (requires SZ.fits (l.ulen))
          (ensures
            (strided_row_major_subtile l trows tcols tr tc).stride
            ==
            sub.stride)
          [SMTPat (strided_row_major_subtile l trows tcols tr tc).stride]

(* Instance for subtile_layout of a col-major parent *)
inline_for_extraction noextract
instance val strided_col_major_subtile (#rows #cols : erased nat)
  (l : layout2 rows cols)
  (#_ : squash (SZ.fits (l.ulen)))
  {| sub : strided_col_major l |}
  (trows : erased int {0 < trows /\ trows /?+ rows})
  (tcols : erased int {0 < tcols /\ tcols /?+ cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : strided_col_major (subtile_layout l trows tcols tr tc)

val lemma_subtile_strided_col_major_offset
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  {| sub : strided_col_major l |}
  (trows : erased int {0 < trows /\ trows /?+ rows})
  (tcols : erased int {0 < tcols /\ tcols /?+ cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : Lemma (requires SZ.fits (l.ulen))
          (ensures
            SZ.v (strided_col_major_subtile l trows tcols tr tc).offset
            ==
            sub.offset + sub.stride * (tc * tcols) + tr * trows)
          [SMTPat (strided_col_major_subtile l trows tcols tr tc).offset]

val lemma_subtile_strided_col_major_stride
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  {| sub : strided_col_major l |}
  (trows : erased int {0 < trows /\ trows /?+ rows})
  (tcols : erased int {0 < tcols /\ tcols /?+ cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : Lemma (requires SZ.fits (l.ulen))
          (ensures
            (strided_col_major_subtile l trows tcols tr tc).stride
            ==
            sub.stride)
          [SMTPat (strided_col_major_subtile l trows tcols tr tc).stride]
