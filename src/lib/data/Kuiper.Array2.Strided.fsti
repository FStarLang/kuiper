module Kuiper.Array2.Strided

(* Strided layout properties for Array2 (Tensor-backed) layouts. *)

#lang-pulse

open Kuiper
open Kuiper.Injection
open Kuiper.Tensor
open Kuiper.TensorRO { vlayout1, vlayout2, vtlayout_of_tlayout, extended_layout }
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Tiling { subtile_layout }
module SZ = Kuiper.SizeT
open FStar.Tactics.Typeclasses { no_method }

(* Stridedness is a property of the index *map* alone, so it is stated over the
   general (possibly non-injective) [vlayout2].  A broadcast row vector, for
   instance, is row-major with [stride == 0], which is exactly what lets the
   GEMM epilogue read a bias through the same vectorized path as a full C
   matrix.  Every writable [layout2] is such a map: clients holding one pass
   [vtlayout_of_tlayout l]. *)

let vcell_of_pos (#rows #cols : nat)
  (l : vlayout2 rows cols) (i : natlt rows) (j : natlt cols) : GTot nat =
  l.imap (idx2 i j)

unfold
let cell_of_pos (#rows #cols : nat)
  (l : layout2 rows cols) (i : natlt rows) (j : natlt cols) : GTot nat =
  vcell_of_pos (vtlayout_of_tlayout l) i j

inline_for_extraction noextract
class strided_row_major (#rows #cols : erased nat) (l : vlayout2 rows cols) = {
  [@@@no_method]
  offset : sz;
  [@@@no_method]
  stride : sz;
  [@@@no_method]
  pf : i:natlt rows -> j:natlt cols ->
         squash (vcell_of_pos l i j == offset + stride * i + j);
}

let aligned_strided_row_major
  (#rows #cols : erased nat)
  (#l : vlayout2 rows cols)
  (n : pos)
  (srm : strided_row_major l)
  : prop =
  n /?+ srm.stride /\ n /?+ srm.offset

inline_for_extraction noextract
class strided_col_major (#rows #cols : erased nat) (l : vlayout2 rows cols) = {
  [@@@no_method]
  offset : sz;
  [@@@no_method]
  stride : sz;
  [@@@no_method]
  pf : i:natlt rows -> j:natlt cols ->
         squash (vcell_of_pos l i j == offset + stride * j + i);
}

let aligned_strided_col_major
  (#rows #cols : erased nat)
  (#l : vlayout2 rows cols)
  (n : pos)
  (srm : strided_col_major l)
  : prop =
  n /?+ srm.stride /\ n /?+ srm.offset


(* Instance for l2_row_major *)
inline_for_extraction noextract
instance val strided_row_major_l2_row_major (#rows #cols : erased nat)
  (#_ : squash (cols > 0))
  {| d : concrete_sz cols |}
  : strided_row_major (vtlayout_of_tlayout (l2_row_major rows cols))

val lemma_aligned_strided_row_major_l2_row_major (#rows #cols : erased nat)
  (#_ : squash (cols > 0))
  {| d : concrete_sz cols |}
  (n : pos)
  : Lemma (requires n /?+ cols)
          (ensures aligned_strided_row_major n (strided_row_major_l2_row_major #rows #cols #_ #d))

(* Broadcast of a 1-D row vector along [rows].  [extended_layout l rows] maps
   (i, j) to [l.imap j], so it is row-major with [stride == 0]: every row reads
   the same [cols]-long run starting at [off]. *)
inline_for_extraction noextract
val mk_strided_row_major_bcast
  (#rows #cols : erased nat)
  (l : vlayout1 cols)
  (off : sz)
  (pf : squash (forall (i:natlt rows) (j:natlt cols).
                  vcell_of_pos (extended_layout l rows) i j == SZ.v off + j))
  : strided_row_major (extended_layout l rows)

val lemma_aligned_mk_strided_row_major_bcast
  (#rows #cols : erased nat)
  (l : vlayout1 cols) (off : sz) (pf : squash _) (n : pos)
  : Lemma (requires n /?+ SZ.v off)
          (ensures aligned_strided_row_major n
                     (mk_strided_row_major_bcast #rows #cols l off pf))

(* Instance for l2_col_major *)
inline_for_extraction noextract
instance val strided_col_major_l2_col_major (#rows #cols : erased nat)
  (#_ : squash (rows > 0))
  {| d : concrete_sz rows |}
  : strided_col_major (vtlayout_of_tlayout (l2_col_major rows cols))

(* Instance for subtile_layout *)
inline_for_extraction noextract
instance val strided_row_major_subtile (#rows #cols : erased nat)
  (l : layout2 rows cols)
  (#_ : squash (SZ.fits (l.ulen)))
  {| sub : strided_row_major (vtlayout_of_tlayout l) |}
  (trows : erased int {0 < trows /\ trows /?+ rows})
  (tcols : erased int {0 < tcols /\ tcols /?+ cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : strided_row_major (vtlayout_of_tlayout (subtile_layout l trows tcols tr tc))

val lemma_subtile_strided_row_major_offset
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  {| sub : strided_row_major (vtlayout_of_tlayout l) |}
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
  {| sub : strided_row_major (vtlayout_of_tlayout l) |}
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
  {| sub : strided_col_major (vtlayout_of_tlayout l) |}
  (trows : erased int {0 < trows /\ trows /?+ rows})
  (tcols : erased int {0 < tcols /\ tcols /?+ cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : strided_col_major (vtlayout_of_tlayout (subtile_layout l trows tcols tr tc))

val lemma_subtile_strided_col_major_offset
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  {| sub : strided_col_major (vtlayout_of_tlayout l) |}
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
  {| sub : strided_col_major (vtlayout_of_tlayout l) |}
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
