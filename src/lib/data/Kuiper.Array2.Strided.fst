module Kuiper.Array2.Strided

#lang-pulse

open Kuiper
open Kuiper.Injection
open Kuiper.Array2 { array2, layout, full_layout, layout_size, adapt_idx_back }
open Kuiper.Tensor.Layout.Alg
module SZ = Kuiper.SizeT
open Kuiper.Tensor.Tiling { subtile_layout, tile_inj }

(* Instance for l2_row_major: cell_of_pos = i * cols + j *)
#push-options "--z3rlimit_factor 4"
inline_for_extraction noextract
instance strided_row_major_l2_row_major (#rows #cols : erased nat)
  (#_ : squash (cols > 0))
  {| d : concrete_sz cols |}
  : strided_row_major (l2_row_major rows cols) =
{
  offset = 0sz;
  stride = concr' d;
  pf = ez;
}

inline_for_extraction noextract
instance strided_col_major_l2_col_major (#rows #cols : erased nat)
  (#_ : squash (rows > 0))
  {| d : concrete_sz rows |}
  : strided_col_major (l2_col_major rows cols) =
{
  offset = 0sz;
  stride = concr' d;
  pf = ez;
}
#pop-options

#push-options "--z3rlimit_factor 4 --fuel 0 --ifuel 0"
inline_for_extraction noextract
let strided_row_major_subtile_offset
  (#rows #cols : erased nat)
  (l : layout rows cols)
  (#_ : squash (SZ.fits (layout_size l)))
  {| sub : strided_row_major l |}
  (trows : sz {trows > 0 /\ trows /?+ rows})
  (tcols : sz {tcols > 0 /\ tcols /?+ cols})
  (tr    : szlt (rows / trows))
  (tc    : szlt (cols / tcols))
  : res : sz { SZ.v res == sub.offset + sub.stride * (tr * trows) + tc * tcols }
  = sub.pf (tr * trows) (tc * tcols);
    assert (cell_of_pos l (tr * trows) (tc * tcols) == sub.offset + sub.stride * (tr * trows) + tc * tcols);
    sub.offset +^ sub.stride *^ (tr *^ trows) +^ tc *^ tcols
#pop-options

#push-options "--z3rlimit_factor 4 --fuel 0 --ifuel 0"
let strided_row_major_subtile_proof
  (#rows #cols : nat)
  (l : layout rows cols)
  {| sub : strided_row_major l |}
  (trows : nat {trows > 0 /\ trows /?+ rows})
  (tcols : nat {tcols > 0 /\ tcols /?+ cols})
  (tr    : natlt (rows / trows))
  (tc    : natlt (cols / tcols))
  (i : natlt trows)
  (j : natlt tcols)
  : Lemma (
      cell_of_pos (subtile_layout l trows tcols tr tc) i j ==
      sub.offset + sub.stride * (tr * trows) + tc * tcols
        + sub.stride * i + j
    )
=
  calc (==) {
    cell_of_pos (subtile_layout l trows tcols tr tc) i j <: int;
    == {}
    cell_of_pos l (tr * trows + i) (tc * tcols + j);
    == { sub.pf (tr * trows + i) (tc * tcols + j) }
    sub.offset + sub.stride * (tr * trows + i) + tc * tcols + j;
    == { FStar.Math.Lemmas.distributivity_add_right sub.stride (tr * trows) i }
    sub.offset + sub.stride * (tr * trows) + sub.stride * i + tc * tcols + j;
    == {}
      sub.offset + sub.stride * (tr * trows) + tc * tcols
        + sub.stride * i + j;
  };
  ()
#pop-options

inline_for_extraction noextract
instance strided_row_major_subtile (#rows #cols : erased nat)
  (l : layout rows cols)
  (#_ : squash (SZ.fits (layout_size l)))
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
  : strided_row_major (subtile_layout l trows tcols tr tc) =
{
  offset = strided_row_major_subtile_offset l (concr' c_trows) (concr' c_tcols) (concr' c_tr) (concr' c_tc);
  stride = sub.stride;
  pf = (fun i j -> strided_row_major_subtile_proof #rows #cols l trows tcols tr tc i j);
}

let lemma_subtile_strided_row_major_offset
  (#rows #cols : erased nat)
  (l : layout rows cols)
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
  : Lemma (requires SZ.fits (layout_size l))
          (ensures
            SZ.v (strided_row_major_subtile l trows tcols tr tc).offset
            ==
            sub.offset + sub.stride * (tr * trows) + tc * tcols)
          [SMTPat (strided_row_major_subtile l trows tcols tr tc).offset]
  = ()

let lemma_subtile_strided_row_major_stride
  (#rows #cols : erased nat)
  (l : layout rows cols)
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
  : Lemma (requires SZ.fits (layout_size l))
          (ensures
            (strided_row_major_subtile l trows tcols tr tc).stride
            ==
            sub.stride)
          [SMTPat (strided_row_major_subtile l trows tcols tr tc).stride]
  = ()
