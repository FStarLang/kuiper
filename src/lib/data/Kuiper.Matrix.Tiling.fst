module Kuiper.Matrix.Tiling
#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix
open Kuiper.Injection
module SZ = Kuiper.SizeT

#push-options "--fuel 0 --ifuel 0 --split_queries always"
inline_for_extraction noextract
let strided_row_major_subtile_offset
  (#rows #cols : erased nat)
  (l : mlayout rows cols)
  (#_ : squash (SZ.fits (mlayout_size l)))
  {| sub : strided_row_major l |}
  (trows : sz {trows > 0 /\ trows /? rows})
  (tcols : sz {tcols > 0 /\ tcols /? cols})
  (tr    : szlt (rows / trows))
  (tc    : szlt (cols / tcols))
  : res : sz { SZ.v res == sub.offset + sub.stride * (tr * trows) + tc * tcols }
  = sub.pf (tr * trows) (tc * tcols);
    assert (l.map.f (tr * trows, tc * tcols) == sub.offset + sub.stride * (tr * trows) + tc * tcols);
    sub.offset +^ sub.stride *^ (tr *^ trows) +^ tc *^ tcols
#pop-options

#push-options "--z3rlimit_factor 4 --fuel 0 --ifuel 0"
let strided_row_major_subtile_proof
  (#rows #cols : nat)
  (l : mlayout rows cols)
  {| sub : strided_row_major l |}
  (trows : nat {trows > 0 /\ trows /? rows})
  (tcols : nat {tcols > 0 /\ tcols /? cols})
  (tr    : natlt (rows / trows))
  (tc    : natlt (cols / tcols))
  (i : natlt trows)
  (j : natlt tcols)
  : Lemma (
      (subtile_layout l trows tcols tr tc).map.f (i, j) ==
      sub.offset + sub.stride * (tr * trows) + tc * tcols
        + sub.stride * i + j
    )
=
  calc (==) {
    (subtile_layout l trows tcols tr tc).map.f (i, j) <: int;
    == {}
    l.map.f (tr * trows + i, tc * tcols + j);
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
  (l : mlayout rows cols)
  (#_ : squash (SZ.fits (mlayout_size l)))
  {| sub : strided_row_major l |}
  (trows : erased int {0 < trows /\ trows /? rows})
  (tcols : erased int {0 < tcols /\ tcols /? cols})
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
  (l : mlayout rows cols)
  {| sub : strided_row_major l |}
  (trows : erased int {0 < trows /\ trows /? rows})
  (tcols : erased int {0 < tcols /\ tcols /? cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : Lemma (requires SZ.fits (mlayout_size l))
          (ensures
            SZ.v (strided_row_major_subtile l trows tcols tr tc).offset
            ==
            sub.offset + sub.stride * (tr * trows) + tc * tcols)
          [SMTPat (strided_row_major_subtile l trows tcols tr tc).offset]
  = ()

let lemma_subtile_strided_row_major_stride
  (#rows #cols : erased nat)
  (l : mlayout rows cols)
  {| sub : strided_row_major l |}
  (trows : erased int {0 < trows /\ trows /? rows})
  (tcols : erased int {0 < tcols /\ tcols /? cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : Lemma (requires SZ.fits (mlayout_size l))
          (ensures
            (strided_row_major_subtile l trows tcols tr tc).stride
            ==
            sub.stride)
          [SMTPat (strided_row_major_subtile l trows tcols tr tc).stride]
  = ()

#push-options "--z3rlimit_factor 4 --fuel 0 --ifuel 0"
inline_for_extraction noextract
let strided_col_major_subtile_offset
  (#rows #cols : erased nat)
  (l : mlayout rows cols)
  (#_ : squash (SZ.fits (mlayout_size l)))
  {| sub : strided_col_major l |}
  (trows : sz {trows > 0 /\ trows /? rows})
  (tcols : sz {tcols > 0 /\ tcols /? cols})
  (tr    : szlt (rows / trows))
  (tc    : szlt (cols / tcols))
  : res : sz { SZ.v res == sub.offset + sub.stride * (tc * tcols) + tr * trows }
  = sub.pf (tr * trows) (tc * tcols);
    assert (l.map.f (tr * trows, tc * tcols) == sub.offset + sub.stride * (tc * tcols) + tr * trows);
    sub.offset +^ sub.stride *^ (tc *^ tcols) +^ tr *^ trows

let strided_col_major_subtile_proof
  (#rows #cols : nat)
  (l : mlayout rows cols)
  {| sub : strided_col_major l |}
  (trows : nat {trows > 0 /\ trows /? rows})
  (tcols : nat {tcols > 0 /\ tcols /? cols})
  (tr    : natlt (rows / trows))
  (tc    : natlt (cols / tcols))
  (i : natlt trows)
  (j : natlt tcols)
  : Lemma (
      (subtile_layout l trows tcols tr tc).map.f (i, j) ==
      sub.offset + sub.stride * (tc * tcols) + tr * trows
        + sub.stride * j + i
    )
=
  calc (==) {
    (subtile_layout l trows tcols tr tc).map.f (i, j) <: int;
    == {}
    l.map.f (tr * trows + i, tc * tcols + j);
    == { sub.pf (tr * trows + i) (tc * tcols + j) }
    sub.offset + sub.stride * (tc * tcols + j) + tr * trows + i;
    == { FStar.Math.Lemmas.distributivity_add_right sub.stride (tc * tcols) j }
    sub.offset + sub.stride * (tc * tcols) + sub.stride * j + tr * trows + i;
    == {}
      sub.offset + sub.stride * (tc * tcols) + tr * trows
        + sub.stride * j + i;
  };
  ()
#pop-options

inline_for_extraction noextract
instance strided_col_major_subtile (#rows #cols : erased nat)
  (l : mlayout rows cols)
  (#_ : squash (SZ.fits (mlayout_size l)))
  {| sub : strided_col_major l |}
  (trows : erased int {0 < trows /\ trows /? rows})
  (tcols : erased int {0 < tcols /\ tcols /? cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : strided_col_major (subtile_layout l trows tcols tr tc) =
{
  offset = strided_col_major_subtile_offset l (concr' c_trows) (concr' c_tcols) (concr' c_tr) (concr' c_tc);
  stride = sub.stride;
  pf = (fun i j -> strided_col_major_subtile_proof #rows #cols l trows tcols tr tc i j);
}

let lemma_subtile_strided_col_major_offset
  (#rows #cols : erased nat)
  (l : mlayout rows cols)
  {| sub : strided_col_major l |}
  (trows : erased int {0 < trows /\ trows /? rows})
  (tcols : erased int {0 < tcols /\ tcols /? cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : Lemma (requires SZ.fits (mlayout_size l))
          (ensures
            SZ.v (strided_col_major_subtile l trows tcols tr tc).offset
            ==
            sub.offset + sub.stride * (tc * tcols) + tr * trows)
          [SMTPat (strided_col_major_subtile l trows tcols tr tc).offset]
  = ()

let lemma_subtile_strided_col_major_stride
  (#rows #cols : erased nat)
  (l : mlayout rows cols)
  {| sub : strided_col_major l |}
  (trows : erased int {0 < trows /\ trows /? rows})
  (tcols : erased int {0 < tcols /\ tcols /? cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : Lemma (requires SZ.fits (mlayout_size l))
          (ensures
            (strided_col_major_subtile l trows tcols tr tc).stride
            ==
            sub.stride)
          [SMTPat (strided_col_major_subtile l trows tcols tr tc).stride]
  = ()

inline_for_extraction noextract
instance c_subtile_layout
  (#rows #cols : erased nat)
  (l : mlayout rows cols) {| c : clayout l |}
  (trows : erased int {0 < trows /\ trows /? rows})
  (tcols : erased int {0 < tcols /\ tcols /? cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : clayout (subtile_layout l trows tcols tr tc) =
  {
    m_len = c.m_len;

    c_to = (fun i j -> c.c_to (concr' c_tr *^ concr' c_trows +^ i) (concr' c_tc *^ concr' c_tcols +^ j));

    // Do we actually use these? Try replacing them by magics and
    // see if anything breaks.
    m_rows = concr' c_trows;
    m_cols = concr' c_tcols;
  }

(* Just a cast *)
inline_for_extraction noextract
let gpu_matrix_subtile
  (#et : _)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : erased nat {trows > 0 /\ trows /? rows})
  (tcols : erased nat {tcols > 0 /\ tcols /? cols})
  (tr : (enatlt (rows / trows)))
  (tc : (enatlt (cols / tcols)))
  : Tot (gpu_matrix et (subtile_layout l trows tcols tr tc))
  = from_array (subtile_layout l trows tcols tr tc)
               (core gm)

let gpu_matrix_subtile_base
  (#et : _)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : erased nat {trows > 0 /\ trows /? rows})
  (tcols : erased nat {tcols > 0 /\ tcols /? cols})
  (tr : enatlt (rows / trows))
  (tc : enatlt (cols / tcols))
  : Lemma (
      core (gpu_matrix_subtile gm trows tcols tr tc)
      ==
      core gm
    )
    [SMTPat (core (gpu_matrix_subtile gm trows tcols tr tc))]
  = ()

let cell_convert_eq
  (#et : _)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : erased nat {trows > 0 /\ trows /? rows})
  (tcols : erased nat {tcols > 0 /\ tcols /? cols})
  (tr : (enatlt (rows / trows)))
  (tc : (enatlt (cols / tcols)))
  (i : natlt trows)
  (j : natlt tcols)
  (f : perm)
  (v : et)
: Lemma (
  gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) #f i j v
  ==
  gpu_matrix_pts_to_cell gm #f (tr * trows + i) (tc * tcols + j) v
)
= gpu_matrix_pts_to_cell_eq (gpu_matrix_subtile gm trows tcols tr tc) i j f v;
  gpu_matrix_pts_to_cell_eq gm (tr * trows + i) (tc * tcols + j) f v

ghost
fn subcell_to_cell
  (#et : _)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : erased nat {trows > 0 /\ trows /? rows})
  (tcols : erased nat {tcols > 0 /\ tcols /? cols})
  (tr : (enatlt (rows / trows)))
  (tc : (enatlt (cols / tcols)))
  (i : natlt trows)
  (j : natlt tcols)
  (#f : perm)
  (#v : et)
  requires
    gpu_matrix_pts_to_cell gm #f (tr * trows + i) (tc * tcols + j) v
  ensures
    gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) #f i j v
{
  cell_convert_eq gm trows tcols tr tc i j f v;
  rewrite
    gpu_matrix_pts_to_cell gm #f (tr * trows + i) (tc * tcols + j) v
       as
    gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) #f i j v;
}

ghost
fn cell_to_subcell
  (#et : _)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : erased nat {trows > 0 /\ trows /? rows})
  (tcols : erased nat {tcols > 0 /\ tcols /? cols})
  (tr : (enatlt (rows / trows)))
  (tc : (enatlt (cols / tcols)))
  (i : natlt trows)
  (j : natlt tcols)
  (#f : perm)
  (#v : et)
  requires
    gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) #f i j v
  ensures
    gpu_matrix_pts_to_cell gm #f (tr * trows + i) (tc * tcols + j) v
{
  cell_convert_eq gm trows tcols tr tc i j f v;
  rewrite
    gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) #f i j v
       as
    gpu_matrix_pts_to_cell gm #f (tr * trows + i) (tc * tcols + j) v;
}

ghost
fn gpu_matrix_tile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  ensures
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
        gpu_matrix_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc)
{
  gpu_matrix_iconcr gm;
  forevery_factor_2 rows (rows / trows) trows
    cols (cols / tcols) tcols
    _;
  forevery_mid_flip _;
  ghost
  fn aux (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
    requires
      forall+ (i : natlt trows) (j : natlt tcols).
        gpu_pts_to_cell (core gm) #f (cell_of_pos l (tr * trows + i) (tc * tcols + j)) (macc em (tr * trows + i) (tc * tcols + j))
    ensures
      gpu_matrix_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc)
  {
    forevery_map_2 #(natlt trows) #(natlt tcols)
      (fun i j -> gpu_pts_to_cell (core gm) #f (cell_of_pos l (tr * trows + i) (tc * tcols + j)) (macc em (tr * trows + i) (tc * tcols + j)))
      (fun i j -> gpu_pts_to_cell (core (gpu_matrix_subtile gm trows tcols tr tc))
                                  #f
                                  (cell_of_pos (subtile_layout l trows tcols tr tc) i j)
                                  (macc (ematrix_subtile em trows tcols tr tc) i j))
      fn i j { rewrite each core gm as core (gpu_matrix_subtile gm trows tcols tr tc); };
    gpu_matrix_iabs (gpu_matrix_subtile gm trows tcols tr tc);
    ();
  };
  forevery_map_2 _ _ aux;
  ()
}

ghost
fn gpu_matrix_untile'
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (tf : natlt (rows / trows) -> natlt (cols / tcols) -> ematrix et trows tcols)
  (#f : perm)
  requires
    pure (SZ.fits (mlayout_size l))
  requires
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
      (gpu_matrix_subtile gm trows tcols tr tc |-> Frac f (tf tr tc))
  ensures
    gm |-> Frac f (ematrix_from_tiles trows tcols tf)
{
  let em = ematrix_from_tiles trows tcols tf;
  ghost
  fn aux (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
    requires
      gpu_matrix_subtile gm trows tcols tr tc |-> Frac f (tf tr tc)
    ensures
      forall+ (i : natlt trows) (j : natlt tcols).
        gpu_pts_to_cell (core gm) #f (cell_of_pos l (tr * trows + i) (tc * tcols + j)) (macc em (tr * trows + i) (tc * tcols + j))
  {
    gpu_matrix_iconcr (gpu_matrix_subtile gm trows tcols tr tc);
    forevery_map_2 #(natlt trows) #(natlt tcols)
      (fun i j -> gpu_pts_to_cell (core (gpu_matrix_subtile gm trows tcols tr tc))
                                  #f
                                  (cell_of_pos (subtile_layout l trows tcols tr tc) i j)
                                  (macc (tf tr tc) i j))
      (fun i j -> gpu_pts_to_cell (core gm) #f (cell_of_pos l (tr * trows + i) (tc * tcols + j)) (macc em (tr * trows + i) (tc * tcols + j)))
      fn i j {
        rewrite each core (gpu_matrix_subtile gm trows tcols tr tc) as core gm;
        (* Help SMT *)
        assert pure ((tr * trows + i) / trows == tr);
        assert pure ((tc * tcols + j) / tcols == tc);
        assert pure ((tr * trows + i) % trows == i);
        assert pure ((tc * tcols + j) % tcols == j);
        assert pure (macc em (tr * trows + i) (tc * tcols + j) == macc (tf tr tc) i j);
      };
    ();
  };
  forevery_map_2 _ _ aux;
  forevery_mid_flip _;
  forevery_unfactor_2 rows (rows / trows) trows
    cols (cols / tcols) tcols
    (fun i j -> gpu_pts_to_cell (core gm) #f (cell_of_pos l i j) (macc em i j));
  gpu_matrix_iabs gm;
  ()
}

ghost
fn gpu_matrix_untile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    pure (SZ.fits (mlayout_size l))
  requires
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
        gpu_matrix_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc)
  ensures
    gm |-> Frac f em
{
  gpu_matrix_untile' gm trows tcols _;
  from_subtiles_id em trows tcols;
  rewrite each ematrix_from_tiles trows tcols (ematrix_subtile em trows tcols)
            as em;
  ();
}

ghost
fn gpu_matrix_untile_underspec
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (#f : perm)
  requires
    pure (SZ.fits (mlayout_size l))
  requires
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
        (exists* (em : ematrix et trows tcols).
          gpu_matrix_subtile gm trows tcols tr tc |-> Frac f em)
  ensures
    exists* (em : ematrix et rows cols). gm |-> Frac f em
{
  forevery_flatten _;
  let cf = forevery_exists #(natlt (rows / trows) & natlt (cols / tcols)) _;
  let em = ematrix_from_tiles trows tcols (fun x y -> cf (x,y));
  ghost
  fn aux (rc : natlt (rows / trows) & natlt (cols / tcols))
    requires
      gpu_matrix_subtile gm trows tcols (fst rc) (snd rc) |-> Frac f (cf rc)
    ensures
      gpu_matrix_subtile gm trows tcols (fst rc) (snd rc) |-> Frac f (ematrix_subtile em trows tcols (fst rc) (snd rc))
  {
    rewrite each cf rc as ematrix_subtile em trows tcols (fst rc) (snd rc);
  };
  forevery_map _ _ aux;
  forevery_unflatten #(natlt (rows / trows)) #(natlt (cols / tcols))
    (fun r c -> gpu_matrix_subtile gm trows tcols r c |-> Frac f (ematrix_subtile em trows tcols r c));
  gpu_matrix_untile gm trows tcols;
}


#push-options "--z3rlimit 20"
ghost
fn gpu_matrix_extract_tile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : nat { trows > 0 /\ trows /? rows })
  (tcols : nat { tcols > 0 /\ tcols /? cols })
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  ensures
    gpu_matrix_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc) **
    (forall* (tm' : ematrix et trows tcols).
      gpu_matrix_subtile gm trows tcols tr tc |-> Frac f tm' @==>
      gm |-> Frac f (update_tile em trows tcols tr tc tm'))
{
  gpu_matrix_pts_to_ref gm;
  gpu_matrix_tile gm trows tcols;
  forevery_flatten _;
  forevery_remove _ (tr, tc);
  ghost
  fn aux (tm' : ematrix et trows tcols)
    requires
      forall+
        (tr'tc' : natlt (rows / trows) & natlt (cols / tcols) { tr'tc' =!= (tr, tc) } ).
          gpu_matrix_subtile gm trows tcols (fst tr'tc') (snd tr'tc') |-> Frac f (ematrix_subtile em trows tcols (fst tr'tc') (snd tr'tc'))
    ensures
      gpu_matrix_subtile gm trows tcols tr tc |-> Frac f tm' @==>
      gm |-> Frac f (update_tile em trows tcols tr tc tm')
  {
    let em' = update_tile em trows tcols tr tc tm';
    assert pure (forall (tc' : natlt (cols / tcols)) (tr' : natlt (rows / trows)).
      tc =!= tc' \/ tr =!= tr' ==>
        (ematrix_subtile em trows tcols tr' tc'
         ==
         ematrix_subtile em' trows tcols tr' tc')
    );
    forevery_ext
      (fun (tr'tc' : natlt (rows / trows) & natlt (cols / tcols) { tr'tc' =!= (tr, tc) } ) ->
        gpu_matrix_subtile gm trows tcols (fst tr'tc') (snd tr'tc') |-> Frac f (ematrix_subtile em trows tcols (fst tr'tc') (snd tr'tc')))
      (fun (tr'tc' : natlt (rows / trows) & natlt (cols / tcols) { tr'tc' =!= (tr, tc) } ) ->
        gpu_matrix_subtile gm trows tcols (fst tr'tc') (snd tr'tc') |-> Frac f (ematrix_subtile em' trows tcols (fst tr'tc') (snd tr'tc')));
    ghost
    fn aux ()
      requires
        forall+
        (tr'tc' : natlt (rows / trows) & natlt (cols / tcols) { tr'tc' =!= (tr, tc) } ).
          gpu_matrix_subtile gm trows tcols (fst tr'tc') (snd tr'tc') |-> Frac f (ematrix_subtile em' trows tcols (fst tr'tc') (snd tr'tc'))
      requires
        gpu_matrix_subtile gm trows tcols tr tc |-> Frac f tm'
      ensures
        gm |-> Frac f (update_tile em trows tcols tr tc tm')
    {
      assert pure (ematrix_subtile em' trows tcols tr tc == tm');
      rewrite
        gpu_matrix_subtile gm trows tcols tr tc |-> Frac f tm'
      as
        gpu_matrix_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em' trows tcols tr tc);
      forevery_insert
        #(natlt (rows / trows) & natlt (cols / tcols))
        #(fun tr'tc' -> tr'tc' =!= (tr, tc))
        (fun (tr'tc' : natlt (rows / trows) & natlt (cols / tcols)) ->
          gpu_matrix_subtile gm trows tcols (fst tr'tc') (snd tr'tc') |-> Frac f (ematrix_subtile em' trows tcols (fst tr'tc') (snd tr'tc')))
        (tr, tc);
      forevery_unrefine _;
      forevery_unflatten' _;
      gpu_matrix_untile gm trows tcols #em';
      ()
    };
    Pulse.Lib.Trade.intro_trade _ _ _ aux;
  };
  Pulse.Lib.Forall.intro_forall _ aux;
}
#pop-options

inline_for_extraction noextract
fn gpu_matrix_extract_tile_st
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : erased nat { trows > 0 /\ trows /? rows })
  (tcols : erased nat { tcols > 0 /\ tcols /? cols })
  (tr : enatlt (rows / trows))
  (tc : enatlt (cols / tcols))
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  returns tc_tile : gpu_matrix et (subtile_layout l trows tcols tr tc)
  // use rewrites_to?
  ensures pure (tc_tile == gpu_matrix_subtile gm trows tcols tr tc)
  ensures
    tc_tile |-> Frac f (ematrix_subtile em trows tcols tr tc) **
    (forall* (tm' : ematrix et trows tcols).
      tc_tile |-> Frac f tm' @==>
      gm |-> Frac f (update_tile em trows tcols tr tc tm'))
{
  gpu_matrix_extract_tile gm trows tcols tr tc;
  gpu_matrix_subtile gm trows tcols tr tc
}

ghost
fn gpu_matrix_extract_tile_ro
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  ensures
    factored
      (gpu_matrix_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc))
      (gm |-> Frac f em)
{
  gpu_matrix_extract_tile gm trows tcols tr tc;
  Pulse.Lib.Forall.elim_forall (ematrix_subtile em trows tcols tr tc);
  rewrite each (update_tile em trows tcols tr tc (ematrix_subtile em trows tcols tr tc))
    as em;
}

inline_for_extraction noextract
fn gpu_matrix_extract_tile_ro'
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : erased nat { trows > 0 /\ trows /? rows })
  (tcols : erased nat { tcols > 0 /\ tcols /? cols })
  (tr : enatlt (rows / trows))
  (tc : enatlt (cols / tcols))
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  returns gm' : gpu_matrix et (subtile_layout l trows tcols tr tc)
  ensures
    rewrites_to gm' (gpu_matrix_subtile gm trows tcols tr tc) **
    factored
      (gm' |-> Frac f (ematrix_subtile em trows tcols tr tc))
      (gm |-> Frac f em)
{
  gpu_matrix_extract_tile_ro gm trows tcols tr tc;
  gpu_matrix_subtile gm trows tcols tr tc;
}

ghost
fn gpu_matrix_explode_tiled
  (#et : Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (#em : ematrix et rows cols)
  requires
    gm |-> em
  ensures
    forall+ (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
            (i : natlt trows) (j : natlt tcols).
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) i j
        (macc em (tr * trows + i) (tc * tcols + j))
{
  (* Step 1: Tile the matrix *)
  gpu_matrix_tile gm trows tcols;
  (* Now: forall+ tr tc. subtile |-> ematrix_subtile em *)

  (* Step 2: For each subtile, explode to per-cell *)
  ghost
  fn aux (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
    requires
      gpu_matrix_subtile gm trows tcols tr tc |-> ematrix_subtile em trows tcols tr tc
    ensures
      forall+ (i : natlt trows) (j : natlt tcols).
        gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) i j
          (macc em (tr * trows + i) (tc * tcols + j))
  {
    gpu_matrix_explode (gpu_matrix_subtile gm trows tcols tr tc);
    (* Now: forall+ i j. subtile_cell i j (macc (ematrix_subtile em ...) i j) *)
    (* Need to rewrite: macc (ematrix_subtile em trows tcols tr tc) i j == macc em (tr*trows+i) (tc*tcols+j) *)
    forevery_ext_2
      (fun (i:natlt trows) (j:natlt tcols) ->
        gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) i j
          (macc (ematrix_subtile em trows tcols tr tc) i j))
      (fun (i:natlt trows) (j:natlt tcols) ->
        gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) i j
          (macc em (tr * trows + i) (tc * tcols + j)));
  };
  forevery_map_2 _ _ aux;
}

ghost
fn gpu_matrix_implode_tiled
  (#et : Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (val_fn : natlt (rows / trows) -> natlt (cols / tcols) -> natlt trows -> natlt tcols -> GTot et)
  requires
    pure (SZ.fits (mlayout_size l))
  requires
    forall+ (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
            (i : natlt trows) (j : natlt tcols).
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) i j
        (val_fn tr tc i j)
  ensures
    gm |-> mkM (fun (row : natlt rows) (col : natlt cols) ->
      val_fn (row / trows) (col / tcols) (row % trows) (col % tcols))
{
  (* Step 1: For each subtile, implode cells back to subtile ownership *)
  let em' = mkM (fun (row : natlt rows) (col : natlt cols) ->
    val_fn (row / trows) (col / tcols) (row % trows) (col % tcols));

  ghost
  fn aux (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
    requires
      forall+ (i : natlt trows) (j : natlt tcols).
        gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) i j
          (val_fn tr tc i j)
    ensures
      gpu_matrix_subtile gm trows tcols tr tc |-> ematrix_subtile em' trows tcols tr tc
  {
    (* Help SMT see val_fn tr tc i j == macc (ematrix_subtile em' ...) i j *)
    assert pure (forall (i:natlt trows) (j:natlt tcols). (tr * trows + i) / trows == tr);
    assert pure (forall (i:natlt trows) (j:natlt tcols). (tc * tcols + j) / tcols == tc);
    assert pure (forall (i:natlt trows) (j:natlt tcols). (tr * trows + i) % trows == i);
    assert pure (forall (i:natlt trows) (j:natlt tcols). (tc * tcols + j) % tcols == j);
    (* Rewrite val_fn to macc form *)
    forevery_ext_2
      (fun (i:natlt trows) (j:natlt tcols) ->
        gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) i j
          (val_fn tr tc i j))
      (fun (i:natlt trows) (j:natlt tcols) ->
        gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) i j
          (macc (ematrix_subtile em' trows tcols tr tc) i j));
    gpu_matrix_implode (gpu_matrix_subtile gm trows tcols tr tc);
  };
  forevery_map_2 _ _ aux;
  (* Now: forall+ tr tc. subtile |-> ematrix_subtile em' *)

  (* Step 2: Untile back to full matrix *)
  gpu_matrix_untile gm trows tcols;
}
