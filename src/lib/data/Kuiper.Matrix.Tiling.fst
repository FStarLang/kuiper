module Kuiper.Matrix.Tiling
#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix

let ematrix_subtile
  (#et : _)
  (#rows #cols : _)
  (em : ematrix et rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : ematrix et trows tcols
=
  mkM fun i j ->
    macc em (tr * trows + i) (tc * tcols + j)

let ij_map
  (#rows #cols : _)
  (l : mlayout rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : GTot (natlt trows & natlt tcols -> natlt l.len)
= let subf = l.map.f in
  fun (i, j) -> subf (tr * trows + i, tc * tcols + j)

let subtile_layout
  (#rows #cols : _)
  (l : mlayout rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : mlayout trows tcols =
  let f = l.map.f in
  {
    len = l.len;
    map = {
      f = ij_map l trows tcols tr tc;
      is_inj = ez;
    }
  }

inline_for_extraction noextract
instance c_subtile_layout
  (#rows #cols : erased nat)
  (l : mlayout rows cols) {| c : clayout l |}
  (trows : erased pos {trows /? rows})
  (tcols : erased pos {tcols /? cols})
  (tr    : (enatlt (rows / trows)))
  (tc    : (enatlt (cols / tcols)))
  {| c_trows : concrete_sz (hide (reveal trows)),
     c_tcols : concrete_sz (hide (reveal tcols)),
     c_tr    : concrete_sz (hide (reveal tr)),
     c_tc    : concrete_sz (hide (reveal tc)),
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
  gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) i j v
  ==
  gpu_matrix_pts_to_cell gm (tr * trows + i) (tc * tcols + j) v
)
=
  admit() // I think this should be provable once we expose enough

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
    gpu_matrix_pts_to_cell gm (tr * trows + i) (tc * tcols + j) v
  ensures
    gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) i j v
{
  cell_convert_eq gm trows tcols tr tc i j f v;
  rewrite
    gpu_matrix_pts_to_cell gm (tr * trows + i) (tc * tcols + j) v
       as
    gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) i j v;
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
    gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) i j v
  ensures
    gpu_matrix_pts_to_cell gm (tr * trows + i) (tc * tcols + j) v
{
  cell_convert_eq gm trows tcols tr tc i j f v;
  rewrite
    gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) i j v
       as
    gpu_matrix_pts_to_cell gm (tr * trows + i) (tc * tcols + j) v;
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
  admit ();
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
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
        gpu_matrix_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc)
  ensures
    gm |-> Frac f em
{
  admit ();
}

ghost
fn gpu_matrix_untile0
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  requires
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
      (exists* em.
        gpu_matrix_subtile gm trows tcols tr tc |-> em)
  ensures
    exists* em. gm |-> em
{
  admit ();
}

ghost
fn gpu_matrix_extract_tile
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
  gpu_matrix_tile gm trows tcols;
  forevery_extract_2 tr tc _;
  trade_map _ _ _ (fun () -> gpu_matrix_untile gm trows tcols);
}
