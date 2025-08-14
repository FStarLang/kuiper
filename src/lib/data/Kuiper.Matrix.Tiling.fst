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

instance c_subtile_layout
  (#rows #cols : _)
  (l : mlayout rows cols) {| c : clayout l |}
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : clayout (subtile_layout l trows tcols tr tc) =
  {
    m_len = c.m_len;

    c_to = (fun i j -> c.c_to (c_tr.x *^ c_trows.x +^ i) (c_tc.x *^ c_tcols.x +^ j));

    // Do we actually use these? Try replacing them by magics and
    // see if anything breaks.
    m_rows = c_trows.x;
    m_cols = c_tcols.x;
  }

(* Just a cast *)
let gpu_matrix_subtile
  (#et : _)
  (#rows #cols : _)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : Tot (gpu_matrix et (subtile_layout l trows tcols tr tc))
  = from_array (subtile_layout l trows tcols tr tc)
               (core gm)

ghost
fn gpu_matrix_tile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (#em : ematrix et rows cols)
  requires
    gm |-> em
  ensures
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
        gpu_matrix_subtile gm trows tcols tr tc |-> ematrix_subtile em trows tcols tr tc
{
  admit();
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
  requires
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
        gpu_matrix_subtile gm trows tcols tr tc |-> ematrix_subtile em trows tcols tr tc
  ensures
    gm |-> em
{
  admit();
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
  admit();
}
