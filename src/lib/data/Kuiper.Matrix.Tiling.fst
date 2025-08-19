module Kuiper.Matrix.Tiling
#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix
open Kuiper.Injection
module SZ = FStar.SizeT
module Enumerable = Kuiper.Enumerable

let from_subtiles_id
  (#et : _)
  (#rows #cols : _)
  (em : ematrix et rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  : Lemma (ematrix_from_tiles trows tcols (ematrix_subtile em trows tcols)
           ==
           em)
= assert (equal (ematrix_from_tiles trows tcols (ematrix_subtile em trows tcols)) em);
  ()

#push-options "--z3rlimit 20"
let tiles_from_subtiles_id
  (#et : _)
  (#rows #cols : _)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (f : natlt (rows / trows) -> natlt (cols / tcols) -> ematrix et trows tcols)
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : Lemma (ematrix_subtile (ematrix_from_tiles trows tcols f) trows tcols tr tc
           ==
           f tr tc)
= assert (equal (ematrix_subtile (ematrix_from_tiles trows tcols f) trows tcols tr tc) (f tr tc));
  ()
#pop-options

let update_tile_self
  (#et : _)
  (#rows #cols : _)
  (em : ematrix et rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : Lemma (update_tile em trows tcols tr tc (ematrix_subtile em trows tcols tr tc)
           ==
           em)
          [SMTPat (update_tile em trows tcols tr tc (ematrix_subtile em trows tcols tr tc))]
= assert (equal (update_tile em trows tcols tr tc (ematrix_subtile em trows tcols tr tc)) em)

inline_for_extraction noextract
let strided_row_major_subtile_offset
  (#rows #cols : erased nat)
  (l : mlayout rows cols)
  {| sub : strided_row_major l |}
  (trows : sz {trows > 0 /\ trows /? rows})
  (tcols : sz {tcols > 0 /\ tcols /? cols})
  (tr    : szlt (rows / trows))
  (tc    : szlt (cols / tcols))
  : res : sz { SZ.v res == sub.offset + sub.stride * (tr * trows) + tc * tcols }
  = sub.pf (tr * trows) (tc * tcols);
    assert (l.map.f (tr * trows, tc * tcols) == sub.offset + sub.stride * (tr * trows) + tc * tcols);
    assume (SZ.fits (l.map.f (tr * trows, tc * tcols))); // We actually don't have this right now
    assume (sub.stride > 0); // This one we should be able to prove, but it's needed to know that tr*trows does not overflow!
    sub.offset +^ sub.stride *^ (tr *^ trows) +^ tc *^ tcols

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
    == {}
    sub.offset + sub.stride * (tr * trows) + sub.stride * i + tc * tcols + j;
    == {}
      sub.offset + sub.stride * (tr * trows) + tc * tcols
        + sub.stride * i + j;
  };
  ()

inline_for_extraction noextract
instance strided_row_major_subtile (#rows #cols : erased nat)
  (l : mlayout rows cols)
  {| sub : strided_row_major l |}
  (trows : erased nat {trows > 0 /\ trows /? rows})
  (tcols : erased nat {tcols > 0 /\ tcols /? cols})
  (tr    : enatlt (rows / trows))
  (tc    : enatlt (cols / tcols))
  {| c_trows : concrete_sz (hide (reveal trows)),
     c_tcols : concrete_sz (hide (reveal tcols)),
     c_tr    : concrete_sz (hide (reveal tr)),
     c_tc    : concrete_sz (hide (reveal tc)),
  |}
  : strided_row_major (subtile_layout l trows tcols tr tc) =
{
  offset = strided_row_major_subtile_offset l (concr' c_trows) (concr' c_tcols) (concr' c_tr) (concr' c_tc);
  stride = sub.stride;
  pf = (fun i j -> strided_row_major_subtile_proof #rows #cols l trows tcols tr tc i j);
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
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
      (gpu_matrix_subtile gm trows tcols tr tc |-> Frac f (tf tr tc))
  ensures
    gm |-> Frac f (ematrix_from_tiles trows tcols tf)
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
  gpu_matrix_untile' gm trows tcols _;
  from_subtiles_id em trows tcols;
  rewrite each ematrix_from_tiles trows tcols (ematrix_subtile em trows tcols)
            as em;
  ();
}

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
  gpu_matrix_tile gm trows tcols;
  forevery_extract_if_2 tr tc _;
  ghost
  fn aux (tm' : ematrix et trows tcols)
    requires
      forall+
        (tr' : natlt (rows / trows))
        (tc' : natlt (cols / tcols)).
          (if (Enumerable.to_nat tr', Enumerable.to_nat tc') = (Enumerable.to_nat tr, Enumerable.to_nat tc)
           then emp
           else
             // Using |-> below fails
             gpu_matrix_pts_to (gpu_matrix_subtile gm trows tcols tr' tc') #f (ematrix_subtile em trows tcols tr' tc'))
    ensures
      gpu_matrix_subtile gm trows tcols tr tc |-> Frac f tm' @==>
      gm |-> Frac f (update_tile em trows tcols tr tc tm')
  {
    ghost
    fn aux ()
      requires
        forall+
          (tr' : natlt (rows / trows))
          (tc' : natlt (cols / tcols)).
            (if (Enumerable.to_nat tr', Enumerable.to_nat tc') = (Enumerable.to_nat tr, Enumerable.to_nat tc)
            then emp
            else
              // Using |-> below fails
              gpu_matrix_pts_to (gpu_matrix_subtile gm trows tcols tr' tc') #f (ematrix_subtile em trows tcols tr' tc'))
      requires
        gpu_matrix_subtile gm trows tcols tr tc |-> Frac f tm'
      ensures
        gm |-> Frac f (update_tile em trows tcols tr tc tm')
    {
      (* This bit is really boring. *)
      admit();
    };
    Pulse.Lib.Trade.intro_trade _ _ _ aux;
  };
  Pulse.Lib.Forall.intro_forall _ aux;
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
  ()
}

fn gpu_matrix_extract_tile'
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
    pure (gm' == gpu_matrix_subtile gm trows tcols tr tc) **
    factored
      (gm' |-> Frac f (ematrix_subtile em trows tcols tr tc))
      (gm |-> Frac f em)
{
  // reveal was only required because of pos in
  //   declaration of gpu_matrix_extract_tile
  gpu_matrix_extract_tile gm trows tcols tr tc;
  gpu_matrix_subtile gm trows tcols tr tc;
}