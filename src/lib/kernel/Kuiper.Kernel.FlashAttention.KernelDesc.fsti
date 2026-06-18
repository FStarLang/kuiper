module Kuiper.Kernel.FlashAttention.KernelDesc

#lang-pulse
open Kuiper
open Kuiper.EMatrix
open Kuiper.Array
open Kuiper.Injection
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Tiling
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major, c_l2_row_major }

module M = Kuiper.Array2 
module SZ = Kuiper.SizeT
module Trade = Pulse.Lib.Trade
module Array1 = Kuiper.Array1
open Kuiper.Array1
open Kuiper.Index
open Pulse.Lib.Trade { (@==>) }

// STRIDE TILE HELPERS


let stride_tile_inj_f
  (#rows #cols : nat)
  // stride between different i and j respectively in the tile
  (srows : pos {srows /? rows})
  (scols : pos {scols /? cols})
  (tr : natlt srows)
  (tc : natlt scols)
  : abs (M.desc (rows/srows) (cols/scols)) -> abs (M.desc rows cols)
=
   (fun (i, (j, ())) -> (tr + i * srows, (tc + j * scols, ())))

let stride_tile_inj
  (#rows #cols : nat)
  // stride between different i and j respectively in the tile
  (srows : pos {srows /? rows})
  (scols : pos {scols /? cols})
  (tr : natlt srows)
  (tc : natlt scols)
  : (abs (M.desc (rows/srows) (cols/scols)) @~> abs (M.desc rows cols))
= {
   f      = stride_tile_inj_f srows scols tr tc;
   is_inj = ez;
}

let stride_subtile_layout
  (#rows #cols : nat)
  (l : M.layout rows cols)
  (srows : pos {srows /? rows})
  (scols : pos {scols /? cols})
  (tr : natlt srows)
  (tc : natlt scols)
  : M.layout (rows/srows) (cols/scols) =
  {
    ulen = l.ulen;
    imap = inj_comp (stride_tile_inj srows scols tr tc) l.imap;
  }

inline_for_extraction noextract
instance val c_stride_subtile_layout
  (#rows #cols : erased nat)
  (l : M.layout rows cols)
  {| cc : ctlayout l |}
  (srows : erased int {0 < srows /\ srows /? rows})
  (scols : erased int {0 < scols /\ scols /? cols})
  (tr    : erased int {tr < srows})
  (tc    : erased int {tc < scols})
  {| concrete_sz srows, concrete_sz scols, concrete_sz tr, concrete_sz tc |}
  : ctlayout (stride_subtile_layout l srows scols tr tc)

inline_for_extraction noextract
val array2_stride_subtile
  (#et : _)
  (#rows #cols : erased nat)
  (#l : M.layout rows cols)
  (gm : M.array2 et l)
  (srows : erased nat {srows > 0 /\ srows /? rows})
  (scols : erased nat {scols > 0 /\ scols /? cols})
  (tr : enatlt srows)
  (tc : enatlt scols)
  : Tot (M.array2 et (stride_subtile_layout l srows scols tr tc))

val array2_stride_subtile_base
  (#et : _)
  (#rows #cols : erased nat)
  (#l : M.layout rows cols)
  (gm : M.array2 et l)
  (srows : erased nat {srows > 0 /\ srows /? rows})
  (scols : erased nat {scols > 0 /\ scols /? cols})
  (tr : enatlt srows)
  (tc : enatlt scols)
  : Lemma (
      M.core (array2_stride_subtile gm srows scols tr tc)
      ==
      M.core gm
    )
    [SMTPat (M.core (array2_stride_subtile gm srows scols tr tc))]

let is_array2_stride_subtile_global
  (#et : _)
  (#rows #cols : erased nat)
  (#l : M.layout rows cols)
  (#gm : M.array2 et l)
  (#srows : erased nat {srows > 0 /\ srows /? rows})
  (#scols : erased nat {scols > 0 /\ scols /? cols})
  (#tr : enatlt srows)
  (#tc : enatlt scols)
: Lemma
  (ensures
    M.is_global (array2_stride_subtile gm srows scols tr tc) <==>
    M.is_global gm)
  [SMTPat (M.is_global (array2_stride_subtile gm srows scols tr tc))]
= array2_stride_subtile_base gm srows scols tr tc

// STRIDE TILE EMATRIX-LEVEL HELPERS
//
// A _stride tile_ (tr, tc) of a (rows x cols) matrix is the sub-matrix of
// shape (rows/srows x cols/scols) whose element (i, j) is the original
// element (i * srows + tr, j * scols + tc).  In other words, consecutive
// tile rows/cols are separated by the strides [srows]/[scols], and the
// tile index (tr, tc) is the _residue_ of the original index modulo the
// stride.  This is the transpose, w.r.t. the (tile-index, in-tile-index)
// split, of the contiguous tiling in [Kuiper.Tensor.Tiling].

let ematrix_stride_subtile
  (#et : _)
  (#rows #cols : _)
  (em : ematrix et rows cols)
  (srows : pos {srows /? rows})
  (scols : pos {scols /? cols})
  (tr : natlt srows)
  (tc : natlt scols)
  : ematrix et (rows/srows) (cols/scols)
=
  mkM fun i j ->
    macc em (i * srows + tr) (j * scols + tc)

let ematrix_stride_from_tiles
  (#et : _)
  (#rows #cols : nat)
  (srows : pos {srows /? rows})
  (scols : pos {scols /? cols})
  (f : natlt srows -> natlt scols -> ematrix et (rows/srows) (cols/scols))
  : ematrix et rows cols
=
  mkM fun i j ->
    macc (f (i % srows) (j % scols)) (i / srows) (j / scols)

let update_stride_tile
  (#et : _)
  (#rows #cols : nat)
  (em : ematrix et rows cols)
  (srows : pos {srows /? rows})
  (scols : pos {scols /? cols})
  (tr : natlt srows)
  (tc : natlt scols)
  (tm : ematrix et (rows/srows) (cols/scols))
  : ematrix et rows cols
=
  mkM fun i j ->
    if i % srows = tr && j % scols = tc then
      macc tm (i / srows) (j / scols)
    else
      macc em i j

val stride_cell_convert_eq
  (#et : _)
  (#rows #cols : erased nat)
  (#l : M.layout rows cols)
  (gm : M.array2 et l)
  (srows : erased nat {srows > 0 /\ srows /? rows})
  (scols : erased nat {scols > 0 /\ scols /? cols})
  (tr : enatlt srows)
  (tc : enatlt scols)
  (i : natlt (rows / srows))
  (j : natlt (cols / scols))
  (f : perm)
  (v : et)
  : Lemma (
    M.pts_to_cell (array2_stride_subtile gm srows scols tr tc) #f (i, j) v
    ==
    M.pts_to_cell gm #f (i * srows + tr, j * scols + tc) v
  )

ghost
fn array2_stride_tile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (gm : M.array2 et l)
  (srows : pos { srows /? rows })
  (scols : pos { scols /? cols })
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  ensures
    forall+
      (tr : natlt srows)
      (tc : natlt scols).
        array2_stride_subtile gm srows scols tr tc |-> Frac f (ematrix_stride_subtile em srows scols tr tc)

ghost
fn array2_stride_untile'
  (#et:Type0)
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (gm : M.array2 et l)
  (srows : pos { srows /? rows })
  (scols : pos { scols /? cols })
  (tf : natlt srows -> natlt scols -> ematrix et (rows/srows) (cols/scols))
  (#f : perm)
  requires
    pure (SZ.fits (M.layout_size l))
  requires
    forall+
      (tr : natlt srows)
      (tc : natlt scols).
      (array2_stride_subtile gm srows scols tr tc |-> Frac f (tf tr tc))
  ensures
    gm |-> Frac f (ematrix_stride_from_tiles srows scols tf)

ghost
fn array2_stride_untile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (gm : M.array2 et l)
  (srows : pos { srows /? rows })
  (scols : pos { scols /? cols })
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    pure (SZ.fits (M.layout_size l))
  requires
    forall+
      (tr : natlt srows)
      (tc : natlt scols).
        array2_stride_subtile gm srows scols tr tc |-> Frac f (ematrix_stride_subtile em srows scols tr tc)
  ensures
    gm |-> Frac f em

ghost
fn array2_extract_stride_tile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (gm : M.array2 et l)
  (srows : pos { srows /? rows })
  (scols : pos { scols /? cols })
  (tr : natlt srows)
  (tc : natlt scols)
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  ensures
    array2_stride_subtile gm srows scols tr tc |-> Frac f (ematrix_stride_subtile em srows scols tr tc) **
    (forall* (tm' : ematrix et (rows/srows) (cols/scols)).
      array2_stride_subtile gm srows scols tr tc |-> Frac f tm' @==>
      gm |-> Frac f (update_stride_tile em srows scols tr tc tm'))

inline_for_extraction noextract
fn array2_extract_stride_tile_st
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : M.layout rows cols)
  (gm : M.array2 et l)
  (srows : erased nat { srows > 0 /\ srows /? rows })
  (scols : erased nat { scols > 0 /\ scols /? cols })
  (tr : enatlt srows)
  (tc : enatlt scols)
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  returns tc_tile : M.array2 et (stride_subtile_layout l srows scols tr tc)
  ensures pure (tc_tile == array2_stride_subtile gm srows scols tr tc)
  ensures
    tc_tile |-> Frac f (ematrix_stride_subtile em srows scols tr tc) **
    (forall* (tm' : ematrix et (rows/srows) (cols/scols)).
      tc_tile |-> Frac f tm' @==>
      gm |-> Frac f (update_stride_tile em srows scols tr tc tm'))

ghost
fn array2_extract_stride_tile_ro
  (#et:Type0)
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (gm : M.array2 et l)
  (srows : nat { srows > 0 /\ srows /? rows })
  (scols : nat { scols > 0 /\ scols /? cols })
  (tr : natlt srows)
  (tc : natlt scols)
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  ensures
    factored
      (array2_stride_subtile gm srows scols tr tc |-> Frac f (ematrix_stride_subtile em srows scols tr tc))
      (gm |-> Frac f em)

inline_for_extraction noextract
fn array2_extract_stride_tile_ro'
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : M.layout rows cols)
  (gm : M.array2 et l)
  (srows : erased nat {srows > 0 /\ srows /? rows })
  (scols : erased nat {scols > 0 /\ scols /? cols })
  (tr : enatlt srows)
  (tc : enatlt scols)
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  returns gm' : M.array2 et (stride_subtile_layout l srows scols tr tc)
  ensures
    rewrites_to gm' (array2_stride_subtile gm srows scols tr tc) **
    factored
      (gm' |-> Frac f (ematrix_stride_subtile em srows scols tr tc))
      (gm |-> Frac f em)



unfold
let kpre_post_inner_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d: szp)
  (bc br: szp { bc /? n /\ br /? n })
  (lSt: layout bc)
  (lK lV lQ: M.layout n d)
  (lOt: M.layout (n /^ br) d)
  (llt lmt: layout (n /^ br))
  {| ctlayout lSt, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lOt, ctlayout llt, ctlayout lmt |}
  (gSt: array1 et lSt)
  (gK: M.array2 et lK) 
  (gV: M.array2 et lV)
  (gQ: M.array2 et lQ)
  (gOt: M.array2 et lOt)
  (glt: array1 et llt)
  (gmt: array1 et lmt)
  (eK eV eQ: ematrix et n d)
  (#fK #fV #fQ: perm)
  : slprop =
  (gK |-> Frac fK eK) **
  (gV |-> Frac fV eV) **
  (gQ |-> Frac fQ eQ) **
  // No functional spec; note that O, l, m would have preconditions here though. S does not
  live gSt ** live gOt ** live glt ** live gmt

unfold
let kpre_post_outer_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d nthr : szp { nthr /? n /\ SZ.fits (nthr * nthr) })
  (#lS: M.layout nthr nthr)
  (#lK #lV #lQ #lO: M.layout n d)
// stupid hack to make it easier to express tiling these into n/^br,
// because we dont have such a ghost on array1 atm
// LATER: fix
  (#ll #lm: M.layout 1 n)
  {| ctlayout lS, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gS : M.array2 et lS { M.is_global gS })
  (gK : M.array2 et lK { M.is_global gK })
  (gV : M.array2 et lV { M.is_global gV })
  (gQ : M.array2 et lQ { M.is_global gQ })
  (gO : M.array2 et lO { M.is_global gO })
  (gl : M.array2 et ll { M.is_global gl })
  (gm : M.array2 et lm { M.is_global gm })
  (eK eV eQ : ematrix et n d)
  (#fK #fV #fQ : perm)
  (tid: szlt nthr)
  : slprop =
  (gK |-> Frac (fK /. nthr) eK) **
  (gV |-> Frac (fV /. nthr) eV) **
  (gQ |-> Frac (fQ /. nthr) eQ) **
  (exists* (eS : ematrix et nthr nthr) (eO : ematrix et n d) (el: ematrix et 1 n) (em: ematrix et 1 n). 
    array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0 |-> ematrix_subtile eS 1 (SZ.v nthr) (SZ.v tid) 0 **
    array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid) |-> ematrix_stride_subtile em 1 (SZ.v nthr) 0 (SZ.v tid) **
    array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid) |-> ematrix_stride_subtile em 1 (SZ.v nthr) 0 (SZ.v tid) **
    array2_stride_subtile gO (SZ.v nthr) 1 (SZ.v tid) 0 |-> ematrix_stride_subtile eO (SZ.v nthr) 1 (SZ.v tid) 0)