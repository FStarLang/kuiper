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
module B = Kuiper.Barrier
open Kuiper.Array1
open Kuiper.Index
open Pulse.Lib.Trade { (@==>) }
open Kuiper.Math { even, odd }

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
  (gS : M.array2 et lS)
  (gK : M.array2 et lK { M.is_global gK })
  (gV : M.array2 et lV { M.is_global gV })
  (gQ : M.array2 et lQ { M.is_global gQ })
  (gO : M.array2 et lO { M.is_global gO })
  (gl : M.array2 et ll { M.is_global gl })
  (gm : M.array2 et lm { M.is_global gm })
  (eK eV eQ : ematrix et n d)
  (fK fV fQ : perm)
  (tid: natlt nthr)
  : slprop =
  (gK |-> Frac (fK /. nthr) eK) **
  (gV |-> Frac (fV /. nthr) eV) **
  (gQ |-> Frac (fQ /. nthr) eQ) **
  (exists* (eS : ematrix et nthr nthr) (eO : ematrix et n d) (el: ematrix et 1 n) (em: ematrix et 1 n). 
    array2_subtile gS 1 (SZ.v nthr) tid 0 |-> ematrix_subtile eS 1 (SZ.v nthr) tid 0 **
    array2_stride_subtile gl 1 (SZ.v nthr) 0 tid |-> ematrix_stride_subtile el 1 (SZ.v nthr) 0 tid **
    array2_stride_subtile gm 1 (SZ.v nthr) 0 tid |-> ematrix_stride_subtile em 1 (SZ.v nthr) 0 tid **
    array2_stride_subtile gO (SZ.v nthr) 1 tid 0 |-> ematrix_stride_subtile eO (SZ.v nthr) 1 tid 0)

(* The full (untiled) global-memory resources the kernel owns.  Since the
   kernel has no functional spec, the post is identical to the pre with O/l/m
   merely live.  Used as both [full_pre] and [full_post] of the kernel_desc. *)
unfold
let full_io_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d nthr : szp { nthr /? n /\ SZ.fits (nthr * nthr) })
  (#lS: M.layout nthr nthr)
  (#lK #lV #lQ #lO: M.layout n d)
  (#ll #lm: M.layout 1 n)
  {| ctlayout lS, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gS : M.array2 et lS)
  (gK : M.array2 et lK { M.is_global gK })
  (gV : M.array2 et lV { M.is_global gV })
  (gQ : M.array2 et lQ { M.is_global gQ })
  (gO : M.array2 et lO { M.is_global gO })
  (gl : M.array2 et ll { M.is_global gl })
  (gm : M.array2 et lm { M.is_global gm })
  (eK eV eQ : ematrix et n d)
  (fK fV fQ : perm)
  : slprop =
  (gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
  live gS ** live gO ** live gl ** live gm

(* The full (untiled) global-memory resources the kernel owns, WITHOUT the
   scratch [gS] (which now lives in shared memory).  Used as both the host
   pre/post and the (single) block's [block_pre]/[block_post]. *)
unfold
let full_io_fa_nos
  (#et : Type0) {| scalar et, floating et |}
  (n d nthr : szp { nthr /? n /\ SZ.fits (nthr * nthr) })
  (#lK #lV #lQ #lO: M.layout n d)
  (#ll #lm: M.layout 1 n)
  {| ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gK : M.array2 et lK { M.is_global gK })
  (gV : M.array2 et lV { M.is_global gV })
  (gQ : M.array2 et lQ { M.is_global gQ })
  (gO : M.array2 et lO { M.is_global gO })
  (gl : M.array2 et ll { M.is_global gl })
  (gm : M.array2 et lm { M.is_global gm })
  (eK eV eQ : ematrix et n d)
  (fK fV fQ : perm)
  : slprop =
  (gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
  live gO ** live gl ** live gm

(* Shared-memory request: a single flat scratch array of size [nthr * nthr],
   viewed (per block) as the [nthr x nthr] gS matrix. *)
let shmems_desc_fa (et:Type0) {| scalar et |} (nthr:szp{SZ.fits (nthr * nthr)}) : list shmem_desc =
  [ SHArray et (nthr *^ nthr) ]

(* View the single shared array of a block as the gS scratch matrix. *)
let gS_of_sh
  (#et:Type0) {| scalar et |}
  (n d nthr:szp{SZ.fits (nthr*nthr)})
  (lS : M.full_layout nthr nthr)
  (sh : c_shmems (shmems_desc_fa et nthr))
  : M.array2 et lS
  = M.from_array lS (fst sh)

(* Pure side-conditions carried across the kernel launch (needed to
   re-assemble the tiled write-side matrices in teardown). *)
unfold
let frame_fa
  (n d nthr : szp)
  (lS: M.layout nthr nthr)
  (lO: M.layout n d)
  (ll lm: M.layout 1 n)
  : slprop =
  pure (SZ.fits (M.layout_size lS) /\ SZ.fits (M.layout_size lO) /\
        SZ.fits (M.layout_size ll) /\ SZ.fits (M.layout_size lm))

(* Split the full resources into per-thread strided sub-views. *)
ghost
fn setup_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d nthr : szp { nthr /? n /\ SZ.fits (nthr * nthr) })
  (#lS: M.layout nthr nthr)
  (#lK #lV #lQ #lO: M.layout n d)
  (#ll #lm: M.layout 1 n)
  {| ctlayout lS, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gS : M.array2 et lS)
  (gK : M.array2 et lK { M.is_global gK })
  (gV : M.array2 et lV { M.is_global gV })
  (gQ : M.array2 et lQ { M.is_global gQ })
  (gO : M.array2 et lO { M.is_global gO })
  (gl : M.array2 et ll { M.is_global gl })
  (gm : M.array2 et lm { M.is_global gm })
  (eK eV eQ : ematrix et n d)
  (#fK #fV #fQ : perm)
  ()
  norewrite
  requires
    full_io_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ
  ensures
    (forall+ (tid : natlt nthr).
       kpre_post_outer_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ tid) **
    frame_fa n d nthr lS lO ll lm

(* Re-assemble the per-thread strided sub-views back into the full
   resources (write-side matrices end up merely live). *)
ghost
fn teardown_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d nthr : szp { nthr /? n /\ SZ.fits (nthr * nthr) })
  (#lS: M.layout nthr nthr)
  (#lK #lV #lQ #lO: M.layout n d)
  (#ll #lm: M.layout 1 n)
  {| ctlayout lS, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gS : M.array2 et lS)
  (gK : M.array2 et lK { M.is_global gK })
  (gV : M.array2 et lV { M.is_global gV })
  (gQ : M.array2 et lQ { M.is_global gQ })
  (gO : M.array2 et lO { M.is_global gO })
  (gl : M.array2 et ll { M.is_global gl })
  (gm : M.array2 et lm { M.is_global gm })
  (eK eV eQ : ematrix et n d)
  (#fK #fV #fQ : perm)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
       kpre_post_outer_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ tid) **
    frame_fa n d nthr lS lO ll lm
  ensures
    full_io_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ

(* Block-level setup: convert the block's shared scratch array into the gS
   matrix view and split (together with the global gO/gl/gm and the
   fractionally-shared gK/gV/gQ) into the per-thread sub-views. *)
ghost
fn block_setup_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d nthr : szp { nthr /? n /\ SZ.fits (nthr * nthr) })
  (lS : M.full_layout nthr nthr)
  (#lK #lV #lQ #lO: M.layout n d)
  (#ll #lm: M.layout 1 n)
  {| ctlayout lS, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gK : M.array2 et lK { M.is_global gK })
  (gV : M.array2 et lV { M.is_global gV })
  (gQ : M.array2 et lQ { M.is_global gQ })
  (gO : M.array2 et lO { M.is_global gO })
  (gl : M.array2 et ll { M.is_global gl })
  (gm : M.array2 et lm { M.is_global gm })
  (eK eV eQ : ematrix et n d)
  (#fK #fV #fQ : perm)
  (sh : c_shmems (shmems_desc_fa et nthr))
  (bid : natlt 1sz)
  ()
  norewrite
  requires
    live_c_shmems sh **
    full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ
  ensures
    (forall+ (tid : natlt nthr).
       kpre_post_outer_fa n d nthr (gS_of_sh n d nthr lS sh) gK gV gQ gO gl gm eK eV eQ fK fV fQ tid) **
    frame_fa n d nthr lS lO ll lm

(* Block-level teardown: reassemble the per-thread sub-views and fold the gS
   matrix view back into the block's shared scratch array. *)
ghost
fn block_teardown_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d nthr : szp { nthr /? n /\ SZ.fits (nthr * nthr) })
  (lS : M.full_layout nthr nthr)
  (#lK #lV #lQ #lO: M.layout n d)
  (#ll #lm: M.layout 1 n)
  {| ctlayout lS, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gK : M.array2 et lK { M.is_global gK })
  (gV : M.array2 et lV { M.is_global gV })
  (gQ : M.array2 et lQ { M.is_global gQ })
  (gO : M.array2 et lO { M.is_global gO })
  (gl : M.array2 et ll { M.is_global gl })
  (gm : M.array2 et lm { M.is_global gm })
  (eK eV eQ : ematrix et n d)
  (#fK #fV #fQ : perm)
  (sh : c_shmems (shmems_desc_fa et nthr))
  (bid : natlt 1sz)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
       kpre_post_outer_fa n d nthr (gS_of_sh n d nthr lS sh) gK gV gQ gO gl gm eK eV eQ fK fV fQ tid) **
    frame_fa n d nthr lS lO ll lm
  ensures
    live_c_shmems sh **
    full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ

(* ═════════════════════════════════════════════════════════════════════════
   SHARED-MEMORY VARIANT (K, V, Q caching with a real barrier).

   Configuration unchanged: 1 block, [nthr] threads, [bc == br == nthr], so
   the K/V tile is [nthr x d].  Four shared arrays:
     - sK : K tile cache   ([nthr x d]), barrier-shared (all threads read all)
     - sV : V tile cache   ([nthr x d]), barrier-shared
     - sQ : Q row cache     ([nthr x d]), per-thread row [tid] (no barrier)
     - gS : softmax scratch ([nthr x nthr]), per-thread row [tid] (no barrier)

   The barrier contract is CONTENT-FREE (permission-only): this kernel has no
   functional spec, so it only tracks ownership of sK/sV.
   ───────────────────────────────────────────────────────────────────────── *)

let shmems_desc_fa_smem
  (et:Type0) {| scalar et |}
  (n d nthr:szp{SZ.fits (nthr * d) /\ SZ.fits (nthr * nthr)})
  : list shmem_desc =
  [ SHArray et (nthr *^ d)
  ; SHArray et (nthr *^ d)
  ; SHArray et (nthr *^ d)
  ; SHArray et (nthr *^ nthr) ]

let sK_of_sh
  (#et:Type0) {| scalar et |}
  (n d nthr:szp{SZ.fits (nthr*d) /\ SZ.fits (nthr*nthr)})
  (lKV : M.full_layout nthr d)
  (sh : c_shmems (shmems_desc_fa_smem et n d nthr))
  : M.array2 et lKV
  = M.from_array lKV (fst sh)

let sV_of_sh
  (#et:Type0) {| scalar et |}
  (n d nthr:szp{SZ.fits (nthr*d) /\ SZ.fits (nthr*nthr)})
  (lKV : M.full_layout nthr d)
  (sh : c_shmems (shmems_desc_fa_smem et n d nthr))
  : M.array2 et lKV
  = M.from_array lKV (fst (snd sh))

let sQ_of_sh
  (#et:Type0) {| scalar et |}
  (n d nthr:szp{SZ.fits (nthr*d) /\ SZ.fits (nthr*nthr)})
  (lKV : M.full_layout nthr d)
  (sh : c_shmems (shmems_desc_fa_smem et n d nthr))
  : M.array2 et lKV
  = M.from_array lKV (fst (snd (snd sh)))

let gS_of_sh'
  (#et:Type0) {| scalar et |}
  (n d nthr:szp{SZ.fits (nthr*d) /\ SZ.fits (nthr*nthr)})
  (lS : M.full_layout nthr nthr)
  (sh : c_shmems (shmems_desc_fa_smem et n d nthr))
  : M.array2 et lS
  = M.from_array lS (fst (snd (snd (snd sh))))

(* ── The (content-free) barrier contract over sK, sV. ─────────────────────
   [tc = n /^ nthr] tiles.  At even step [2j] (before the inner loop) each
   thread holds write-ownership of its own row of sK/sV (just loaded) and
   receives a fractional read of the WHOLE tile.  At odd step [2j+1] (after
   the inner loop) it gives back the fractional read and receives its row. *)

let fa_barrier_side_rin
  (#et:Type0) {| scalar et |}
  (n d nthr : szp)
  (#lKV : M.full_layout nthr d)
  (sK sV : M.array2 et lKV)
  : B.barrier_side (SZ.v nthr)
  = fun it tid ->
    if it >= 2 * SZ.v (n /^ nthr) then emp
    else if even it then
      (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sK 1 (SZ.v d) tid 0 |-> Frac 1.0R r) **
      (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sV 1 (SZ.v d) tid 0 |-> Frac 1.0R r)
    else
      (exists* (x:ematrix et (SZ.v nthr) (SZ.v d)). sK |-> Frac (1.0R /. (SZ.v nthr)) x) **
      (exists* (y:ematrix et (SZ.v nthr) (SZ.v d)). sV |-> Frac (1.0R /. (SZ.v nthr)) y)

let fa_barrier_side_rout
  (#et:Type0) {| scalar et |}
  (n d nthr : szp)
  (#lKV : M.full_layout nthr d)
  (sK sV : M.array2 et lKV)
  : B.barrier_side (SZ.v nthr)
  = fun it tid ->
    if it >= 2 * SZ.v (n /^ nthr) then emp
    else if even it then
      (exists* (x:ematrix et (SZ.v nthr) (SZ.v d)). sK |-> Frac (1.0R /. (SZ.v nthr)) x) **
      (exists* (y:ematrix et (SZ.v nthr) (SZ.v d)). sV |-> Frac (1.0R /. (SZ.v nthr)) y)
    else
      (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sK 1 (SZ.v d) tid 0 |-> Frac 1.0R r) **
      (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sV 1 (SZ.v d) tid 0 |-> Frac 1.0R r)

let fa_barrier_contract
  (#et:Type0) {| scalar et |}
  (n d nthr : szp)
  (#lKV : M.full_layout nthr d)
  (sK sV : M.array2 et lKV)
  : B.contract (SZ.v nthr)
  = {
    rin  = fa_barrier_side_rin  n d nthr sK sV;
    rout = fa_barrier_side_rout n d nthr sK sV;
  }

let fa_barrier_count (n d nthr : szp) : GTot nat = 2 * SZ.v (n /^ nthr)

(* The barrier transform proof obligation (content-free). *)
ghost
fn fa_barrier_ok
  (#et:Type0) {| scalar et |}
  (n d nthr : szp { nthr /? n /\ SZ.fits (nthr * d) })
  (#lKV : M.full_layout nthr d)
  {| ctlayout lKV |}
  (sK sV : M.array2 et lKV)
  (it : nat)
  requires
    forall+ (i:natlt (SZ.v nthr)). fa_barrier_side_rin n d nthr sK sV it i
  ensures
    forall+ (i:natlt (SZ.v nthr)). fa_barrier_side_rout n d nthr sK sV it i

(* ── Per-thread pre/post for the shared-memory kernel. ────────────────────
   The non-shared resources are exactly [kpre_post_outer_fa]; sK/sV/sQ add a
   content-free write-row each (barrier tokens are supplied by [f]). *)
unfold
let kpre_post_outer_fa_smem
  (#et : Type0) {| scalar et, floating et |}
  (n d nthr : szp { nthr /? n /\ SZ.fits (nthr * nthr) /\ SZ.fits (nthr * d) })
  (#lS: M.layout nthr nthr)
  (#lKV: M.full_layout nthr d)
  (#lK #lV #lQ #lO: M.layout n d)
  (#ll #lm: M.layout 1 n)
  {| ctlayout lS, ctlayout lKV, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gS : M.array2 et lS)
  (sK sV sQ : M.array2 et lKV)
  (gK : M.array2 et lK { M.is_global gK })
  (gV : M.array2 et lV { M.is_global gV })
  (gQ : M.array2 et lQ { M.is_global gQ })
  (gO : M.array2 et lO { M.is_global gO })
  (gl : M.array2 et ll { M.is_global gl })
  (gm : M.array2 et lm { M.is_global gm })
  (eK eV eQ : ematrix et n d)
  (fK fV fQ : perm)
  (tid: natlt nthr)
  : slprop =
  kpre_post_outer_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ tid **
  (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sK 1 (SZ.v d) tid 0 |-> Frac 1.0R r) **
  (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sV 1 (SZ.v d) tid 0 |-> Frac 1.0R r) **
  (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sQ 1 (SZ.v d) tid 0 |-> Frac 1.0R r)

(* Pure side-conditions carried across the launch for the smem variant
   (adds the [lKV] fits-fact needed to re-assemble sK/sV/sQ). *)
unfold
let frame_fa_smem
  (n d nthr : szp)
  (lS: M.layout nthr nthr)
  (lKV: M.layout nthr d)
  (lO: M.layout n d)
  (ll lm: M.layout 1 n)
  : slprop =
  pure (SZ.fits (M.layout_size lS) /\ SZ.fits (M.layout_size lKV) /\
        SZ.fits (M.layout_size lO) /\
        SZ.fits (M.layout_size ll) /\ SZ.fits (M.layout_size lm))

(* Block-level setup: view the four shared arrays as gS/sK/sV/sQ and split
   into per-thread sub-views (reusing [setup_fa] for the non-shared part). *)
ghost
fn block_setup_fa_smem
  (#et : Type0) {| scalar et, floating et |}
  (n d nthr : szp { nthr /? n /\ SZ.fits (nthr * nthr) /\ SZ.fits (nthr * d) })
  (lS : M.full_layout nthr nthr)
  (lKV : M.full_layout nthr d)
  (#lK #lV #lQ #lO: M.layout n d)
  (#ll #lm: M.layout 1 n)
  {| ctlayout lS, ctlayout lKV, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gK : M.array2 et lK { M.is_global gK })
  (gV : M.array2 et lV { M.is_global gV })
  (gQ : M.array2 et lQ { M.is_global gQ })
  (gO : M.array2 et lO { M.is_global gO })
  (gl : M.array2 et ll { M.is_global gl })
  (gm : M.array2 et lm { M.is_global gm })
  (eK eV eQ : ematrix et n d)
  (#fK #fV #fQ : perm)
  (sh : c_shmems (shmems_desc_fa_smem et n d nthr))
  (bid : natlt 1sz)
  ()
  norewrite
  requires
    live_c_shmems sh **
    full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ
  ensures
    (forall+ (tid : natlt nthr).
       kpre_post_outer_fa_smem n d nthr
         (gS_of_sh' n d nthr lS sh) (sK_of_sh n d nthr lKV sh) (sV_of_sh n d nthr lKV sh) (sQ_of_sh n d nthr lKV sh)
         gK gV gQ gO gl gm eK eV eQ fK fV fQ tid) **
    frame_fa_smem n d nthr lS lKV lO ll lm

ghost
fn block_teardown_fa_smem
  (#et : Type0) {| scalar et, floating et |}
  (n d nthr : szp { nthr /? n /\ SZ.fits (nthr * nthr) /\ SZ.fits (nthr * d) })
  (lS : M.full_layout nthr nthr)
  (lKV : M.full_layout nthr d)
  (#lK #lV #lQ #lO: M.layout n d)
  (#ll #lm: M.layout 1 n)
  {| ctlayout lS, ctlayout lKV, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gK : M.array2 et lK { M.is_global gK })
  (gV : M.array2 et lV { M.is_global gV })
  (gQ : M.array2 et lQ { M.is_global gQ })
  (gO : M.array2 et lO { M.is_global gO })
  (gl : M.array2 et ll { M.is_global gl })
  (gm : M.array2 et lm { M.is_global gm })
  (eK eV eQ : ematrix et n d)
  (#fK #fV #fQ : perm)
  (sh : c_shmems (shmems_desc_fa_smem et n d nthr))
  (bid : natlt 1sz)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
       kpre_post_outer_fa_smem n d nthr
         (gS_of_sh' n d nthr lS sh) (sK_of_sh n d nthr lKV sh) (sV_of_sh n d nthr lKV sh) (sQ_of_sh n d nthr lKV sh)
         gK gV gQ gO gl gm eK eV eQ fK fV fQ tid) **
    frame_fa_smem n d nthr lS lKV lO ll lm
  ensures
    live_c_shmems sh **
    full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ
