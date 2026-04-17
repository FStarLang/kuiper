module Kuiper.EMatrix.Tiling
#lang-pulse

open Kuiper
open Kuiper.EMatrix

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

let ematrix_tiled
  (#et : _)
  (#rows #cols : _)
  (em : ematrix et rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  : ematrix (ematrix et trows tcols) (rows/trows) (cols/tcols)
=
  mkM fun i j -> ematrix_subtile em trows tcols i j

let ematrix_from_tiles
  (#et : _)
  (#rows #cols : nat)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (f : natlt (rows / trows) -> natlt (cols / tcols) -> ematrix et trows tcols)
  : ematrix et rows cols
=
  mkM fun i j ->
    macc (f (i / trows) (j / tcols)) (i % trows) (j % tcols)

let update_tile
  (#et : _)
  (#rows #cols : nat)
  (em : ematrix et rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  (tm : ematrix et trows tcols)
  : ematrix et rows cols
=
  mkM fun i j ->
    if i / trows = tr && j / tcols = tc then
      macc tm (i % trows) (j % tcols)
    else
      macc em i j

val from_subtiles_id
  (#et : _)
  (#rows #cols : _)
  (em : ematrix et rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  : Lemma (ematrix_from_tiles trows tcols (ematrix_subtile em trows tcols)
           ==
           em)
          [SMTPat (ematrix_from_tiles trows tcols (ematrix_subtile em trows tcols))]

val tiles_from_subtiles_id
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
           [SMTPat (ematrix_subtile (ematrix_from_tiles trows tcols f) trows tcols tr tc)]

val update_tile_self
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

val subtile_of_update_tile
  (#et : _)
  (#rows #cols : _)
  (em : ematrix et rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  (etile : ematrix et trows tcols)
  (tr' : natlt (rows / trows))
  (tc' : natlt (cols / tcols))
  : Lemma (ematrix_subtile (update_tile em trows tcols tr tc etile) trows tcols tr' tc'
           ==
           (if tr = tr' && tc = tc' then etile else ematrix_subtile em trows tcols tr' tc'))
          [SMTPat (ematrix_subtile (update_tile em trows tcols tr tc etile) trows tcols tr' tc')]
