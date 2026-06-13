module Kuiper.EMatrix.Tiling
#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix
open Kuiper.Injection

let macc_ematrix_tiled #et #rows #cols em trows tcols i j = ()

#push-options "--z3rlimit 10"
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
#pop-options

#push-options "--z3rlimit 20 --fuel 0 --ifuel 0 --split_queries always --retry 3"
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

#push-options "--split_queries always"
let subtile_of_update_tile
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
  = if tr' = tr && tc' = tc then
      assert (equal (ematrix_subtile (update_tile em trows tcols tr tc etile) trows tcols tr' tc') etile)
    else
      assert (equal (ematrix_subtile (update_tile em trows tcols tr tc etile) trows tcols tr' tc') (ematrix_subtile em trows tcols tr' tc'))
#pop-options
