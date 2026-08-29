module Kuiper.EMatrix.Tiling
open Kuiper.Chest
#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Injection

let subtile_acc2
  (#et : Type)
  (#rows #cols : nat)
  (em : chest2 et rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  (i : natlt trows)
  (j : natlt tcols)
  : Lemma (
      acc2 (ematrix_subtile em trows tcols tr tc) i j
      == acc2 em (tr * trows + i) (tc * tcols + j))
= Kuiper.EMatrix.macc_mkM
    (fun i j -> acc2 em (tr * trows + i) (tc * tcols + j)) i j

let chest_comb_subtile
  (#rows #cols : nat)
  (#t1 #t2 #t3 : Type)
  (f : t1 -> t2 -> t3)
  (c1 : chest2 t1 rows cols)
  (c2 : chest2 t2 rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : Lemma (
      ematrix_subtile (chest_comb f c1 c2) trows tcols tr tc
      == chest_comb f
          (ematrix_subtile c1 trows tcols tr tc)
          (ematrix_subtile c2 trows tcols tr tc))
= assert (Kuiper.Chest.equal
    (ematrix_subtile (chest_comb f c1 c2) trows tcols tr tc)
    (chest_comb f
      (ematrix_subtile c1 trows tcols tr tc)
      (ematrix_subtile c2 trows tcols tr tc)))

let macc_ematrix_tiled #et #rows #cols em trows tcols i j = ()

#push-options "--z3rlimit 10"
let from_subtiles_id
  (#et : _)
  (#rows #cols : _)
  (em : chest2 et rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  : Lemma (ematrix_from_tiles trows tcols (ematrix_subtile em trows tcols)
           ==
           em)
= assert (equal (ematrix_from_tiles trows tcols (ematrix_subtile em trows tcols)) em);
  ()
#pop-options

#push-options "--z3rlimit 20 --fuel 0 --ifuel 0 --retry 5"
let tiles_from_subtiles_id
  (#et : _)
  (#rows #cols : _)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (f : natlt (rows / trows) -> natlt (cols / tcols) -> chest2 et trows tcols)
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
  (em : chest2 et rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : Lemma (update_tile em trows tcols tr tc (ematrix_subtile em trows tcols tr tc)
           ==
           em)
          [SMTPat (update_tile em trows tcols tr tc (ematrix_subtile em trows tcols tr tc))]
= assert (equal (update_tile em trows tcols tr tc (ematrix_subtile em trows tcols tr tc)) em)

#push-options "--z3rlimit 40"
let subtile_of_update_tile
  (#et : _)
  (#rows #cols : _)
  (em : chest2 et rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  (etile : chest2 et trows tcols)
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
