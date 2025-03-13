module Kuiper.View.Layout4

#lang-pulse
open Kuiper
open Kuiper.Bijection

noeq
type cfg = {
  rows : pos;
  cols : pos;
  tile : pos;
  rblocks : pos;
  cblocks : pos;
  p : squash (
        tile /? rows /\
        tile /? cols /\
        rblocks == rows / tile /\
        cblocks == cols / tile
     );
}

let idxt1 (c:cfg) =
  natlt c.rblocks &
  natlt c.tile &
  natlt c.cblocks &
  natlt c.tile

let idxt2 (c:cfg) =
  natlt c.rows &
  natlt c.cols

let f1 (c:cfg) (i : idxt1 c) : idxt2 c =
  let bi, si, bj, sj = i in
  let r = bi * c.tile + si in
  let c = bj * c.tile + sj in
  r, c

let f2 (c:cfg) (i : idxt2 c) : idxt1 c =
  let row, col = i in
  let bi = row / c.tile in
  let bj = col / c.tile in
  let si = row % c.tile in
  let sj = col % c.tile in
  bi, si, bj, sj

val inv1 (c:cfg) (i : idxt1 c) : Lemma (f2 c (f1 c i) == i)
val inv2 (c:cfg) (i : idxt2 c) : Lemma (f1 c (f2 c i) == i)

let bij (rows cols : pos) (tile : pos{tile /? rows /\ tile /? cols})
  : ((natlt rows & natlt cols) =~ (natlt (rows / tile) & natlt tile & natlt (cols / tile) & natlt tile))
= let c : cfg = {
    rows = rows;
    cols = cols;
    tile = tile;
    rblocks = rows / tile;
    cblocks = cols / tile;
    p = ()
  }
  in
  {
    ff = (fun x -> f2 c x);
    gg = (fun x -> f1 c x);
    ff_gg = (fun x -> inv1 c x);
    gg_ff = (fun x -> inv2 c x);
  }
