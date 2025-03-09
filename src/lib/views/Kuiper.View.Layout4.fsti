module Kuiper.View.Layout4

#lang-pulse
open Kuiper
open Kuiper.Bijection

noeq
type cfg = {
  rows : pos;
  cols : pos;
  bdim : pos;
  rblocks : pos;
  cblocks : pos;
  p : squash (
        bdim /? rows /\
        bdim /? cols /\
        rblocks == rows / bdim /\
        cblocks == cols / bdim
     );
}

let idxt1 (c:cfg) =
  natlt c.rblocks &
  natlt c.bdim &
  natlt c.cblocks &
  natlt c.bdim

let idxt2 (c:cfg) =
  natlt c.rows &
  natlt c.cols

let f1 (c:cfg) (i : idxt1 c) : idxt2 c =
  let bi, si, bj, sj = i in
  let r = bi * c.bdim + si in
  let c = bj * c.bdim + sj in
  r, c

let f2 (c:cfg) (i : idxt2 c) : idxt1 c =
  let row, col = i in
  let bi = row / c.bdim in
  let bj = col / c.bdim in
  let si = row % c.bdim in
  let sj = col % c.bdim in
  bi, si, bj, sj

val inv1 (c:cfg) (i : idxt1 c) : Lemma (f2 c (f1 c i) == i)
val inv2 (c:cfg) (i : idxt2 c) : Lemma (f1 c (f2 c i) == i)

let bij (rows cols : pos) (bdim : pos{bdim /? rows /\ bdim /? cols})
  : ((natlt rows & natlt cols) =~ (natlt (rows / bdim) & natlt bdim & natlt (cols / bdim) & natlt bdim))
= let c : cfg = {
    rows = rows;
    cols = cols;
    bdim = bdim;
    rblocks = rows / bdim;
    cblocks = cols / bdim;
    p = ()
  }
  in
  {
    ff = (fun x -> f2 c x);
    gg = (fun x -> f1 c x);
    ff_gg = (fun x -> inv1 c x);
    gg_ff = (fun x -> inv2 c x);
  }
