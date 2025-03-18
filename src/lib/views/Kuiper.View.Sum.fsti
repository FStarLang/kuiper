module Kuiper.View.Sum

#lang-pulse

open Kuiper
open Kuiper.View
open Kuiper.GhostMap
open Kuiper.Bijection
module SZ = FStar.SizeT

let aview_sum
  (#a : Type)
  (#len1 : nat) (#vt1 : Type)
  (#len2 : nat) (#vt2 : Type)
  (vw1 : aview a len1 vt1)
  (vw2 : aview a len2 vt2)
  : aview a (len1 + len2) (vt1 & vt2)
= {
  it = either vw1.it vw2.it;
  igm = is_ghost_map_prod vw1.igm vw2.igm;
  ibij = bij_either vw1.ibij vw2.ibij `bij_comp` bij_nat_sum _ _;
}

inline_for_extraction noextract
let cview_sum
  (#a:Type)
  (#len1:sz) (#vt1:Type)
  (#len2:sz) (#vt2:Type)
  (vw1 : aview a len1 vt1)
  (cw1 : cview vw1)
  (vw2 : aview a len2 vt2)
  (cw2 : cview vw2)
  (_ : squash (SZ.fits (len1 + len2)))
  : cview (aview_sum vw1 vw2)
= {
  lenfits = ();
  cit = either cw1.cit cw2.cit;
  cibij = bij_either cw1.cibij cw2.cibij `bij_comp` bij_sz_sum len1 len2;
}
