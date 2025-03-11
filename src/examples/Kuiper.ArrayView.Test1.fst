module Kuiper.ArrayView.Test1
#lang-pulse

open Kuiper
open Kuiper.ArrayView
open Kuiper.GhostMap
open Kuiper.Bijection
module F = FStar.FunctionalExtensionality
module SZ = FStar.SizeT
open FStar.SizeT

let lseq_is_ghost_map (et:Type) (len:nat) : is_ghost_map (erased (lseq et len)) (natlt len) et = {
  bij = {
    ff = (fun (v : erased (lseq et len)) -> F.on_g _ fun (i:natlt len) -> Seq.index (reveal v) i <: et);
    gg = (fun f -> hide (Seq.init_ghost len f));
    ff_gg = (fun f -> admit());
    gg_ff = magic();
  };
  acc = (fun v i -> reveal v @! i);
  upd = (fun v i x -> Seq.upd (reveal v) i x);
  l1 = ez;
  l2 = ez;
}

let aview_bij (et len vt vt' : _) (vw : aview et len vt) (b : vt =~ vt')
  : aview et len vt' = {
  it   = vw.it;
  igm  = {
    bij = bij_sym b `bij_comp` vw.igm.bij;
    acc = (fun v i   -> vw.igm.acc (b.gg v) i);
    upd = (fun v i x -> vw.igm.upd (b.gg v) i x |> b.ff);
    l1  = ez;
    l2  = ez;
  };
  ibij = vw.ibij;
}

inline_for_extraction noextract
let bij_nat_rev (len:nat) : (natlt len =~ natlt len) = {
  ff = (fun (i:natlt len) -> len-1-i <: natlt len);
  gg = (fun (i:natlt len) -> len-1-i <: natlt len);
  ff_gg = ez;
  gg_ff = ez;
}

inline_for_extraction noextract
let bij_sz_rev (len:nat{SZ.fits len}) : (szlt len =~ szlt len) = {
  ff = (fun (i:szlt len) -> SZ.uint_to_t len -^1sz-^i <: szlt len);
  gg = (fun (i:szlt len) -> SZ.uint_to_t len -^1sz-^i <: szlt len);
  ff_gg = ez;
  gg_ff = ez;
}

let base_view (et len : _) : aview et len (erased (lseq et len)) =
{
  it   = natlt len;
  igm  = lseq_is_ghost_map et len;
  ibij = bij_self _;
}

let r_base_view (et len : _) : aview et len (erased (lseq et len)) =
{
  it   = natlt len;
  igm  = lseq_is_ghost_map et len;
  ibij = bij_nat_rev _;
}

noeq
inline_for_extraction noextract
type _normal et len = | N of erased (lseq et len)

inline_for_extraction noextract
let bij__normal (et len : _) : (erased (lseq et len) =~ _normal et len) = {
  ff = (fun (v:erased (lseq et len)) -> N v);
  gg = (fun (N v:_normal et len) -> v);
  ff_gg = ez;
  gg_ff = ez;
}

let normal_view (et:Type) (len:nat) : aview et len (_normal et len) =
  aview_bij et len (erased (lseq et len)) (_normal et len)
    (base_view et len)
    (bij__normal et len)

noeq
inline_for_extraction noextract
type _reverse et len = | R of erased (lseq et len)

inline_for_extraction noextract
let bij__reverse (et len : _) : (erased (lseq et len) =~ _reverse et len) = {
  ff = (fun (v:erased (lseq et len)) -> R v);
  gg = (fun (R v:_reverse et len) -> v);
  ff_gg = ez;
  gg_ff = ez;
}

let reverse_view (et:Type) (len:nat) : aview et len (_reverse et len) =
  aview_bij et len (erased (lseq et len)) (_reverse et len)
    (r_base_view et len)
    (bij__reverse et len)

inline_for_extraction noextract
instance cnormal_view et (len : nat{SZ.fits len}) : cview (normal_view et len) = {
  cit = szlt len;
  cibij = bij_self _;
  lenfits = ();
}

inline_for_extraction noextract
instance creverse_view et (len : nat{SZ.fits len}) : cview (reverse_view et len) = {
  cit = szlt len;
  cibij = bij_sz_rev len;
  lenfits = ();
}

fn test (a : varray (normal_view u32 50))
  preserves gpu
  requires a |-> N 's
  returns u32
  ensures a |-> N 's
{ varray_read a 0sz; }

fn test2 (a : varray (reverse_view u32 50))
  preserves gpu
  requires a |-> R 's
  returns u32
  ensures a |-> R 's
{ varray_read a 0sz; }

fn write1 (a : varray (normal_view u32 50))
  preserves gpu
  requires a |-> N 's
  ensures  a |-> N (Seq.upd 's 0 123ul)
{ varray_write a 0sz 123ul; }

fn write2 (a : varray (reverse_view u32 50))
  (#s : erased (lseq u32 50))
  preserves gpu
  requires a |-> R s
  ensures  a |-> R (Seq.upd s 0 123ul)
{ varray_write a 0sz 123ul; }


(* awkward, we should be able to start from a random array (not "core a")
   and use abs on it. *)
fn write3 (a : varray (reverse_view u32 50))
  (#s : erased (lseq u32 50))
  preserves gpu
  requires core a |-> to_seq (reverse_view u32 50) (R s)
  ensures  core a |-> Seq.upd (to_seq (reverse_view u32 50) (R s)) 49 123ul
{
  let a' = varray_abs a (reverse_view u32 50);
  write2 a';
  varray_concr a';
  rewrite each core #UInt32.t #(hide #nat 50) #(_reverse UInt32.t 50)
          #(reverse_view UInt32.t 50) a' as core a;

  assert (pure (Seq.equal
    (Seq.upd (to_seq (reverse_view u32 50) (R s)) 49 123ul)
    (to_seq (reverse_view u32 50) (R (Seq.upd s 0 123ul)))));
  ();
}
