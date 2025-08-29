module Kuiper.Example.Sparse

#lang-pulse
open Kuiper
module SZ = FStar.SizeT


noeq
type sarray (et : Type0)
  (len : erased nat) =
  // ^ longitud "virtual" del array
{
  nnz   : sz; // número de no-zeros
  elems : gpu_array et nnz; // elementos (no zero)
  pos   : gpu_array sz nnz; // posición de cada elemento
}

let in_bounds (#nnz len : nat) (s : lseq sz nnz) : prop =
  forall i. i < nnz ==> Seq.index s i < len 

let no_repeats (#nnz : nat) (s : lseq sz nnz) : prop =
  // Seq.index s es una inyección
  forall (i j : natlt nnz). i <> j ==> Seq.index s i <> Seq.index s j

let valid_pos (#nnz len : nat) (s : lseq sz nnz) : prop =
  in_bounds len s /\ no_repeats s

let unsparse
  (#et:Type0) {| scalar et |}
  (nnz len : nat)
  (elems : lseq et nnz)
  (pos   : lseq sz nnz{valid_pos len pos}) 
  : lseq et len
=
  let open FStar.Seq in

  init len fun i ->
    let nat_pos = map_seq SZ.v pos in
    map_seq_len SZ.v pos;
    if mem i nat_pos
      then index elems (index_mem i nat_pos)
      else zero

let sarray_pts_to
  (#et:Type0) {| d : scalar et |} #len
  (a : sarray et len)
  (#[Tactics.exact (`1.0R)] _f : perm) // TODO: no ignorar, y usarlo
  (s : seq et)
  : slprop
=
  exists* v_elems v_pos.
    a.elems |-> v_elems **
    a.pos   |-> v_pos **
    pure (
      valid_pos len v_pos /\
      //a.nnz <= a.len ????
      s == unsparse a.nnz len v_elems v_pos
    )

instance has_pts_to_sarray
  (#et: eqtype) (#len : nat) {| scalar et |}
  : has_pts_to (sarray et len) (seq et) =
{
  pts_to = sarray_pts_to;
}

unfold
let sarray_live #et {| scalar et |} #len (s : sarray et len) : slprop =
  exists* v. sarray_pts_to s v
  // (exists* v. s.elems |-> v) **
  // (exists* v. s.pos |-> v)

inline_for_extraction noextract
fn _add1
  (#et : Type0) {| scalar et |}
  (nnz : sz) // número de no-zeros
  (len : erased nat) // longitud "virtual" del array
  (elems : gpu_array et nnz)
  (pos : gpu_array sz nnz)
  // ^ TODO: cambiar size_t a algo más chico?
  preserves gpu
  preserves (exists* v. elems |-> v)
  preserves (exists* v. pos   |-> v)
{
  let mut i = 0sz;
  while (FStar.SizeT.(!i <^ nnz))
    invariant
      (exists* v. i |-> v) **
      (exists* v. elems |-> v)
  {
    let v = gpu_array_read elems !i; // v = elems[i]
    let v' = v `add` one;            // v' = v + 1
    gpu_array_write elems !i v';
    i := !i `SZ.add` 1sz;
  };
}

let _f_u32 = _add1 #u32 #_

inline_for_extraction noextract
fn add1
  (#et : Type0) {| scalar et |}
  (#len : erased nat)
  (a : sarray et len)
  preserves gpu
  preserves sarray_live a
{
  let mut i = 0sz;
  while (FStar.SizeT.(!i <^ a.nnz))
    invariant
      (exists* v. i |-> v) **
      sarray_live a
  {
    with s.
      assert sarray_pts_to a s;
    unfold sarray_pts_to a s;
    let v = gpu_array_read a.elems !i; // v = elems[i]
    let v' = v `add` one;            // v' = v + 1
    gpu_array_write a.elems !i v';
    i := !i `SZ.add` 1sz;
    with s'.
      fold sarray_pts_to a s';
    ()
  };
}

let f_u32 #len = add1 #u32 #_ #len
