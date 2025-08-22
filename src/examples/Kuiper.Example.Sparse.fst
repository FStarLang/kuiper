module Kuiper.Example.Sparse

#lang-pulse
open Kuiper
module SZ = FStar.SizeT

let x = 1ul

noeq
type sarray (et : Type0)
  (len : erased nat) =
  // ^ longitud "virtual" del array
{
  nnz   : sz; // número de no-zeros
  elems : gpu_array et nnz; // elementos (no zero)
  pos   : gpu_array sz nnz; // posición de cada elemento
}

let unsparse
  (#et:Type0) {| scalar et |}
  (nnz len : nat)
  // (elems : seq et { Seq.length elems == nnz})
  (elems : lseq et nnz)
  (pos   : lseq sz nnz)
  : seq et
  = magic()

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
    pure (s == unsparse a.nnz len v_elems v_pos)

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
  ();
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
    unfold sarray_pts_to;
    let v = gpu_array_read a.elems !i; // v = elems[i]
    let v' = v `add` one;            // v' = v + 1
    gpu_array_write a.elems !i v';
    i := !i `SZ.add` 1sz;
  };
  ();
}

let f_u32 #len = add1 #u32 #_ #len
