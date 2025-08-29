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
fn sarray_id
  (#et : Type0) {| scalar et |}
  (len : erased nat)
  (a : sarray et len)
  preserves gpu
  preserves (exists* s. sarray_pts_to a s)
{
  with s.
    assert(sarray_pts_to a s);
  let mut i = 0sz;
  while(FStar.SizeT.(!i <^ a.nnz))
    invariant
      (sarray_pts_to a s) **
      (exists* i_v. i |-> i_v)
  {
    unfold (sarray_pts_to a s);
    with v_elems. assert a.elems |-> v_elems;

    let v = gpu_array_read a.elems !i;
    gpu_array_write a.elems !i v;

    with i_v.
      assert i |-> i_v;
      assert pure (Seq.equal v_elems (Seq.upd v_elems i_v v));

    i := !i `SZ.add` 1sz;
    
    fold(sarray_pts_to a s);
  }
}

let _id_u32 = sarray_id #u32 #_

assume
val zero_is_id
  (#et:_) {| d : scalar et |}
  (k : et)
  : Lemma
    (requires true)
    (ensures k `d.mul` zero == zero /\ zero `d.mul` k == zero)

let scale_seq
  (#et:_) {| d : scalar et |}
  (k : et)
  (#len : nat)
  (s : lseq et len)
  : Pure (lseq et len)
    (requires true)
    (ensures fun s' -> forall i. Seq.index s' i == k `d.mul` Seq.index s i)
=
  let open FStar.Seq in

  map_seq_len (mul k) s;
  let s' = map_seq (mul k) s in
  introduce forall i. index s' i == k `mul` index s i
  with map_seq_index (mul k) s i;
  s'

let scale_unsparse
  (#et:_) {| scalar et |}
  (k : et)
  (#nnz #len : nat)
  (elems : lseq et nnz)
  (pos   : lseq sz nnz{valid_pos len pos})
  : Lemma
    (requires true)
    (ensures scale_seq k (unsparse nnz len elems pos) == unsparse nnz len (scale_seq k elems) pos)
=
  let open FStar.Seq in

  let (nat_pos : seq nat) = map_seq SZ.v pos in
  map_seq_len SZ.v pos;

  let s = scale_seq k (unsparse nnz len elems pos) in
  let s' = unsparse nnz len (scale_seq k elems) pos in
  
  introduce forall (i : nat{i < len}). Seq.index s i == Seq.index s' i
  with
    if Seq.mem i nat_pos
      then ()
      else zero_is_id k;

  assert (s `equal` s')

inline_for_extraction noextract
fn sarray_scale
  (#et : eqtype) {| scalar et |}
  (k : et)
  (len : erased nat)
  (a : sarray et len)
  (s : erased (lseq et len))
  preserves gpu
  requires sarray_pts_to a s
  ensures sarray_pts_to a (scale_seq k s)
{
  unfold sarray_pts_to a s;

  let mut i = 0sz;

  with v_elems. assert a.elems |-> v_elems;
  with v_pos. assert a.pos |-> v_pos;
  
  while(FStar.SizeT.(!i <^ a.nnz))
    invariant
      (exists* i_v v_elems'. 
        i |-> i_v **
        a.elems |-> v_elems' **
        pure FStar.Seq.(
          FStar.SizeT.(i_v <=^ a.nnz) /\ 
          length v_elems' == a.nnz /\
          forall (i : nat{i < a.nnz}).
            (i < (SZ.v i_v) ==> index v_elems' i == k `mul` index v_elems i) /\
            (i >= (SZ.v i_v) ==> index v_elems' i == index v_elems i)))
  {
    let v = gpu_array_read a.elems !i;
    gpu_array_write a.elems !i (k `mul` v);
    i := !i `SZ.add` 1sz;
  };

  with v_elems'. assert a.elems |-> v_elems';

  assert pure FStar.Seq.(v_elems' `equal` scale_seq k v_elems);

  scale_unsparse k #a.nnz #len v_elems v_pos;

  fold sarray_pts_to a (scale_seq k s);
}

let _scale_u32 = sarray_scale #u32 #_

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
      fold sarray_pts_to a s'
  };
}

let f_u32 #len = add1 #u32 #_ #len
