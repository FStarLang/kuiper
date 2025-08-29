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
  forall (i j : natlt nnz). i > j ==> Seq.index s i > Seq.index s j

let valid_pos (#nnz len : nat) (s : lseq sz nnz) : prop =
  in_bounds len s /\ no_repeats s

let unsparse
  (#et:Type0) {| scalar et |}
  (nnz len : nat)
  (elems : lseq et nnz)
  (pos   : lseq sz nnz{valid_pos len pos}) 
  : GTot (lseq et len)
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
  (#[Tactics.exact (`1.0R)] f : perm)
  (s : seq et)
  : slprop
=
  exists* v_elems (v_pos : lseq sz a.nnz).
    a.elems |-> Frac f v_elems **
    a.pos   |-> Frac f v_pos **
    pure (
      valid_pos len v_pos /\
      //a.nnz <= a.len ????
      s == unsparse a.nnz len v_elems v_pos
    )

inline_for_extraction noextract
unfold
instance has_pts_to_sarray
  (#et: Type0) (#len : nat) {| scalar et |}
  : has_pts_to (sarray et len) (seq et) =
{
  pts_to = sarray_pts_to;
}

inline_for_extraction noextract
fn sarray_id
  (#et : Type0) {| scalar et |}
  (len : erased nat)
  (a : sarray et len)
  (#s0 : erased (lseq et len))
  preserves gpu
  preserves sarray_pts_to a s0
{
  let mut i = 0sz;
  while (FStar.SizeT.(!i <^ a.nnz))
    invariant a |-> s0 ** live i
  {
    unfold sarray_pts_to a s0;
    with v_elems. assert a.elems |-> v_elems;

    let v = gpu_array_read a.elems !i;
    gpu_array_write a.elems !i v;

    with i_v.
      assert i |-> i_v;
      assert pure (Seq.equal v_elems (Seq.upd v_elems i_v v));

    i := !i `SZ.add` 1sz;
    
    fold sarray_pts_to a s0;
  }
}

let _id_u32 = sarray_id #u32 #_

assume
val zero_is_absorbing_l
  (#et:_) {| scalar et |}
  (k : et)
  : Lemma
    (requires true)
    (ensures k `mul` zero == zero)
    [SMTPat (k `mul` zero)]
    // FIXME: ^ this pattern does not kick in
    // if we use `d.mul` instead of `mul`. Why?

assume
val zero_is_absorbing_r
  (#et:_) {| scalar et |}
  (k : et)
  : Lemma
    (requires true)
    (ensures zero `mul` k == zero)
    [SMTPat (zero `mul` k )]

let map_seq_len (#a #b:Type) (f:a -> Tot b) (s:Seq.seq a)
  : Lemma (ensures Seq.length (Seq.map_seq f s) == Seq.length s)
          [SMTPat (Seq.map_seq f s)]
  = Seq.map_seq_len f s

let my_map_seq_index (#a #b:Type) (f:a -> Tot b) (s:Seq.seq a) (i:nat{i < Seq.length s})
  : Lemma (ensures (Seq.map_seq_len f s; Seq.index (Seq.map_seq f s) i == f (Seq.index s i)))
          [SMTPat (Seq.index (Seq.map_seq f s) i)]
  = Seq.map_seq_index f s i

let scale_seq
  (#et:_) {| d : scalar et |}
  (k : et)
  (#len : nat)
  (s : lseq et len)
  : seq et
=
  Seq.map_seq (mul k) s

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
  let s = scale_seq k (unsparse nnz len elems pos) in
  let s' = unsparse nnz len (scale_seq k elems) pos in
  assert (s `equal` s')

inline_for_extraction noextract
fn sarray_scale
  (#et : eqtype) {| scalar et |}
  (k : et)
  (len : erased nat)
  (a : sarray et len)
  (#s : erased (lseq et len))
  preserves gpu
  requires a |-> s
  ensures  a |-> scale_seq k s
{
  unfold sarray_pts_to a s;

  let mut i = 0sz;

  with v_elems. assert a.elems |-> v_elems;
  with v_pos. assert a.pos |-> v_pos;
  
  while (FStar.SizeT.(!i <^ a.nnz))
    invariant
      (exists* i_v v_elems'. 
        i |-> i_v **
        a.elems |-> v_elems' **
        pure FStar.Seq.(
          // FStar.SizeT.(i_v <=^ a.nnz) /\ 
          length v_elems' == a.nnz /\
          forall (j : nat{j < a.nnz}).
            (j <  i_v ==> index v_elems' j == k `mul` index v_elems j) /\
            (j >= i_v ==> index v_elems' j == index v_elems j)))
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
  preserves live a
{
  let mut i = 0sz;
  while (FStar.SizeT.(!i <^ a.nnz))
    invariant
      (exists* v. i |-> v) **
      live a
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
