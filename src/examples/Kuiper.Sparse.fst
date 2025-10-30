module Kuiper.Sparse

#lang-pulse
open Kuiper
module SZ = FStar.SizeT
open Kuiper.Sparse.Extra

// This is here to force extraction.
let _ = 1ul

(* Propiedades sobre escalares *)

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

assume
val zero_is_id_l
  (#et:_) {| scalar et |}
  (k : et)
  : Lemma
    (requires true)
    (ensures k `add` zero == k)
    [SMTPat (k `add` zero)]

assume
val zero_is_id_r
  (#et:_) {| scalar et |}
  (k : et)
  : Lemma
    (requires true)
    (ensures zero `add` k == k)
    [SMTPat (zero `add` k)]

(* Secuencias *)

noextract
let seq_drop
  (#a:_) (#l: nat)
  (n : nat{n <= l})
  (s : lseq a l)
  : Ghost (lseq a (l - n))
    (requires true)
    (ensures fun s' -> forall i. s' @! i == s @! (i + n))
=
  Seq.slice s n l


let map_seq_len (#a #b:Type) (f:a -> Tot b) (s:Seq.seq a)
  : Lemma (ensures len (Seq.map_seq f s) == len s)
          [SMTPat (Seq.map_seq f s)]
  = Seq.map_seq_len f s

let my_map_seq_index (#a #b:Type) (f:a -> Tot b) (s:Seq.seq a) (i:nat{i < len s})
  : Lemma (ensures (Seq.map_seq_len f s; Seq.map_seq f s @! i == f (s @! i)))
          [SMTPat (Seq.map_seq f s @! i)]
  = Seq.map_seq_index f s i


(* Propiedades sobre las posiciones de un array esparso *)

noextract
let in_bounds (l h : nat) (s : seq nat) : prop =
  forall i. l <= s @! i /\ s @! i < h

noextract
let sorted (s : seq nat) : prop =
  forall i j. i < j ==> s @! i < s @! j

noextract
let valid_pos (#nnz l : nat) (s : lseq nat nnz) : prop =
  in_bounds 0 l s /\ sorted s


let rec bounded_from_sorted_in_bounds
  (#nnz l h : nat)
  (s : lseq nat nnz)
  : Lemma
    (requires l <= h /\ sorted s /\ in_bounds l h s)
    (ensures nnz + l <= h)
=
  let open FStar.Seq in

  if nnz = 0
    then ()
    else bounded_from_sorted_in_bounds #(nnz - 1) ((s @! 0) + 1) h (tail s)

let cast_pos
  (#nnz : nat)
  (pos : lseq sz nnz)
  : Ghost
    (lseq nat nnz)
    (requires true)
    (ensures fun npos -> forall i. npos @! i == SZ.v (pos @! i))
=
  Seq.map_seq SZ.v pos


(* Sparse array *)
noeq
inline_for_extraction
type sarray (et : Type0)
  (l : erased nat) =
  // ^ longitud "virtual" del array
{
  nnz   : sz; // número de no-zeros
  len   : (len : sz {SZ.v len == reveal l}); // longitud "real" del array virtual
  elems : gpu_array et nnz; // elementos (no zero)
  pos   : gpu_array sz nnz; // posición de cada elemento
}

let unsparse
  (#et:Type0) {| scalar et |}
  (nnz l : nat)
  (elems : lseq et nnz)
  (pos   : lseq nat nnz{valid_pos l pos}) 
  : GTot (lseq et l)
=
  let open FStar.Seq in
  init l fun i ->
    if mem i pos
      then elems @! index_mem i pos
      else zero

let sarray_pts_to'
  (#et:Type0) {| d : scalar et |} (#l : nat)
  (a : sarray et l)
  (#[Tactics.exact (`1.0R)] f : perm)
  (s : seq et)
  (v_elems : lseq et a.nnz)
  (v_pos   : lseq sz a.nnz)
  : slprop
=
    a.elems |-> Frac f v_elems **
    a.pos   |-> Frac f v_pos **
    pure (
      valid_pos l (cast_pos #a.nnz v_pos <: lseq nat a.nnz)
      /\ s == unsparse a.nnz l v_elems (cast_pos v_pos)
    )

let sarray_pts_to
  (#et:Type0) {| d : scalar et |} #l
  (a : sarray et l)
  (#[Tactics.exact (`1.0R)] f : perm)
  (s : seq et)
  : slprop
=
  exists* (v_elems : lseq et a.nnz) (v_pos : lseq sz a.nnz).
    sarray_pts_to' a #f s v_elems v_pos

inline_for_extraction noextract
unfold
instance has_pts_to_sarray
  (#et: Type0) (#l : nat) {| scalar et |}
  : has_pts_to (sarray et l) (seq et) =
{
  pts_to = sarray_pts_to;
}

(* iterador sobre array esparso *)

inline_for_extraction
type sarray_iterator
  (#et : Type0) (#l : erased nat)
  (a : sarray et l) =
{
  i   : (i   : sz{i <= a.nnz}); // índice en elems
}

inline_for_extraction noextract
fn sarray_iterator_init
  (#et : Type0) {| scalar et |}
  (#l : erased nat)
  (a : sarray et l)
  (#f : perm)
  (#s : erased (seq et))
  (#v_elems : erased (seq et){Seq.length v_elems == a.nnz})
  (#v_pos   : erased (seq sz){Seq.length v_pos   == a.nnz})
  preserves gpu
  preserves sarray_pts_to' a #f s v_elems v_pos
  returns i : sarray_iterator #et #l a
  ensures pure (
    forall (j : natlt (Seq.length s)).
      i.i < a.nnz /\ j < v_pos @! i.i ==> s @! j == zero
  )
{
    let i : sarray_iterator a = { i = 0sz };
    unfold sarray_pts_to' a #f s v_elems v_pos;
    fold sarray_pts_to' a #f s v_elems v_pos;
    i;
}

inline_for_extraction noextract
let sarray_iterator_end
  (#et : Type0) (#l : erased nat)
  (#a : sarray et l)
  (i : sarray_iterator a)
  : bool =
  i.i = a.nnz

inline_for_extraction noextract
fn sarray_iterator_get
  (#et : Type0) {| scalar et |}
  (#l : erased nat)
  (#a : sarray et l)
  (#s : erased (seq et))
  (i : sarray_iterator a)
  preserves gpu ** a |-> s
  requires
    pure (not (sarray_iterator_end i))
  returns v : sz & et 
{
  unfold sarray_pts_to a s;
  with v_elems v_pos.
    assert sarray_pts_to' a s v_elems v_pos;
    unfold sarray_pts_to' a s v_elems v_pos;

  let v = gpu_array_read a.elems i.i;
  let p = gpu_array_read a.pos i.i;

  fold sarray_pts_to' a s v_elems v_pos;
  fold sarray_pts_to a s;
  (p, v)
}

inline_for_extraction noextract
fn sarray_iterator_next
  (#et : Type0) {| scalar et |}
  (#l : erased nat)
  (#a : sarray et l)
  (i : sarray_iterator a)
  (#f : perm)
  (#s : erased (seq et))
  (#v_elems : erased (seq et){len v_elems == a.nnz})
  (#v_pos   : erased (seq sz){len v_pos   == a.nnz})
  (#_ : squash (not (sarray_iterator_end i)))
  preserves gpu
  preserves sarray_pts_to' a #f s v_elems v_pos
  requires
    pure (not (sarray_iterator_end i))
  returns i' : sarray_iterator a
  ensures pure (
    forall (j : natlt (len s)).
    v_pos @! i.i < j /\
      (if i'.i = a.nnz then true else j < v_pos @! i'.i)
      ==> s @! j == zero
  )
{
  unfold sarray_pts_to' a #f s v_elems v_pos;
    let i' : sarray_iterator a = {i = i.i +^ 1sz};
    fold sarray_pts_to' a #f s v_elems v_pos;
    i'
}

// sparse matrix

open Kuiper.EMatrix

// CSR
inline_for_extraction
noeq
type smatrix (et : Type0)
  (rows cols : erased nat) =
{
  nnz       : sz; // número de no-zeros
  elems     : gpu_array et nnz; // elementos (no zero)
  col_ind   : gpu_array sz nnz; // columna de cada elemento
  row_off   : gpu_array sz (rows + 1); // posición de cada comienzo de  fila
}

// Medio fea esta
noextract
let slice_row
  #a (#rows #nnz : nat) 
  (row_off : lseq nat (rows + 1){forall k. row_off @! k <= nnz})
  (s : lseq a nnz)
  (i : nat{i < rows /\ row_off @! i <= row_off @! (i + 1)})
  : GTot (lseq a ((row_off @! (i + 1)) - (row_off @! i)))
=
  let ri = row_off @! i in
  let re = row_off @! (i + 1) in
  Seq.slice s ri re

let valid_smatrix
  (#nnz rows cols : nat)
  (col_ind : lseq nat nnz)
  (row_off : lseq nat (rows + 1))
  : prop
=
  // los offsets de fila están ordenados y dentro de rango
  (row_off @! 0 == 0) /\
  (row_off @! rows == nnz) /\
  (forall i j. i < j ==> row_off @! i <= row_off @! j) /\
  // indices de columna son posiciones validas por cada fila
  (forall (i : natlt rows).
    let row_cols = slice_row row_off col_ind i in
    valid_pos cols row_cols 
  )
   

let smatrix_unsparse
  (#et:Type0) {| scalar et |}
  (#nnz rows cols : nat)
  (elems : lseq et nnz)
  (col_ind : lseq nat nnz)
  (row_off : lseq nat (rows + 1)) 
  : Ghost (ematrix et rows cols)
    (requires valid_smatrix rows cols col_ind row_off)
    (ensures fun _ -> true) 
=
  mkM fun i j ->
    let row_cols = slice_row row_off col_ind i in
    let row_elems = slice_row row_off elems i in
    unsparse _ cols row_elems row_cols @! j


unfold
let pure_smatrix_pt_to
  (#et:Type0) {| d : scalar et |}
  #rows #cols
  (m : smatrix et rows cols)
  (#[Tactics.exact (`1.0R)] f : perm)
  (e : ematrix et rows cols)
  (v_elems   : lseq et m.nnz)
  (v_col_ind : lseq sz m.nnz)
  (v_row_off : lseq sz (rows + 1))
  : prop
=
  valid_smatrix rows cols (cast_pos v_col_ind) (cast_pos v_row_off) /\
  e == smatrix_unsparse rows cols v_elems (cast_pos v_col_ind) (cast_pos v_row_off) 


let smatrix_pts_to
  (#et:Type0) {| d : scalar et |}
  #rows #cols
  (m : smatrix et rows cols)
  (#[Tactics.exact (`1.0R)] f : perm)
  (e : ematrix et rows cols)
  : slprop
=
  exists* (v_elems    : lseq et m.nnz).
  exists* (v_col_ind  : lseq sz m.nnz).
  exists* (v_row_off  : lseq sz (rows + 1)).
    m.elems   |-> Frac f v_elems **
    m.col_ind |-> Frac f v_col_ind **
    m.row_off |-> Frac f v_row_off **
    pure (pure_smatrix_pt_to m e v_elems v_col_ind v_row_off)

inline_for_extraction noextract
unfold
instance has_pts_to_smatrix
  (#et: Type0) (#rows #cols : nat) {| scalar et |}
  : has_pts_to (smatrix et rows cols) (ematrix et rows cols) =
{
  pts_to = smatrix_pts_to;
}

module T = FStar.Tactics.V2

ghost
fn smatrix_share_n
  (#et:Type0) {| scalar et |}
  (#[T.exact (`0)]uid: int)
  (#rows #cols : nat)
  (m : smatrix et rows cols)
  (k : pos)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    smatrix_pts_to m #f em
  ensures
    bigstar #uid 0 k (fun _ -> smatrix_pts_to m #(f /. k) em)
{
  unfold smatrix_pts_to m #f em;
  with v_elems.
    assert gpu_pts_to_array m.elems #f v_elems;
  with v_col_ind.
    assert gpu_pts_to_array m.col_ind #f v_col_ind;
  with v_row_off.
    assert gpu_pts_to_array m.row_off #f v_row_off;

  gpu_slice_share #uid m.elems _ _ k;
  gpu_slice_share #uid m.col_ind _ _ k;
  gpu_slice_share #uid m.row_off _ _ k;

  bigstar_zip 0 k
    (fun _ -> gpu_pts_to_array m.col_ind #(f /. k) v_col_ind)
    (fun _ -> gpu_pts_to_array m.row_off #(f /. k) v_row_off);
  bigstar_zip 0 k
    (fun _ -> gpu_pts_to_array m.elems #(f /.k) v_elems) _;

  ghost
  fn aux (i:natlt k)
    requires (
      gpu_pts_to_array m.elems #(f /. k) v_elems **
      gpu_pts_to_array m.col_ind #(f /. k) v_col_ind **
      gpu_pts_to_array m.row_off #(f /. k) v_row_off
    )
    ensures smatrix_pts_to m #(f /. k) em
  {
    fold smatrix_pts_to m #(f /. k) em;
  };
  bigstar_map #0 #uid #0 #k aux;
}

ghost
fn smatrix_gather_n
  (#et:Type0) {| scalar et |}
  (#rows #cols : nat)
  (m : smatrix et rows cols)
  (k : pos)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    forall+ (_ : natlt k). smatrix_pts_to m #(f /. k) em
  ensures
    smatrix_pts_to m #f em
{
  forevery_extract_if_eqtype #(natlt k) 0 _;
  unfold smatrix_pts_to;
  with v_elems.   assert gpu_pts_to_array m.elems   #(f /. k) v_elems;
  with v_col_ind. assert gpu_pts_to_array m.col_ind #(f /. k) v_col_ind;
  with v_row_off. assert gpu_pts_to_array m.row_off #(f /. k) v_row_off;

  ghost
  fn aux (x : natlt k)
    norewrite
    preserves
      gpu_pts_to_array m.elems   #(f /. k) v_elems **
      gpu_pts_to_array m.col_ind #(f /. k) v_col_ind **
      gpu_pts_to_array m.row_off #(f /. k) v_row_off
    requires
      (if op_Equality #(natlt k) x 0 then emp else
        smatrix_pts_to m #(f /. k) em
      )
    ensures
      (if op_Equality #(natlt k) x 0 then emp else
        gpu_pts_to_array m.elems   #(f /. k) v_elems **
        gpu_pts_to_array m.col_ind #(f /. k) v_col_ind **
        gpu_pts_to_array m.row_off #(f /. k) v_row_off
      )
  {
    if (x = 0) {
      rewrite each op_Equality #(natlt k) x 0 as true;
      rewrite emp as
      (if op_Equality #(natlt k) x 0 then emp else
        gpu_pts_to_array m.elems   #(f /. k) v_elems **
        gpu_pts_to_array m.col_ind #(f /. k) v_col_ind **
        gpu_pts_to_array m.row_off #(f /. k) v_row_off
      );
      ();
    } else {
      rewrite each op_Equality #(natlt k) x 0 as false;
      unfold smatrix_pts_to;

      gpu_array_pts_to_eq m.elems;
      gpu_array_pts_to_eq m.col_ind;
      gpu_array_pts_to_eq m.row_off;

      rewrite 
        gpu_pts_to_array m.elems   #(f /. k) v_elems **
        gpu_pts_to_array m.col_ind #(f /. k) v_col_ind **
        gpu_pts_to_array m.row_off #(f /. k) v_row_off
      as
      (if op_Equality #(natlt k) x 0 then emp else
        gpu_pts_to_array m.elems   #(f /. k) v_elems **
        gpu_pts_to_array m.col_ind #(f /. k) v_col_ind **
        gpu_pts_to_array m.row_off #(f /. k) v_row_off
      );
    }
  };

  forevery_map_extra _ _ _ aux;
  forevery_unextract_if_eqtype #(natlt k) 0 _;

  forevery_unzip #(natlt k) _ _;
  forevery_unzip #(natlt k) _ _;

  gpu_slice_gather' m.elems   _ _ k;
  gpu_slice_gather' m.col_ind _ _ k;
  gpu_slice_gather' m.row_off _ _ k;

  fold smatrix_pts_to m #f em;
}

inline_for_extraction noextract
fn smatrix_id
  (#et : Type0) {| scalar et |}
  (rows cols : erased nat)
  (m : smatrix et rows cols)
  (#e : erased (ematrix et rows cols))
  preserves gpu ** m |-> e
{

  let mut i = 0sz;
  while ((!i <^ m.nnz))
    invariant
      m |-> e ** live i 
  {
    unfold smatrix_pts_to m e;

    with v_elems. assert m.elems |-> v_elems;

    let v = gpu_array_read m.elems !i;
    gpu_array_write m.elems !i v;

    with i_v.
      assert i |-> i_v;
    assert pure (Seq.equal v_elems (Seq.upd v_elems i_v v));

    i := !i `SZ.add` 1sz;
    
    fold smatrix_pts_to m e;
    
  }
}

let smatrix_id_u32 = smatrix_id #u32 #_

inline_for_extraction noextract
fn sarray_iterator_test
  (#et : eqtype) {| ets: scalar et |}
  (#l : erased nat)
  (a : sarray et l)
  (#s : erased (seq et))
  preserves gpu ** a |-> s
  ensures emp
{
  unfold sarray_pts_to a;

  with v_elems v_pos.
    assert sarray_pts_to' a s v_elems v_pos;

  let mut it : sarray_iterator #et #l a = sarray_iterator_init a;

  fold sarray_pts_to a s;

  while (not (sarray_iterator_end !it))
    invariant
      live it
  {
    let r = sarray_iterator_get !it;

    unfold sarray_pts_to a s;

    it := sarray_iterator_next #et #ets #l #a !it #1.0R #s;

    fold sarray_pts_to a s;
  };
}

let sarray_iterator_test_u32 #l = sarray_iterator_test #u32 #_ #l
