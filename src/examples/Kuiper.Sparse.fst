module Kuiper.Sparse

#lang-pulse
open Kuiper
module SZ = FStar.SizeT

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
  forall i. {:pattern (s @! i)} l <= s @! i /\ s @! i < h

noextract
let sorted_slice
  (s : seq nat)
  (a b : nat{a <= b /\ b <= len s})
  : prop
=
  forall i j. {:pattern (s @! i); (s @! j)} a <= i /\ i < j /\ j < b ==> s @! i < s @! j


noextract
let sorted (s : seq nat) : prop =
  sorted_slice s 0 (len s)

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

let is_global_smatrix
  (#et:Type0) {| scalar et |}
  (#rows #cols : nat)
  (m : smatrix et rows cols)
  : prop
  = is_global_array m.elems
    /\ is_global_array m.col_ind
    /\ is_global_array m.row_off

let valid_smatrix
  (#nnz rows cols : nat)
  (col_ind : lseq nat nnz)
  (row_off : lseq nat (rows + 1))
  : prop
=
  // los offsets de fila están ordenados y dentro de rango
  (row_off @! 0 == 0) /\
  (row_off @! rows == nnz) /\
  (forall i j. {:pattern (row_off @! i); (row_off @! j)} i < j ==> row_off @! i <= row_off @! j) /\
  // indices de columna son posiciones validas por cada fila
  (in_bounds 0 cols col_ind) /\
  (forall (i : natlt rows).
    sorted_slice col_ind (row_off @! i) (row_off @! (i + 1))
  )

let rec mem_slice
  (#et : eqtype)
  (x : et) (s : seq et)
  (a b : nat {a <= b /\ b <= len s})
  : Pure bool (decreases (b - a))
    (requires true)
    (ensures fun r -> forall i. a <= i /\ i < b /\ x == s @! i ==> r)
=
  if a < b
    then (s @! a) = x || mem_slice x s (a + 1) b
    else false

let rec index_mem_slice
  (#et : eqtype)
  (x : et) (s : seq et)
  (a b : nat {a <= b /\ b <= len s})
  : Pure nat
    (requires (mem_slice x s a b))
    (ensures (fun i -> a <= i /\ i < b /\ s @! i == x))
    (decreases (b - a))
=
  if (s @! a) = x
    then a
    else index_mem_slice x s (a + 1) b

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
    //let row_cols = slice_row row_off col_ind i in
    //let row_elems = slice_row row_off elems i in
    //unsparse _ cols row_elems row_cols @! j
    let ri = row_off @! i in
    let re = row_off @! (i + 1) in
    if mem_slice j col_ind ri re
      then elems @! index_mem_slice j col_ind ri re
      else zero

let smatrix_all_zeros
  (#et:Type0) {| scalar et |}
  (#nnz rows cols : nat)
  (elems : lseq et nnz)
  (col_ind : lseq nat nnz)
  (row_off : lseq nat (rows + 1))
  (i : natlt rows)
  (r : nat{(row_off @! i) < r /\ r < (row_off @! (i + 1))})
  : Lemma
    (requires valid_smatrix rows cols col_ind row_off)
    (ensures
      forall j. (col_ind @! r - 1) < j /\ j < col_ind @! r ==>
      macc (smatrix_unsparse rows cols elems col_ind row_off) i j == zero)
= ()


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


unfold
let smatrix_pts_to'
  (#et:Type0) {| d : scalar et |}
  #rows #cols
  (m : smatrix et rows cols)
  (#[Tactics.exact (`1.0R)] f : perm)
  (v_elems   : lseq et m.nnz)
  (v_col_ind : lseq sz m.nnz)
  (v_row_off : lseq sz (rows + 1))
  (e : ematrix et rows cols)
  : slprop
=
  m.elems   |-> Frac f v_elems **
  m.col_ind |-> Frac f v_col_ind **
  m.row_off |-> Frac f v_row_off **
  pure (pure_smatrix_pt_to m e v_elems v_col_ind v_row_off)



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


ghost
fn unfold_smatrix
  (#et:Type0) {| d : scalar et |}
  #rows #cols
  (m : smatrix et rows cols)
  (#[Tactics.exact (`1.0R)] f : perm)
  (e : ematrix et rows cols)
  requires smatrix_pts_to m #f e
  ensures
    exists* (v_elems    : lseq et m.nnz).
    exists* (v_col_ind  : lseq sz m.nnz).
    exists* (v_row_off  : lseq sz (rows + 1)).
      smatrix_pts_to' m #f v_elems v_col_ind v_row_off e
{
  unfold smatrix_pts_to m #f e
}

ghost
fn fold_smatrix
  (#et:Type0) {| d : scalar et |}
  #rows #cols
  (m : smatrix et rows cols)
  (#[Tactics.exact (`1.0R)] f : perm)
  (v_elems    : lseq et m.nnz)
  (v_col_ind  : lseq sz m.nnz)      
  (v_row_off  : lseq sz (rows + 1))
  (e : ematrix et rows cols)
  requires smatrix_pts_to' m v_elems v_col_ind v_row_off e
  ensures smatrix_pts_to m e
{
  fold smatrix_pts_to m e
}

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
fn smatrix_share_n'
  (#et:Type0) {| d : scalar et |}
  #rows #cols
  (m : smatrix et rows cols)
  (#[Tactics.exact (`1.0R)] f : perm)
  (v_elems   : lseq et m.nnz)
  (v_col_ind : lseq sz m.nnz)
  (v_row_off : lseq sz (rows + 1))
  (em : ematrix et rows cols)
  (k : pos)
  requires smatrix_pts_to' m #f v_elems v_col_ind v_row_off em
  ensures forall+ (_ : natlt k).
    smatrix_pts_to' m #(f /. k) v_elems v_col_ind v_row_off  em
{
  gpu_slice_share m.elems _ _ k;
  gpu_slice_share m.col_ind _ _ k;
  gpu_slice_share m.row_off _ _ k;

  forevery_zip
    (fun _ -> gpu_pts_to_array m.col_ind #(f /. k) v_col_ind)
    (fun _ -> gpu_pts_to_array m.row_off #(f /. k) v_row_off);
  forevery_zip
    (fun _ -> gpu_pts_to_array m.elems #(f /.k) v_elems) _;

  forevery_map #(natlt k)
    (fun _ ->
      gpu_pts_to_slice m.elems #(f /. k) 0 (SZ.v m.nnz) v_elems **
      gpu_pts_to_slice m.col_ind #(f /. k)0 (SZ.v m.nnz) v_col_ind **
      gpu_pts_to_slice m.row_off #(f /. k) 0 (rows + 1) v_row_off
    ) 
    (fun _ -> smatrix_pts_to' m #(f /. k) v_elems v_col_ind v_row_off em)
    fn _ {};

}

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
    forall+ (_ : natlt k).
      smatrix_pts_to m #(f /. k) em
{
  unfold smatrix_pts_to m #f em;
  with v_elems.
    assert gpu_pts_to_array m.elems #f v_elems;
  with v_col_ind.
    assert gpu_pts_to_array m.col_ind #f v_col_ind;
  with v_row_off.
    assert gpu_pts_to_array m.row_off #f v_row_off;

  smatrix_share_n' m #f _ _ _ em k;

  forevery_map #(natlt k)
    (fun _ -> smatrix_pts_to' m #(f /. k) v_elems v_col_ind v_row_off em)
    (fun _ -> smatrix_pts_to m #(f /. k) em)
    fn _ { fold smatrix_pts_to m #(f /. k) em; };

}


let forall_natlt_elim (n : pos) (p : prop)
: Lemma (requires forall (_ : natlt n). p) (ensures p)
= eliminate forall (_ : natlt n). p with 0

ghost
fn forevery_natlt_elim
  (n : pos) (p : prop)
  requires forall+ (_ : natlt n). pure p
  ensures pure p
{
  forevery_extract_pure #(natlt n)
    (fun _ -> pure p) (fun _ -> p) fn _ {};

  forall_natlt_elim n p;

  forevery_map #(natlt n) (fun _ -> pure p) (fun _ -> emp) fn _ {}; 
  forevery_emp_elim _;

}

ghost
fn smatrix_gather_n'
  (#et:Type0) {| d : scalar et |}
  #rows #cols
  (m : smatrix et rows cols)
  (#[Tactics.exact (`1.0R)] f : perm)
  (v_elems   : lseq et m.nnz)
  (v_col_ind : lseq sz m.nnz)
  (v_row_off : lseq sz (rows + 1))
  (em : ematrix et rows cols)
  (k : pos)
  requires forall+ (_ : natlt k).
    smatrix_pts_to' m #(f /. k) v_elems v_col_ind v_row_off  em
  ensures smatrix_pts_to' m #f v_elems v_col_ind v_row_off em
{
  forevery_unzip #(natlt k) _ _;
  forevery_unzip #(natlt k) _ _;
  forevery_unzip #(natlt k) _ _;

  gpu_slice_gather m.elems   _ _ k;
  gpu_slice_gather m.col_ind _ _ k;
  gpu_slice_gather m.row_off _ _ k;

  forevery_natlt_elim k _;

  ();
}

// Para escribir esto en terminos de smatrix_gather_n'
// tendriamos que probar smatrix_pts_to_eq'
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
  forevery_natlt_pop k _;
  unfold smatrix_pts_to m #(f /. k) em;
  with v_elems.   assert gpu_pts_to_array m.elems   #(f /. k) v_elems;
  with v_col_ind. assert gpu_pts_to_array m.col_ind #(f /. k) v_col_ind;
  with v_row_off. assert gpu_pts_to_array m.row_off #(f /. k) v_row_off;

  ghost
  fn aux (_ : natlt (k-1))
    norewrite
    preserves
      gpu_pts_to_array m.elems   #(f /. k) v_elems **
      gpu_pts_to_array m.col_ind #(f /. k) v_col_ind **
      gpu_pts_to_array m.row_off #(f /. k) v_row_off
    requires
      smatrix_pts_to m #(f /. k) em
    ensures
      gpu_pts_to_array m.elems   #(f /. k) v_elems **
      gpu_pts_to_array m.col_ind #(f /. k) v_col_ind **
      gpu_pts_to_array m.row_off #(f /. k) v_row_off
  {
    unfold smatrix_pts_to m #(f /. k) em;

    gpu_slice_pts_to_eq m.elems 0 m.nnz (f /. k) #_ #v_elems;
    gpu_slice_pts_to_eq m.col_ind 0 m.nnz (f /. k) #_ #v_col_ind;
    gpu_slice_pts_to_eq m.row_off 0 (rows + 1) (f /. k) #_ #v_row_off;
    ()
  };

  forevery_map_extra _ _ _ aux;
  forevery_natlt_push k _;

  forevery_unzip #(natlt k) _ _;
  forevery_unzip #(natlt k) _ _;

  gpu_slice_gather m.elems   _ _ k;
  gpu_slice_gather m.col_ind _ _ k;
  gpu_slice_gather m.row_off _ _ k;

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
  while (!i <^ m.nnz)
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
