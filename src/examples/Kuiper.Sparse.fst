module Kuiper.Sparse

#lang-pulse
open Kuiper
module SZ = FStar.SizeT

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
type sarray (et : Type0)
  (l : erased nat) =
  // ^ longitud "virtual" del array
{
  nnz   : sz; // número de no-zeros
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

let sarray_pts_to
  (#et:Type0) {| d : scalar et |} #l
  (a : sarray et l)
  (#[Tactics.exact (`1.0R)] f : perm)
  (s : seq et)
  : slprop
=
  exists* v_elems (v_pos : lseq sz a.nnz).
    a.elems |-> Frac f v_elems **
    a.pos   |-> Frac f v_pos **
    pure (
      valid_pos l (cast_pos v_pos) /\
      s == unsparse a.nnz l v_elems (cast_pos v_pos) 
    )

inline_for_extraction noextract
unfold
instance has_pts_to_sarray
  (#et: Type0) (#l : nat) {| scalar et |}
  : has_pts_to (sarray et l) (seq et) =
{
  pts_to = sarray_pts_to;
}

(* iterador sobre array esparso *)

type sarray_iterator
  (#et : Type0) (#l : erased nat)
  (a : sarray et l) =
{
  i   : (i   : sz{i <= a.nnz}); // índice en elems
  // pos : (pos : sz{pos <= l});   // posición dentro del array "virtual"
}

let sarray_iterator_pts_to
  (#et:Type0) {| d : scalar et |}
  (#l : erased nat)
  (#a : sarray et l)
  (i : sarray_iterator a)
  (#[Tactics.exact (`1.0R)] f : perm)
  (elem : nat & et)
  : slprop
=
  exists* (v_elems : lseq et a.nnz) (v_pos : lseq sz a.nnz).
    a.elems |-> Frac f v_elems **
    a.pos   |-> Frac f v_pos **
    pure (
      i.i < a.nnz /\
      elem == (SZ.v (v_pos @! i.i), v_elems @! i.i)
    )
    

inline_for_extraction noextract
unfold
instance has_pts_to_sarray_iterator
  (#et: Type0) {| scalar et |}
  (#l : erased nat)
  (#a : sarray et l) 
  : has_pts_to (sarray_iterator a) (nat & et) =
{
  pts_to = sarray_iterator_pts_to
}


fn sarray_iterator_init
  (#et : Type0) {| scalar et |}
  (#l : erased nat)
  (a : sarray et l)
  (#s : erased (seq et))
  // requires pure (a.nnz > 0)
  preserves gpu
  preserves a |-> s
  returns i : sarray_iterator a
{
  let i : sarray_iterator a = { i = 0sz;};
  i
}

let sarray_iterator_end
  (#et : Type0) (#l : erased nat)
  (#a : sarray et l)
  (i : sarray_iterator a)
  : bool =
  i.i = a.nnz


fn sarray_iterator_is_done
  (#et : Type0) {| scalar et |}
  (#l : erased nat)
  (#a : sarray et l)
  (i : sarray_iterator a)
  (#s : erased (seq et))
  preserves gpu ** a |-> s
  returns is_done : bool
  ensures pure (is_done == sarray_iterator_end i)
{
  (i.i = a.nnz)
}

fn sarray_iterator_get
  (#et : Type0) {| scalar et |}
  (#l : erased nat)
  (#a : sarray et l)
  (#s : erased (seq et))
  (i : sarray_iterator a)
  (#p : sz)
  (#e : et)
  preserves gpu ** a |-> s
  requires
    pure (not (sarray_iterator_end i))
  returns v : sz & et 
{
  unfold sarray_pts_to a s;

  let v = gpu_array_read a.elems i.i;
  let p = gpu_array_read a.pos i.i;

  fold sarray_pts_to a s;
  (p, v)
}

fn sarray_iterator_next
  (#et : Type0) {| scalar et |}
  (#l : erased nat)
  (#a : sarray et l)
  (i : sarray_iterator a)
  preserves gpu
  requires
    pure (not (sarray_iterator_end i))
  returns sarray_iterator a
{
  let i' : sarray_iterator a = {i = i.i +^ 1sz};
  i'
}

// sparse matrix

open Kuiper.EMatrix

noeq
type smatrix (et : Type0)
  (rows cols : erased nat) =
{
  nnz       : sz; // número de no-zeros
  elems     : gpu_array et nnz; // elementos (no zero)
  col_ind   : gpu_array sz nnz; // columna de cada elemento
  row_off   : gpu_array sz (rows + 1); // posición de cada elemento
}


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
  // índices de columnas están dentro de rango
  in_bounds 0 cols col_ind /\
  // índices de columna están ordenados (por fila)
  (forall (i : natlt rows).
    let cols = Seq.slice col_ind (row_off @! i) (row_off @! (i + 1)) in
    forall j k. j < k ==> cols @! j < cols @! k
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
    let cols = Seq.slice col_ind (row_off @! i) (row_off @! (i + 1)) in
    if Seq.mem j cols
      then elems @! Seq.index_mem j cols
      else zero

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
    pure (
      valid_smatrix rows cols (cast_pos v_col_ind) (cast_pos v_row_off) /\
      e == smatrix_unsparse rows cols v_elems (cast_pos v_col_ind) (cast_pos v_row_off) 
    )

inline_for_extraction noextract
unfold
instance has_pts_to_smatrix
  (#et: Type0) (#rows #cols : nat) {| scalar et |}
  : has_pts_to (smatrix et rows cols) (ematrix et rows cols) =
{
  pts_to = smatrix_pts_to;
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
    
  };
}