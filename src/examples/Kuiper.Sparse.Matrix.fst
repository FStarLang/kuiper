module Kuiper.Sparse.Matrix

#lang-pulse
open Kuiper
open Kuiper.Sparse.Common
open Kuiper.EMatrix
module SZ = Kuiper.SizeT

// This is here to force extraction.
let _ = 1ul

// sparse matrix


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
    (ensures fun r -> (exists i. a <= i /\ i < b /\ x == s @! i) <==> r)
=
  if a < b
    then s @! a = x || mem_slice x s (a + 1) b
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
  if s @! a = x
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
let pure_smatrix_pts_to
  (#et:Type0) {| d : scalar et |}
  #rows #cols
  (#nnz : nat)
  (e : ematrix et rows cols)
  (v_elems   : lseq et nnz)
  (v_col_ind : lseq sz nnz)
  (v_row_off : lseq sz (rows + 1))
  : prop
=
  valid_smatrix rows cols (cast_pos v_col_ind) (cast_pos v_row_off) /\
  e == smatrix_unsparse rows cols v_elems (cast_pos v_col_ind) (cast_pos v_row_off)


// TODO quizas quiero una estructura tipo esmatrix que guarde
// v_elems, v_col_ind, v_row_off
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
  pure (pure_smatrix_pts_to e v_elems v_col_ind v_row_off)



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
    pure (pure_smatrix_pts_to e v_elems v_col_ind v_row_off)


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


open Kuiper.Sparse.Array
open Pulse.Lib.Trade
open Kuiper.Seq.Common

// TODO k tiene que ser mayor a 0? tiene que ser menor a n?
let gpu_array_take 
  (#a : Type u#0)
  (#n : erased nat)
  (arr : gpu_array a n)
  (k : szle n) 
: Pure (gpu_array a k)
  (requires true)
  (ensures fun v -> base_address v == base_address arr)
= admit()

let gpu_array_drop 
  (#a : Type u#0)
  (#n : erased nat)
  (arr : gpu_array a n)
  (k : szle n) 
: Pure (gpu_array a (n - k))
  (requires true)
  (ensures fun v -> base_address v == base_address arr + k)
= admit()

ghost
fn gpu_array_cut
  (#a : Type u#0) {| sized a |}
  (#n : nat)
  (arr : gpu_array a n)
  (#[Tactics.exact (`1.0R)] f : perm)
  (k : szle n)
  (#s : lseq a n)
  requires
    arr |-> Frac f s
  ensures
    gpu_array_take arr k |-> Frac f (seq_take k s) **
    gpu_array_drop arr k |-> Frac f (seq_drop k s)
{
  admit()
}

ghost
fn gpu_array_paste
  (#a : Type u#0) {| sized a |}
  (#n : nat)
  (arr : gpu_array a n)
  (#[Tactics.exact (`1.0R)] f : perm)
  (k : szle n)
  (#s : lseq a n)
  requires
    gpu_array_take arr k |-> Frac f (seq_take k s) **
    gpu_array_drop arr k |-> Frac f (seq_drop k s)
  ensures
    arr |-> Frac f s
{
  admit()
}

ghost
fn gpu_array_paste'
  (#a : Type u#0) {| sized a |}
  (#n : nat)
  (arr : gpu_array a n)
  (#[Tactics.exact (`1.0R)] f : perm)
  (k : szle n)
  (#s : lseq a k) (#t : lseq a (n - k))
  requires
    gpu_array_take arr k |-> Frac f s **
    gpu_array_drop arr k |-> Frac f t
  ensures
    arr |-> Frac f (Seq.append s t) 
{
  admit()
}

let rec mem_slice_lemma
  (#et : eqtype)
  (x : et) (s : seq et)
  (a b : nat {a <= b /\ b <= len s})
  : Lemma
    (ensures mem_slice x s a b <==> Seq.mem x (Seq.slice s a b))
    (decreases (b - a))
    [SMTPatOr
      [[SMTPat (mem_slice x s a b)];
       [SMTPat (Seq.mem x (Seq.slice s a b))]]]
=
  if a < b && s @! a <> x
    then mem_slice_lemma x s (a + 1) b
    else ()

let rec index_mem_slice_lemma
  (#et : eqtype)
  (x : et) (s : seq et)
  (a b : nat {a <= b /\ b <= len s})
  : Lemma
    (requires mem_slice x s a b /\ Seq.mem x (Seq.slice s a b))
    (ensures
      s @! index_mem_slice x s a b ==
      Seq.slice s a b @! Seq.index_mem x (Seq.slice s a b)
    )
    (decreases (b - a))
= 
  if a < b && s @! a <> x
    then index_mem_slice_lemma x s (a + 1) b
    else ()



let unsparse_row_lemma
  (#et:Type0) {| scalar et |}
  (#nnz rows cols : nat)
  (elems : lseq et nnz)
  (col_ind : lseq nat nnz)
  (row_off : lseq nat (rows + 1))
  (i : natlt rows)
  : Lemma
    (requires valid_smatrix rows cols col_ind row_off)
    (ensures
      ematrix_row (smatrix_unsparse rows cols elems col_ind row_off) i ==
      unsparse
        ((row_off @! i + 1) - (row_off @! i)) cols
        (Seq.slice elems (row_off @! i) (row_off @! i + 1))
        (Seq.slice col_ind (row_off @! i) (row_off @! i + 1))
    )
=
  let m = smatrix_unsparse rows cols elems col_ind row_off in
  let row = ematrix_row m i in

  let ri = row_off @! i in
  let re = row_off @! (i + 1) in

  let selems = Seq.slice elems ri re in
  let spos = Seq.slice col_ind ri re in
  let s = unsparse (re - ri) cols selems spos in

  introduce forall (j : natlt cols).
    row @! j == s @! j
  with (
    if mem_slice j col_ind ri re
      then (
        mem_slice_lemma j col_ind ri re;
        index_mem_slice_lemma j col_ind ri re;
        ()
      )
      else ()
  );
  assert row `Seq.equal` s

// TODO probablemente no nos interese probar esto
let smatrix_extract_lemma
  (#et:Type0) {| scalar et |}
  (#rows #cols : nat)
  (#nnz : nat)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (rows + 1))
  (e : ematrix et rows cols)
  (r : natlt rows)
  (#row_nnz : nat)
  (row_elems : lseq et row_nnz)
  (row_pos : lseq sz row_nnz)
: Lemma
  (requires
    pure_smatrix_pts_to e elems col_ind row_off /\
    pure_sarray_pts_to cols (ematrix_row e r) row_elems row_pos /\
    row_nnz == (row_off @! r + 1) - (row_off @! r)
  )
  (ensures
    pure_smatrix_pts_to #_ #_ #rows #cols #nnz e
      (Seq.append (Seq.slice elems 0 (row_off @! r)) (Seq.append row_elems (Seq.slice elems (row_off @! r + 1) nnz)))
      (Seq.append (Seq.slice col_ind 0 (row_off @! r)) (Seq.append row_pos (Seq.slice col_ind (row_off @! r + 1) nnz)))
      row_off
  )
=
  admit();
  //e == smatrix_unsparse rows cols v_elems (cast_pos v_col_ind) (cast_pos v_row_off)
  let ri = row_off @! r in
  let re = row_off @! r + 1 in
  let first_elems = Seq.slice elems 0 ri in
  let last_elems = Seq.slice elems re nnz in
  let first_col_ind = Seq.slice col_ind 0 ri in
  let last_col_ind = Seq.slice col_ind re nnz in
  let elems' = Seq.append first_elems (Seq.append row_elems last_elems) in
  let col_ind' : lseq sz nnz = Seq.append first_col_ind (Seq.append row_pos last_col_ind) in
  let e' = smatrix_unsparse rows cols elems' (cast_pos col_ind') (cast_pos row_off) in

  assert valid_smatrix rows cols (cast_pos col_ind') (cast_pos row_off) ;

    //if mem_slice j col_ind ri re
      //then elems @! index_mem_slice j col_ind ri re
      //else zero
  introduce forall (i:natlt rows) (j:natlt cols). macc e i j == macc e' i j
  with (
    let ri' = row_off @! i in
    let re' = row_off @! i + 1 in
    if i < r
      then (
        assert (
          forall (k : nat{ri' <= k /\ k < re'}).
            (cast_pos col_ind') @! k == (cast_pos col_ind) @! k
        );
        if mem_slice j (cast_pos col_ind') ri' re'
          then (
            assert mem_slice j (cast_pos col_ind) ri' re';
            assert
              index_mem_slice j (cast_pos col_ind') ri' re' ==
              index_mem_slice j (cast_pos col_ind) ri' re';
            let k = index_mem_slice j (cast_pos col_ind) ri' re' in
            let k' = index_mem_slice j (cast_pos col_ind') ri' re' in
            admit()
          )
          else 
            assert not (mem_slice j (cast_pos col_ind) ri' re')
      )
      else admit()
  );
  assert e `equal` e';

  ()

ghost
fn gpu_array_share
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#v : seq a)
  requires arr |-> Frac f v
  ensures
    arr |-> Frac (f /. 2) v **
    arr |-> Frac (f /. 2) v
{
  gpu_slice_share arr 0 sz 2;
  forevery_natlt_pop 2 _;
  forevery_natlt_pop 1 _;
  forevery_elim_empty _;
}

ghost
fn gpu_array_gather
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#v : seq a)
  requires
    arr |-> Frac (f /. 2) v **
    arr |-> Frac (f /. 2) v
  ensures arr |-> Frac f v
{
  forevery_intro_empty #(natlt 0) (fun _ -> arr |-> Frac (f /. 2) v);
  forevery_natlt_push_shift 1 _;
  forevery_natlt_push_shift 2 _;
  gpu_slice_gather arr 0 sz 2;
}

fn smatrix_extract
  {| sized sz |} // que onda esto?
  (#et:Type0) {| scalar et |}
  (#rows #cols : szp)
  (m : smatrix et (SZ.v rows) (SZ.v cols))
  (#[Tactics.exact (`1.0R)] f : perm)
  (e : ematrix et rows cols)
  (i : szlt rows)
  requires
    m |-> Frac f e
  returns
    v : sarray et (SZ.v cols)
  ensures
    factored
      (v |-> Frac (f /. 2) (ematrix_row e i))
      (m |-> Frac f e)

{
  
  unfold smatrix_pts_to m #f e;

  with (v_elems : seq _). assert m.elems |-> Frac f v_elems;
  with (v_col_ind : seq sz).  assert m.col_ind |-> Frac f v_col_ind;
  with (v_row_off : seq sz). assert m.row_off |-> Frac f v_row_off;

  let ri : sz = gpu_array_read m.row_off i; 
  let i' : sz = i +^ 1sz;
  // por que falla esto?
  //let re : sz = gpu_array_read m.row_off (i +^ 1sz); 
  let re : sz = gpu_array_read m.row_off i';

  gpu_array_cut m.elems #f ri;
  gpu_array_cut (gpu_array_drop m.elems ri) #f (re -^ ri);

  gpu_array_cut m.col_ind #f ri;
  gpu_array_cut (gpu_array_drop m.col_ind ri) #f (re -^ ri);
  

  let srow : sarray et (SZ.v cols) =  {
    nnz = (re -^ ri);
    elems = gpu_array_take (gpu_array_drop m.elems ri) (re -^ ri);
    pos = gpu_array_take (gpu_array_drop m.col_ind ri) (re -^ ri)
  };

  rewrite each (gpu_array_take (gpu_array_drop m.elems ri) (re -^ ri)) as srow.elems;
  rewrite each (gpu_array_take (gpu_array_drop m.col_ind ri) (re -^ ri)) as srow.pos;

  let v_row_elems : erased (seq et) = seq_take (re - ri) (seq_drop ri v_elems);
  assert srow.elems |-> Frac f v_row_elems;
  assert pure (Seq.equal v_row_elems (Seq.slice v_elems ri re));

  let v_row_pos : erased (seq sz) = seq_take (re - ri) (seq_drop ri v_col_ind);
  assert srow.pos |-> Frac f v_row_pos;
  assert pure (Seq.equal v_row_pos (Seq.slice v_col_ind ri re));

  unsparse_row_lemma rows cols v_elems (cast_pos v_col_ind) (cast_pos v_row_off) i;
  assert pure (
    Seq.equal
      (cast_pos v_row_pos <: lseq nat (re - ri))
      (Seq.slice (cast_pos v_col_ind) ri re)
  ); 

  gpu_array_share srow.elems #f;
  gpu_array_share srow.pos #f;

  intro
    (srow |-> Frac (f /. 2) (ematrix_row e i) @==> m |-> Frac f e)
    #(
      gpu_array_take m.elems ri |-> Frac f (seq_take (v ri) v_elems) **
      gpu_array_drop (gpu_array_drop m.elems ri) (re -^ ri) |->
        Frac f (seq_drop (re - ri) (seq_drop ri v_elems)) **
      gpu_array_take m.col_ind ri |-> Frac f (seq_take (v ri) v_col_ind) **
      gpu_array_drop (gpu_array_drop m.col_ind ri) (re -^ ri) |->
        Frac f (seq_drop (re - ri) (seq_drop ri v_col_ind)) **
      m.row_off |-> Frac f v_row_off **
      srow.elems |-> Frac (f /. 2) (Seq.slice v_elems ri re) **
      srow.pos |-> Frac (f /. 2) (Seq.slice v_col_ind ri re)
    )
    fn _ {
      unfold sarray_pts_to srow #(f /. 2) (ematrix_row e i);

      gpu_slice_pts_to_eq srow.elems 0 srow.nnz (f /. 2) #_ #(Seq.slice v_elems ri re);
      gpu_slice_pts_to_eq srow.pos 0 srow.nnz (f /. 2) #_ #(Seq.slice v_col_ind ri re);

      gpu_array_gather srow.elems;
      gpu_array_gather srow.pos;

      rewrite each srow.elems as (gpu_array_take (gpu_array_drop m.elems ri) (re -^ ri));
      rewrite each srow.pos as (gpu_array_take (gpu_array_drop m.col_ind ri) (re -^ ri));
      rewrite each srow.nnz as (re -^ ri);

      gpu_array_paste (gpu_array_drop m.elems ri) #f (re -^ ri);
      gpu_array_paste (gpu_array_drop m.col_ind ri) #f (re -^ ri);

      gpu_array_paste m.elems #f ri;
      gpu_array_paste m.col_ind #f ri;

      fold smatrix_pts_to m #f e;
    };

  fold sarray_pts_to srow #(f /. 2) (ematrix_row e i);

  srow;
}
