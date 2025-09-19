module Kuiper.Example.Sparse

#lang-pulse
open Kuiper
module SZ = FStar.SizeT
module KSeq = Kuiper.Seq.Common

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


(* Producto interno *)

module DP = Kuiper.Poly.DotProduct

let pmul 
  (#et:_) {| scalar et |}
  (#l : nat)
  (s t : lseq et l)
  : GTot (lseq et l)
=
  DP.pmul s t


let sum
  (#et:_) {| scalar et |}
  (#l : nat)
  (s : lseq et l)
  : GTot et
= 
  DP.sum s

let dprod
  (#et:_) {| scalar et |}
  (#l : nat)
  (s t : lseq et l)
  : GTot et
=
  sum (pmul s t)

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
  (#nnz l : nat)
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
      valid_pos l (cast_pos l v_pos) /\
      s == unsparse a.nnz l v_elems (cast_pos l v_pos) 
    )

inline_for_extraction noextract
unfold
instance has_pts_to_sarray
  (#et: Type0) (#l : nat) {| scalar et |}
  : has_pts_to (sarray et l) (seq et) =
{
  pts_to = sarray_pts_to;
}


(* Ejemplos *)


(* sarray_id: lee y escribe el arreglo sin modificarlo *)

inline_for_extraction noextract
fn sarray_id
  (#et : Type0) {| scalar et |}
  (l : erased nat)
  (a : sarray et l)
  (#s0 : erased (lseq et l))
  preserves gpu
  preserves sarray_pts_to a s0
{
  let mut i = 0sz;
  while ((!i <^ a.nnz))
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


(* scale_sarray: producto escalar *)

let scale_seq
  (#et:_) {| d : scalar et |}
  (k : et)
  (#l : nat)
  (s : lseq et l)
  : seq et
=
  Seq.map_seq (mul k) s

let scale_unsparse
  (#et:_) {| scalar et |}
  (k : et)
  (#nnz #l : nat)
  (elems : lseq et nnz)
  (pos   : lseq nat nnz{valid_pos l pos})
  : Lemma
    (requires true)
    (ensures scale_seq k (unsparse nnz l elems pos) == unsparse nnz l (scale_seq k elems) pos)
=
  let open FStar.Seq in
  let s = scale_seq k (unsparse nnz l elems pos) in
  let s' = unsparse nnz l (scale_seq k elems) pos in
  assert (s `equal` s')

inline_for_extraction noextract
fn sarray_scale
  (#et : eqtype) {| scalar et |}
  (k : et)
  (#l : erased nat)
  (a : sarray et l)
  (#s : erased (lseq et l))
  preserves gpu
  requires a |-> s
  ensures  a |-> scale_seq k s
{
  unfold sarray_pts_to a s;

  let mut i = 0sz;

  with v_elems. assert a.elems |-> v_elems;
  with v_pos. assert a.pos |-> v_pos;
  
  while ((!i <^ a.nnz))
    invariant
      (exists* i_v v_elems'. 
        i |-> i_v **
        a.elems |-> v_elems' **
        pure FStar.Seq.(
          len v_elems' == a.nnz /\
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

  scale_unsparse k #a.nnz #l v_elems (cast_pos l v_pos);

  fold sarray_pts_to a (scale_seq k s);
}

let _scale_u32 = sarray_scale #u32 #_



(* producto interno sparse x dense *)


let seq_project
  (#a:_)
  (#nnz #l : nat)
  (pos : lseq nat nnz{valid_pos l pos})
  (s : lseq a l)
  : GTot (lseq a nnz)
=
  let open FStar.Seq in
  // me gustaría escribir:
  // map_seq (index s) pos
  init_ghost nnz fun i ->
    index s (index pos i)


noextract
let rec sum_all_zeros
  (#et : _) {| scalar et |}
  (l : nat)
  (k : et)
  : Lemma
    (requires true)
    (ensures KSeq.seq_fold_left add k (Seq.create #et l zero) == k)
= 
  let open FStar.Seq in
  if l = 0
    then ()
    else (
      sum_all_zeros #et (l - 1) k;
      assert create #et (l -1) zero `equal` tail (create #et l zero)
    )


noextract
let shift
  (#l a b : nat)
  (s : lseq nat l)
  : Ghost (lseq nat l)
    (requires a > 0 /\ b > 0 /\ in_bounds a b s)
    (ensures fun s' -> in_bounds (a - 1) (b -1) s')
=
  Seq.init_ghost l fun i -> let (k : nat) = (s @! i) - 1 in k

noextract
let shift_tail
  (#nnz l : nat{nnz > 0 /\ l > 0})
  (pos : lseq nat nnz{valid_pos l pos})
  : Ghost (lseq nat (nnz -1))
    (requires true)
    (ensures valid_pos (l - 1))
= 
  assert (pos @! 0 >= 0);
  shift 1 l (Seq.tail pos) 


noextract
let rec shift_tail_mem
  (#nnz l : nat{nnz > 0 /\ l > 0})
  (pos : lseq nat nnz{valid_pos l pos})
  (i : nat)
  : Lemma
    (requires true)
    (ensures Seq.mem (i + 1) (Seq.tail pos) <==> Seq.mem i (shift_tail l pos))
=
  let open FStar.Seq in
  
  let pos' = tail pos in
  if len pos' = 0
    then ()
    else (
      assert shift_tail #(nnz - 1) l (tail pos) `equal` tail (shift_tail l pos);
      shift_tail_mem #(nnz - 1) l (tail pos) i
    )


noextract
let shift_tail_unsparse
  (#et:_) {| scalar et |}
  (#nnz l : nat{nnz > 0 /\ l > 0})
  (pos : lseq nat nnz{valid_pos l pos})
  (elems : lseq et nnz)
  : Lemma
    (requires pos @! 0 == 0)
    (ensures
      Seq.tail (unsparse nnz l elems pos) ==
      unsparse (nnz - 1) (l - 1) (Seq.tail elems) (shift_tail l pos)
    )
= 
  let open FStar.Seq in

  let pos' = shift_tail l pos in
  let s1 = tail (unsparse nnz l elems pos) in
  let s2 = unsparse (nnz - 1) (l - 1) (tail elems) pos' in

  introduce forall i. s1 @! i == s2 @! i
  with (
    shift_tail_mem l pos i;
    if mem i pos'
      then assert index_mem i pos' == index_mem (i + 1) pos - 1
      else ()
  );
  assert s1 `equal` s2

noextract
let rec shift_mem
  (#nnz l : nat{nnz > 0 /\ l > 0})
  (pos : lseq nat nnz{valid_pos l pos})
  (i : nat)
  : Lemma
    (requires in_bounds 1 l pos)
    (ensures Seq.mem (i + 1) pos <==> Seq.mem i (shift 1 l pos))
= 
  let open FStar.Seq in

  if nnz = 1
    then ()
    else (
      assert tail (shift 1 l pos) `equal` shift #(nnz - 1) 1 l (tail pos);
      shift_mem #(nnz - 1) l (tail pos) i
    )

noextract
let shift_unsparse
  (#et:_) {| scalar et |}
  (#nnz l : nat{nnz > 0 /\ nnz <= l})
  (pos : lseq nat nnz{valid_pos l pos})
  (elems : lseq et nnz)
  : Lemma
    (requires in_bounds 1 l pos)
    (ensures
      Seq.tail (unsparse nnz l elems pos) ==
      unsparse nnz (l - 1) elems (shift 1 l pos))
= 
  let open FStar.Seq in
  
  let s1 = tail (unsparse nnz l elems pos) in
  let s2 = unsparse nnz (l - 1) elems (shift 1 l pos) in
  introduce forall i. s1 @! i == s2 @! i
  with  shift_mem l pos i;
  assert s1 `equal` s2

noextract
let rec lemma_sparse_dprod
  (#et : eqtype) {| scalar et |}
  (#nnz #l : nat)
  (elems : lseq et nnz)
  (pos   : lseq nat nnz{valid_pos l pos})
  (s : lseq et l)
  (k : et)
  : Lemma
    (requires true)
    (ensures
      KSeq.seq_fold_left add k (elems `pmul` seq_project pos s) ==
      KSeq.seq_fold_left add k (unsparse nnz l elems pos `pmul` s))
= 
  let open FStar.Seq in
  
  bounded_from_sorted_in_bounds 0 l pos;

  let p1 = elems `pmul` seq_project pos s in
  let p2 = unsparse _ _ elems pos `pmul` s in
  
  if l = 0
    then ()
    else (
      if nnz = 0
        then (
          assert p2 `equal` create l zero;
          sum_all_zeros #et l k
        )
        else (
          if mem 0 pos
            then (
              let (k' : et) = k `add` (p1 @! 0) in
              assert k' == k `add` (p2 @! 0);
              let pos' = shift_tail l pos in
              shift_tail_unsparse l pos elems;
              assert tail p1 `equal` (tail elems `pmul` seq_project  #_ #(nnz - 1) #(l - 1) pos' (tail s));
              assert tail p2 `equal` (unsparse (nnz - 1) (l -1) (tail elems) pos' `pmul` tail s);
              lemma_sparse_dprod #_ #_ #(nnz - 1) #(l - 1) (tail elems) pos' (tail s) k'
            )
            else (
              let pos' = shift 1 l pos in
              shift_unsparse l pos elems;
              assert p1 `equal` (elems `pmul` seq_project #_ #nnz #(l - 1) pos' (tail s));
              assert tail p2 `equal` (unsparse nnz (l - 1) elems pos' `pmul` tail s);
              lemma_sparse_dprod #_ #_ #nnz #(l - 1) elems pos' (tail s) k
            )
        )
    )

inline_for_extraction noextract
fn sarray_product_dense
  (#et : eqtype) {| scalar et |}
  (#l : erased nat)
  (a : sarray et l)
  (v : gpu_array et l)
  (#s #t : erased (lseq et l))
  preserves gpu ** a |-> s ** v |-> t
  returns
    dp: et
  ensures
    pure (dp == dprod s t)
{
  unfold a |-> s; // que pasa con esto?
  unfold v |-> t;
  unfold sarray_pts_to a s;

  with v_elems. assert a.elems |-> v_elems;
  with v_pos. assert a.pos |-> v_pos;

  let mut i = 0sz;
  let mut dp : et = zero;

  let pos : erased (lseq nat a.nnz) = cast_pos l v_pos;

  while ((!i <^ a.nnz))
    invariant
      (exists* i_v dp_v.
        i |-> i_v **
        dp |-> dp_v **
        pure (
          i_v <= a.nnz /\
          KSeq.seq_fold_left add !dp (seq_drop i_v (v_elems `pmul` seq_project pos t)) ==
          dprod v_elems (seq_project pos t)
        )
      )
  {
    let p = gpu_array_read a.pos !i;
    let x = gpu_array_read a.elems !i;
    let y = gpu_array_read v p;

    dp := !dp `add` (x `mul` y);
    i := !i `SZ.add` 1sz;
  };

  lemma_sparse_dprod v_elems pos t zero; 

  fold sarray_pts_to a s;
  fold v |-> t;

  !dp;
}

let product_dense_u32 #l = sarray_product_dense #u32 #_ #l


(* sarray_product: producto intero sparse x sparse *)

inline_for_extraction noextract
fn sarray_product_quadratic
  (#et : eqtype) {| scalar et |}
  (#l : erased nat)
  (a b : sarray et l)
  (#s #t : erased (lseq et l))
  preserves gpu ** a |-> s ** b |-> t
  requires
    emp
  returns
    dp: et
  ensures
    emp //pure (dp == dprod s t)
{
  unfold sarray_pts_to a s;
  unfold sarray_pts_to b t;

  let mut dp : et = zero;

  let mut i = 0sz;
  while ((!i <^ a.nnz))
    invariant live i ** live dp
  {
    let mut j = 0sz;
    let p_a = gpu_array_read a.pos !i;
    while ((!j <^ b.nnz))
      invariant live j ** live dp
    {
      let p_b = gpu_array_read b.pos !j;
      if (p_a = p_b) {
        let x = gpu_array_read a.elems !i;
        let y = gpu_array_read b.elems !j;
        dp := !dp `add` (x `mul` y);
        j := !j `SZ.add` 1sz;
      } else {
        j := !j `SZ.add` 1sz;
      };
    };
    i := !i `SZ.add` 1sz;
  };


  fold sarray_pts_to a s;
  fold sarray_pts_to b t;

  !dp;
}

let product_sparse_quadratic_u32 #l = sarray_product_quadratic #u32 #_ #l

inline_for_extraction noextract
fn sarray_product
  (#et : eqtype) {| scalar et |}
  (#l : erased nat)
  (a b : sarray et l)
  (#s #t : erased (lseq et l))
  preserves gpu ** a |-> s ** b |-> t
  requires
    emp
  returns
    dp: et
  // ensures
  //   pure (dp == dprod s t)
{
  unfold sarray_pts_to a s;
  unfold sarray_pts_to b t;

  let mut dp : et = zero;

  let mut i = 0sz;
  let mut j = 0sz;
  while ((!i <^ a.nnz && !j <^ b.nnz))
    invariant live i ** live j
    invariant live dp
  {
    // esta lectura podria hacerse una sola vez
    let p_a = gpu_array_read a.pos !i;
    let p_b = gpu_array_read b.pos !j;
    if ((p_a <^ p_b)) {
      i := !i `SZ.add` 1sz
    } else if ((p_b <^ p_a)) {
      j := !j `SZ.add` 1sz;
    } else {
      let x = gpu_array_read a.elems !i;
      let y = gpu_array_read b.elems !j;
      dp := !dp `add` (x `mul` y);
      i := !i `SZ.add` 1sz;
      j := !j `SZ.add` 1sz;
    };
  };


  fold sarray_pts_to a s;
  fold sarray_pts_to b t;

  !dp;
}

let product_sparse_u32 #l = sarray_product #u32 #_ #l

type sarray_iterator
  (#et : Type0) (#l : erased nat)
  (a : sarray et l) =
{
  i   : (i   : sz{i <= a.nnz}); // índice en elems
  // pos : (pos : sz{pos <= l});   // posición dentro del array "virtual"
}

fn sarray_iterator_init
  (#et : Type0) {| scalar et |}
  (#l : erased nat)
  (a : sarray et l)
  (s : erased (seq et))
  // requires pure (a.nnz > 0)
  preserves gpu
  preserves a |-> s
  returns sarray_iterator a
{
  // unfold sarray_pts_to a s;
  // let p = gpu_array_read a.pos 0sz;
  // fold sarray_pts_to a s;
  let r : sarray_iterator a = { i = 0sz; (* pos = p; *) };
  r
}

let sarray_iterator_end
  (#et : Type0) (#l : erased nat)
  (#a : sarray et l)
  (i : sarray_iterator a)
  : bool =
  i.i = a.nnz

fn sarray_iterator_get
  (#et : Type0) {| scalar et |}
  (#l : erased nat)
  (#a : sarray et l)
  (i : sarray_iterator a)
  (#s : erased (seq et))
  preserves gpu
  preserves a |-> s
  requires pure (not (sarray_iterator_end i))
  returns
    et & sz
{
  let i = i.i;
  unfold sarray_pts_to a s;
  let v = gpu_array_read a.elems i;
  let p = gpu_array_read a.pos i;
  fold sarray_pts_to a s;
  (v, p)
}

fn sarray_iterator_next
  (#et : Type0) (#l : erased nat)
  (#a : sarray et l)
  (i : sarray_iterator a)
  requires pure (not (sarray_iterator_end i))
  returns
    sarray_iterator a
{
  admit();
}
