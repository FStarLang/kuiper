module Kuiper.Example.Sparse

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
let seq_take
  (#a:_) (#l: nat)
  (n : nat)
  (s : lseq a l)
  : Ghost (lseq a n)
    (requires n <= l)
    (ensures fun s' -> forall i. s' @! i == s @! i)
=
  Seq.slice s 0 n 

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

noextract
let inits
  (#a:_) (#l : nat{l > 0})
  (s : lseq a l)
  : Ghost (lseq a (l - 1))
    (requires true)
    (ensures fun s' -> forall i. s' @! i == s @! i)
=
  seq_take (l - 1) s

noextract
let last
  (#a:_) (#l : nat{l > 0})
  (s : lseq a l)
  : GTot a
=
  s @! (l - 1)


let map_seq_len (#a #b:Type) (f:a -> Tot b) (s:Seq.seq a)
  : Lemma (ensures len (Seq.map_seq f s) == len s)
          [SMTPat (Seq.map_seq f s)]
  = Seq.map_seq_len f s

let my_map_seq_index (#a #b:Type) (f:a -> Tot b) (s:Seq.seq a) (i:nat{i < len s})
  : Lemma (ensures (Seq.map_seq_len f s; Seq.map_seq f s @! i == f (s @! i)))
          [SMTPat (Seq.map_seq f s @! i)]
  = Seq.map_seq_index f s i


(* Fold *)


let rec seq_fold_left'
  (#a #b : _)
  (f : b -> a -> b)
  (#l : nat)
  (acc : b)
  (s : lseq a l)
  : GTot b
=
  if l = 0
    then acc
    else f (seq_fold_left' f acc (inits s)) (last s)


let rec lemma_fold_drop
  (#a #b : _)
  (f : b -> a -> b)
  (#l : nat{l > 0})
  (acc : b)
  (s : lseq a l)
  (n : nat{n <= l})
  : Lemma 
    (requires n > 0)
    (ensures
      seq_fold_left' f (f acc (s @! (n - 1))) (seq_drop n s) ==
      seq_fold_left' f acc (seq_drop (n - 1) s) 
    )
=
  if n = l
    then ()
    else lemma_fold_drop f acc (inits s) n

let rec lemma_fold_left'
  (#a #b : _)
  (f : b -> a -> b)
  (#l : nat{l > 0})
  (acc : b)
  (s : lseq a l)
  (n : nat{n <= l})
  : Lemma 
    (requires true)
    (ensures
      seq_fold_left' f (seq_fold_left' f acc (seq_take n s)) (seq_drop n s) ==
      seq_fold_left' f acc s
    )
=
  if n = 0
    then ()
    else (
      lemma_fold_left' f acc s (n - 1);
      lemma_fold_drop f (seq_fold_left' f acc (seq_take (n - 1) s)) s n
    )

let rec lemma_fold_left
  (#a #b : _)
  (f : b -> a -> b)
  (#l : nat)
  (acc : b)
  (s : lseq a l)
  : Lemma 
    (requires true)
    (ensures
      Kuiper.Seq.Common.seq_fold_left f acc s == seq_fold_left' f acc s
    )
=
  if l = 0
    then ()
    else (
      lemma_fold_left f #(l - 1) (f acc (Seq.head s)) (Seq.tail s);
      lemma_fold_left' f acc s 1
    )



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
  : Ghost et
    (requires true)
    (ensures fun r -> r == seq_fold_left' add zero s)
= 
  lemma_fold_left add zero s;
  DP.sum s

let dprod
  (#et:_) {| scalar et |}
  (#l : nat)
  (s t : lseq et l)
  : GTot et
=
  sum (pmul s t)


let seq_inits_pmul
  (#et:_) {| scalar et |}
  (#l : nat)
  (s t : lseq et l)
  : Lemma (requires l > 0) (ensures inits (s `pmul` t) == inits s `pmul` inits t)
=
  assert inits (s `pmul` t) `Seq.equal` (inits s `pmul` inits t)

noextract
let sum_last
  (#et:_) {| scalar et |}
  (#l n : nat{n < l})
  (s : lseq et l)
  : Lemma
    (requires true)
    (ensures sum (seq_take n s) `add` (s @! n) == sum (seq_take (n + 1) s))
=
  let sn = seq_take n s in
  let sn' = seq_take (n + 1) s in
  assert (seq_take n sn' `Seq.equal` sn)

noextract
let dprod_add
  (#et:_) {| scalar et |}
  (#nnz : nat)
  (n : nat{n < nnz})
  (s t : lseq et nnz)
  (dp : et)
  : Lemma
    (requires dp == sum (seq_take n (pmul s t)))
    (ensures dp `add` ((s @! n) `mul`(t @! n)) == sum (seq_take (n + 1) (pmul s t)))
=
  let open FStar.Seq in
  let ps = seq_take n (pmul s t) in
  let ps' = init_ghost (n + 1) (fun i ->
    if i = n
      then (s @! i) `mul` (t @! i)
      else ps @! i
  ) in

  assert (ps' `equal` seq_take (n + 1) (pmul s t));
  sum_last n (pmul s t)



(* Propiedades sobre las posiciones de un array esparso *)

noextract
let in_bounds (l : nat) (s : seq nat) : prop =
  forall i. 0 <= s @! i /\ s @! i < l

noextract
let sorted (s : seq nat) : prop =
  forall i j. i < j ==> s @! i < s @! j

// MAYBE let valid_pos (nnz l : nat) = s : lseq nat nnz{in_bounds l s /\ sorted s}
noextract
let valid_pos (#nnz l : nat) (s : lseq nat nnz) : prop =
  in_bounds l s /\ sorted s


let valid_pos_implies_len_bounded
  (#nnz l : nat)
  (s : lseq nat nnz)
  : Lemma
    (requires valid_pos l s)
    (ensures nnz <= l)
=
  if l = 0
    then (
      // assert nnz == 0
      //assert s `Seq.equal` Seq.empty
      // que pasa acá?
      admit()
    )
    else admit()

let rec valid_pos_mem
  (#nnz l : nat)
  (s : lseq nat nnz{valid_pos l s})
  : Lemma
      (requires l > 0 /\ Seq.mem (l - 1) s)
      (ensures last s == l - 1)
=
  if len s = 1
    then ()
    else valid_pos_mem #(nnz - 1) l (Seq.tail s)

let rec valid_pos_not_mem_aux
  (m n k : nat{k <= n})
  (s : lseq nat m{valid_pos (n + 1) s})
  : Lemma
    (requires ~(Seq.mem n s) /\ forall i. k <= s @! i)
    (ensures m + k <= n)
=
  if m = 0
    then ()
    else 
      valid_pos_not_mem_aux (m - 1) n (k + 1) (Seq.tail s)


let valid_pos_not_mem
  (#nnz l : nat{l > 0})
  (s : lseq nat nnz{valid_pos l s})
  : Lemma
      (requires ~(Seq.mem (l - 1) s))
      (ensures nnz < l)
=
  valid_pos_not_mem_aux nnz (l - 1) 0 s

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
  
  while (FStar.SizeT.(!i <^ a.nnz))
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

noextract
let seq_take_len
  (#a:Type) (#l: nat)
  (s : lseq a l)
  : Lemma
    (requires true)
    (ensures seq_take l s == s)
=
  assert (seq_take l s `Seq.equal` s)



let seq_project
  (#et:_) {| scalar et |}
  (#nnz #l : nat)
  (pos : lseq nat nnz{valid_pos l pos})
  (s : lseq et l)
  : GTot (lseq et nnz)
=
  let open FStar.Seq in
  // me gustaría escribir:
  // map_seq (index s) pos
  init_ghost nnz fun i ->
    index s (index pos i)

let sparse_pmul
  (#et:_) {| scalar et |}
  (#nnz #l : nat{nnz <= l})
  (elems : lseq et nnz)
  (pos   : lseq nat nnz{valid_pos l pos})
  (s : lseq et l)
  : GTot (lseq et nnz)
=
  let open FStar.Seq in
  elems `pmul` seq_project pos s

let sparse_dprod
  (#et:_) {| scalar et |}
  (#nnz #l : nat{nnz <= l})
  (elems : lseq et nnz)
  (pos   : lseq nat nnz{valid_pos l pos})
  (s : lseq et l)
  : GTot et
=
  sum (sparse_pmul elems pos s)


noextract
let rec count_nonzeros
  (#et : eqtype) {| scalar et |}
  (#l : nat)
  (s : lseq et l)
  : GTot nat
=
  if l = 0
    then 0
    else 
      let c = count_nonzeros (inits s) in
      if last s = zero then c else c + 1

noextract
let rec nonzeros
  (#et : eqtype) {| scalar et |}
  (#l : nat)
  (s : lseq et l)
  : Ghost (seq et)
    (requires true)
    (ensures fun s' -> len s' == count_nonzeros s)
=
  if l = 0
    then seq![] 
    else
      let nnz = nonzeros (inits s) in
      let c = len nnz in
      if last s = zero
        then nnz
        else Seq.init_ghost (c + 1) fun i ->
          if i = c then last s else nnz @! i
          
noextract
let rec lemma_nonzeros
  (#et: eqtype) {| scalar et |}
  (#l : nat)
  (s : lseq et l)
  : Lemma
    (requires forall i. s @! i == zero)
    (ensures nonzeros s == seq![])
=
  if l = 0
    then ()
    else lemma_nonzeros (inits s)

noextract
let rec sum_nonzeros
  (#et: eqtype) {| scalar et |}
  (#l : nat)
  (s : lseq et l)
  : Lemma
    (requires true)
    (ensures sum s == sum #_ #_ #(count_nonzeros s)(nonzeros s))
=
  let c = count_nonzeros s in
  if l = 0
    then ()
    else
      let s' = inits s in
      sum_nonzeros (inits s);
      assert (sum s == sum (inits s) `add` last s);
      if last s = zero
        then ()
        else assert (inits #_ #c (nonzeros s) `Seq.equal` nonzeros s')


noextract
let lemma_inits_seq_project
  (#et : eqtype) {| scalar et |}
  (#nnz #l : nat{0 < nnz /\ nnz <= l})
  (pos   : lseq nat nnz{valid_pos l pos})
  (s : lseq et l)
  : Lemma
    (requires last pos == l - 1 /\ valid_pos (l - 1) (inits pos))
    (ensures inits (seq_project pos s) == seq_project (inits pos) (inits s))
=
  assert inits (seq_project pos s) `Seq.equal` seq_project (inits pos) (inits s)

noextract
let lemma_inits_unsparse
  (#et : eqtype) {| scalar et |}
  (#nnz #l : nat{0 < nnz /\ nnz <= l})
  (elems : lseq et nnz)
  (pos   : lseq nat nnz{valid_pos l pos})
  (s : lseq et l)
  : Lemma
    (requires last pos == l - 1 /\ valid_pos (l - 1) (inits pos))
    (ensures
      inits (unsparse nnz l elems pos) == unsparse (nnz - 1) (l - 1) (inits elems) (inits pos)
    )
=
  assert inits (unsparse nnz l elems pos) `Seq.equal` unsparse (nnz - 1) (l - 1) (inits elems) (inits pos)

noextract
let lemma_unsparse_not_mem
  (#et : eqtype) {| scalar et |}
  (#nnz l : nat{0 < nnz /\ nnz < l})
  (elems : lseq et nnz)
  (pos   : lseq nat nnz{valid_pos l pos})
  : Lemma
    (requires valid_pos (l - 1) pos)
    (ensures
      inits(unsparse nnz l elems pos) == unsparse nnz (l - 1) elems pos
    )
=
  assert inits (unsparse nnz l elems pos) `Seq.equal` unsparse nnz (l - 1) elems pos

noextract
let lemma_seq_project_not_mem
  (#et : eqtype) {| scalar et |}
  (#nnz #l : nat{0 < nnz /\ nnz < l})
  (pos   : lseq nat nnz{valid_pos l pos})
  (s : lseq et l)
  : Lemma
    (requires valid_pos (l - 1) pos)
    (ensures
      seq_project pos s == seq_project pos (inits s)
    )
=
  assert seq_project pos s `Seq.equal` seq_project pos (inits s)


noextract
let rec lemma_sparse_nonzeros
  (#et : eqtype) {| scalar et |}
  (#nnz #l : nat{nnz <= l})
  (elems : lseq et nnz)
  (pos   : lseq nat nnz{valid_pos l pos})
  (s : lseq et l)
  : Lemma
    (requires true)
    (ensures
      nonzeros (elems `pmul` seq_project pos s) ==
      nonzeros (unsparse nnz l elems pos `pmul` s)
    )
=
  let open FStar.Seq in
  
  let p1 = elems `pmul` seq_project pos s in
  let p2 = unsparse nnz l elems pos `pmul` s in
  
  if l = 0
    then ()
    else
      if nnz = 0
        then lemma_nonzeros p2
        else (
          if (mem (l - 1) pos)
            then (
              valid_pos_mem l pos;
              seq_inits_pmul elems (seq_project pos s);
              seq_inits_pmul (unsparse nnz l elems pos) s;
              lemma_inits_seq_project pos s;
              lemma_inits_unsparse elems pos s;
              lemma_sparse_nonzeros (inits elems) (inits pos) (inits s);
              assert nonzeros p1 `equal` nonzeros p2
            )
            else (
              valid_pos_not_mem l pos;
              lemma_unsparse_not_mem l elems pos;
              seq_inits_pmul (unsparse nnz l elems pos) s;
              lemma_seq_project_not_mem pos s;
              lemma_sparse_nonzeros elems pos (inits s);
              assert nonzeros p1 `equal` nonzeros p2
            )
        )

noextract
let lemma_sparse_dprod
  (#et : eqtype) {| scalar et |}
  (#nnz #l : nat{nnz <= l})
  (elems : lseq et nnz)
  (pos   : lseq nat nnz{valid_pos l pos})
  (s : lseq et l)
  : Lemma
    (requires true)
    (ensures sparse_dprod elems pos s == dprod (unsparse nnz l elems pos) s)
= 
  let p1 = elems `pmul` seq_project pos s in
  let p2 = unsparse _ _ elems pos `pmul` s in
  
  lemma_sparse_nonzeros elems pos s;
  sum_nonzeros p1;
  sum_nonzeros p2


inline_for_extraction noextract
fn sarray_product_dense
  (#et : eqtype) {| scalar et |}
  (#l : erased nat)
  (a : sarray et l)
  (v : gpu_array et l)
  (#s #t : erased (lseq et l))
  preserves gpu ** a |-> s ** v |-> t
  requires
    pure (a.nnz <= l) // TODO esto se deduce de valid_pos
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

  while (FStar.SizeT.(!i <^ a.nnz))
    invariant
      (exists* i_v dp_v.
        i |-> i_v **
        dp |-> dp_v **
        pure (
          i_v <= a.nnz /\
          dp_v == sum (seq_take i_v (sparse_pmul v_elems pos t))
        )
      )
  {
    let p = gpu_array_read a.pos !i;
    let x = gpu_array_read a.elems !i;
    let y = gpu_array_read v p;
    
    dprod_add !i v_elems (seq_project pos t) !dp; 

    dp := !dp `add` (x `mul` y);
    i := !i `SZ.add` 1sz;
  };

  seq_take_len (sparse_pmul v_elems pos t);
  lemma_sparse_dprod v_elems pos t; 

  fold sarray_pts_to a s;
  fold v |-> t;

  !dp;
}

//TODO que pasa con esto?
// let _product_dense_u32 = sarray_product_dense #u32 #_

(* producto intero sparse x sparse *)

inline_for_extraction noextract
fn sarray_product
  (#et : eqtype) {| scalar et |}
  (#l : erased nat)
  (a b : sarray et l)
  (#s #t : erased (lseq et l))
  preserves gpu ** a |-> s ** b |-> t
  requires
    emp//pure (a.nnz <= l) // TODO esto se deduce de valid_pos
  returns
    dp: et
  ensures
    emp //pure (dp == dprod s t)
{
  unfold sarray_pts_to a s;
  unfold sarray_pts_to b t;

  let mut dp : et = zero;

  let mut i = 0sz;
  while (FStar.SizeT.(!i <^ a.nnz))
    invariant live i ** live dp
  {
    let mut j = 0sz;
    let p_a = gpu_array_read a.pos !i;
    while (FStar.SizeT.(!j <^ b.nnz))
      invariant live j ** live dp
    {
      let p_b = gpu_array_read b.pos !j;
      if (p_a = p_b)
        ensures live j ** live dp
      {
        let x = gpu_array_read a.elems !i;
        let y = gpu_array_read b.elems !j;
        dp := !dp `add` (x `mul` y);
      };
      j := !j `SZ.add` 1sz;
    };
    i := !i `SZ.add` 1sz;
  };


  fold sarray_pts_to a s;
  fold sarray_pts_to b t;

  !dp;
}