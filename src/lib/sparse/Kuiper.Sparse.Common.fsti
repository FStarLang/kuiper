module Kuiper.Sparse.Common

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Sparse.Math { divup }
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Array2.Strided { strided_row_major, aligned_strided_row_major }
open FStar.Tactics.Typeclasses { no_method }
open Kuiper.Array.Vectorized
open Kuiper.Seq.Common { (@!) }
open Kuiper.Tensor.Layout.Slice

module SZ = FStar.SizeT
module Chest = Kuiper.Chest

(* Class instances *)

inline_for_extraction noextract
instance val has_vec_cpy_sz : has_vec_cpy sz

(* Orderings *)

open Kuiper.Bijection

let permutation a = bijection a a

let ordering (#n : nat{ fits n }) (p : permutation (natlt n))
: GTot (seq sz)
= Seq.init_ghost n (fun i -> uint_to_t (i |~> p))

(* Propiedades sobre escalares *)

val zero_is_absorbing_l
  (#et:_) {| scalar et |}
  (k : et)
  : Lemma
    (requires true)
    (ensures k `mul` zero == zero)
    [SMTPat (k `mul` zero)]

val zero_is_absorbing_r
  (#et:_) {| scalar et |}
  (k : et)
  : Lemma
    (requires true)
    (ensures zero `mul` k == zero)
    [SMTPat (zero `mul` k )]

val zero_is_id_l
  (#et:_) {| scalar et |}
  (k : et)
  : Lemma
    (requires true)
    (ensures k `add` zero == k)
    [SMTPat (k `add` zero)]

val  zero_is_id_r
  (#et:_) {| scalar et |}
  (k : et)
  : Lemma
    (requires true)
    (ensures zero `add` k == k)
    [SMTPat (zero `add` k)]

(* Secuencias *)

let map_seq_len (#a #b:Type) (f:a -> Tot b) (s:Seq.seq a)
  : Lemma (ensures len (Seq.map_seq f s) == len s)
          [SMTPat (Seq.map_seq f s)]
  = Seq.map_seq_len f s

let my_map_seq_index (#a #b:Type) (f:a -> Tot b) (s:Seq.seq a) (i:nat{i < len s})
  : Lemma (ensures (Seq.map_seq_len f s; Seq.map_seq f s @! i == f (s @! i)))
          [SMTPat (Seq.map_seq f s @! i)]
  = Seq.map_seq_index f s i

let seq_chunk
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n : nat)
  (s : lseq et n)
  (k : nat { k + chunk et <= n })
: GTot (lseq et (chunk et))
= Seq.slice s k (k + chunk et)

val mem_slice
  (#et : eqtype)
  (x : et) (s : seq et)
  (a b : nat {a <= b /\ b <= len s})
  : Pure bool (decreases (b - a))
    (requires true)
    (ensures fun r -> (exists i. a <= i /\ i < b /\ x == s @! i) <==> r)

val index_mem_slice
  (#et : eqtype)
  (x : et) (s : seq et)
  (a b : nat {a <= b /\ b <= len s})
  : Pure nat
    (requires (mem_slice x s a b))
    (ensures (fun i -> a <= i /\ i < b /\ s @! i == x))

val mem_slice_lemma
  (#et : eqtype)
  (x : et) (s : seq et)
  (a b : nat {a <= b /\ b <= len s})
  : Lemma
    (ensures mem_slice x s a b <==> Seq.mem x (Seq.slice s a b))
    [SMTPatOr
      [[SMTPat (mem_slice x s a b)];
       [SMTPat (Seq.mem x (Seq.slice s a b))]]]

val index_mem_slice_lemma
  (#et : eqtype)
  (x : et) (s : seq et)
  (a b : nat {a <= b /\ b <= len s})
  : Lemma
    (requires mem_slice x s a b /\ Seq.mem x (Seq.slice s a b))
    (ensures
      s @! index_mem_slice x s a b ==
      Seq.slice s a b @! Seq.index_mem x (Seq.slice s a b)
    )

(* Matrices *)

val ematrix_row_chunk
  (#et : Type0) {| sized et, has_vec_cpy et |}
  // usamos pos y no nat porque garantiza que chunk et <= cols
  (#rows #cols : pos { chunk et /? cols })
  (em : chest2 et rows cols)
  (i : natlt rows)
  (j : natlt cols { chunk et /? j })
: GTot (lseq et (chunk et))

let offset_chunk
  (et : Type0) {| sized et, has_vec_cpy et |}
  (j : nat { chunk et /? j })
  (k nthr : nat)
: Pure nat (requires true) (ensures divides (chunk et))
=
  lemma_divides_product (chunk et) (k * nthr);
  lemma_divides_sum (chunk et) j (k * nthr * chunk et);
  j + k * nthr * v (chunk et)

val is_ematrix_tile_at
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (em : chest2 et rows cols)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#row_tile : nat { chunk et /? row_tile })
  (s : lseq et row_tile)
  (nthr : nat)
  (k : natlt (row_tile / chunk et))
: Pure prop
  (requires offset_chunk et j k nthr < cols)
  (ensures fun _ -> true)

val is_ematrix_tile
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (em : chest2 et rows cols)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#row_tile : nat { chunk et /? row_tile })
  (s : lseq et row_tile)
  (nthr : nat)
: prop

(* Propiedades sobre las posiciones de un arreglo ralo *)

let in_bounds (l h : nat) (s : seq nat) : prop =
  forall i. {:pattern (s @! i)} l <= s @! i /\ s @! i < h

let sorted_slice
  (s : seq nat)
  (a b : nat{a <= b /\ b <= len s})
: prop
= forall i j. {:pattern (s @! i); (s @! j)} a <= i /\ i < j /\ j < b ==> s @! i < s @! j

let sorted (s : seq nat) : prop = sorted_slice s 0 (len s)

val bounded_from_sorted_in_bounds
  (#nnz l h : nat)
  (s : lseq nat nnz)
: Lemma
    (requires l <= h /\ sorted s /\ in_bounds l h s)
    (ensures nnz + l <= h)

let cast_pos
  (#nnz : nat)
  (pos : lseq sz nnz)
: Ghost
  (lseq nat nnz)
  (requires true)
  (ensures fun npos -> forall i. npos @! i == SZ.v (pos @! i))
= Seq.map_seq SZ.v pos


let valid_pos (#nnz l : nat) (s : lseq nat nnz) : prop = in_bounds 0 l s /\ sorted s

let seq_make_sparse
  (#et : Type0)
  (#nnz #n : nat)
  (pos : lseq nat nnz{in_bounds 0 n pos})
  (s : lseq et n)
  : lseq et nnz
= Seq.init nnz (fun i -> s @! (pos @! i))

val seq_make_sparse_slice
  (#et : Type0) {| scalar et |}
  (#nnz #n : nat)
  (pos : lseq nat nnz { in_bounds 0 n pos })
  (i j : natle nnz { i <= j })
  (s : lseq et n)
: Lemma
  (requires true)
  (ensures
    Seq.slice (seq_make_sparse pos s) i j ==
    seq_make_sparse #_ #(j - i) #n (Seq.slice pos i j) s
  )

// renombrar a seq_unsparse
let seq_unsparse
  (#et:Type0) {| scalar et |}
  (nnz l : nat)
  (elems : lseq et nnz)
  (pos   : lseq nat nnz)
  : GTot (lseq et l)
=
  let open FStar.Seq in
  init l fun i ->
    if mem i pos
      then elems @! index_mem i pos
      else zero

(* Utils *)

inline_for_extraction noextract
fn foreach
  (n : sz)
  (p q : natlt n -> slprop)
  (#frame : slprop)
  (f : (i : szlt n) -> stt unit (p i ** frame) (fun _ -> q i ** frame))
  preserves
    frame
  requires
    (forall+ (k : natlt n). p k)
  ensures
    (forall+ (k : natlt n). q k)

(* SL helpers *)

ghost
fn when__intro_true (p : prop) (q : slprop)
  requires pure p
  requires q
  ensures when__ p (fun _ -> q)

ghost
fn when__intro_false (p : prop) (q : squash p -> slprop)
  requires pure (~p)
  ensures when__ p q

ghost
fn when__elim_true (p : prop) (q : slprop)
  requires pure p
  requires when__ p (fun _ -> q)
  ensures q

ghost
fn when__elim_false (p : prop) (q : squash p -> slprop)
  requires pure (~p)
  requires when__ p q

ghost
fn forevery_refine_pred'
  (#a:Type0)
  (f: a -> prop)
  (p: (x:a) -> squash (f x) -> slprop)
  requires
    forall+ (x:a). when__ (f x) (p x)
  ensures
    forall+ (x:a { f x }). p x ()

ghost
fn forevery_factor_
  (n : nat)
  (d : pos)
  (p : natlt n -> slprop)
  requires forall+ (i:natlt n). p i
  ensures forall+ (i1:natlt (divup n d)) (i2:natlt d {i1 * d + i2 < n}).
    p (i1 * d + i2)

ghost
fn forevery_unfactor_
  (n : nat)
  (d : pos)
  (p : natlt n -> slprop)
  requires forall+ (i1:natlt (divup n d)) (i2:natlt d {i1 * d + i2 < n}).
    p (i1 * d + i2)
  ensures forall+ (i:natlt n). p i
