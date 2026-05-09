module Kuiper.Sparse.Common

#lang-pulse
open Kuiper
module SZ = FStar.SizeT

// This is here to force extraction.
let _ = 1ul

(* Misc *)

let divup (n : nat) (d : pos) : GTot nat = ((n + d) - 1) / d

(* sdivup is implemented as (n + (d-1))/d. Associating
that way usually performs more partial evaluation as d is usually
known. *)
[@@"opaque_to_smt"] // Important to prevent a trigger cascade apparently... investigate
inline_for_extraction noextract
let divup_ (n : sz) (d : szp)
: Pure sz (requires fits (n + d)) (ensures fun r -> SZ.v r == divup n d)
= sdivup n d

(* Orderings *)

open Kuiper.Bijection

let permutation a = bijection a a

let ordering (#n : nat{ fits n }) (p : permutation (natlt n))
: GTot (seq sz)
// : Ghost (seq sz) (requires fits n) (ensures fun s -> forall i. s @! i < n)
= Seq.init_ghost n (fun i -> uint_to_t (i |~> p))

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


let valid_pos (#nnz l : nat) (s : lseq nat nnz) : prop
= in_bounds 0 l s /\ sorted s

let seq_make_sparse
  (#et : Type0) {| scalar et |}
  (#nnz #n : nat)
  (pos : lseq nat nnz{valid_pos n pos})
  (s : lseq et n)
  : lseq et nnz
=
  Seq.init nnz (fun i -> s @! (pos @! i))

// renombrar a seq_unsparse
let unsparse
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