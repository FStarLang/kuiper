module Kuiper.Sparse.Common

#lang-pulse
open Kuiper
module SZ = FStar.SizeT

// This is here to force extraction.
let _ = 1ul

(* Class instances *)

open Kuiper.Array.Vectorized

inline_for_extraction noextract
instance has_vec_cpy_sz : has_vec_cpy sz = { _chunk = 4sz; _pf = ez }

(* Aritmetica *)

let divup (n : int) (d : pos {n + d > 0}) : GTot nat = (n + d - 1) / d

let round2 (k : pos) (n : nat) = (n / k) * k

(* sdivup is implemented as (n + (d-1))/d. Associating
that way usually performs more partial evaluation as d is usually
known. *)
[@@"opaque_to_smt"] // Important to prevent a trigger cascade apparently... investigate
inline_for_extraction noextract
let divup_ (n : sz) (d : szp)
: Pure sz (requires fits (n + d)) (ensures fun r -> SZ.v r == divup n d)
= sdivup n d

(* en sputnik está especializado para potencias de 2: n & (k - 1) *)
inline_for_extraction noextract
let round2_  (k : szp) (n : sz)
: Pure sz (requires true) (ensures fun r -> SZ.v r == round2 k n)
= (n /^ k) *^ k

let div2_lemma (a : nat)
: Lemma (requires a % 2 <> 0) (ensures a % 2 == 1)
= ()

let div2_prod_odd (a b : nat)
: Lemma
  (requires a % 2 == 1 /\ b % 2 == 1)
  (ensures (a * b) % 2 == 1)
=
  let p = a / 2 in
  let q = b / 2 in

  assert a * b = 2 * (p + q + 2*p*q) + 1;
  assert (a * b) % 2 == 1


let div2_lemma_prod (a b : nat)
: Lemma (requires 2 /? (a * b)) (ensures 2 /? a \/ 2 /? b)
=
  if 2 /? a then ()
  else if 2 /? b then ()
  else (
    div2_lemma a;
    div2_lemma b;
    div2_prod_odd a b
  )

let rec factor_pow2 (n : nat) (a b : nat)
: Ghost (nat & nat)
  (requires a * b == pow2 n)
  (ensures fun (p, q) -> pow2 p == a /\ pow2 q == b)
=
  if n = 0
    then (0, 0)
    else (
      assert a * b == 2 * pow2 (n - 1);
      div2_lemma_prod a b;
      if 2 /? a
        then let (p, q) = factor_pow2 (n - 1) (a / 2) b in (p + 1, q)
        else let (p, q) = factor_pow2 (n - 1) a (b / 2) in (p, q + 1)
    )


let pow2_div_log (n : nat) (a : nat)
: Lemma (requires a /? pow2 n) (ensures exists r. pow2 r == a)
= let _ = (factor_pow2 n a (pow2 n / a)) in ()

let pow_div_lemma (n : nat) (a b : nat)
: Lemma
  (requires a /? (pow2 n) /\ b /? (pow2 n) /\ a <= b)
  (ensures a /? b)
= pow2_div_log n a; pow2_div_log n b

let round2_lemma (a b : nat) (n : nat) (k : nat)
: Lemma
  (requires a /? pow2 k /\  b /? pow2 k /\ a <= b)
  (ensures a /? round2 b n)
=
  pow_div_lemma k a b;
  assert a /? b;
  assert b /? round2 b n;
  ()

let round2_chunk_lemma
  (a b : Type0) {| sized a, has_vec_cpy a, sized b, has_vec_cpy b |}
  (n : nat)
: Lemma
  (requires true)
  (ensures
    chunk a /? round2 (max (chunk a) (chunk b)) n /\
    chunk b /? round2 (max (chunk a) (chunk b)) n
  )
=
  let m = max (chunk a) (chunk b) in
  round2_lemma (chunk a) m n 4;
  round2_lemma (chunk b) m n 4;
  ()

let intro_divides (a b c : nat)
: Lemma (requires a * b = c) (ensures a /? c)
= ()

let prod_divides (a b c : pos)
: Lemma (requires (a * b) /? c) (ensures a /? c /\ b /? c)
=
  let k = c / (a * b) in
  intro_divides a (k * b) c;
  intro_divides b (k * a) c;
  ()

let lineal_divides (d : pos) (a b k : nat)
: Lemma (requires d /? a /\ d /? b) (ensures d /? (a + k * b))
=
  lemma_divides_product_r d k b;
  lemma_divides_sum d a (k * b)



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
  (pos : lseq nat nnz{in_bounds 0 n pos})
  (s : lseq et n)
  : lseq et nnz
=
  Seq.init nnz (fun i -> s @! (pos @! i))

let seq_make_sparse_slice
  (#et : Type0) {| scalar et |}
  (#nnz #n : nat)
  (pos : lseq nat nnz { in_bounds 0 n pos })
  (i j : natle nnz { i <= j })
  (s : lseq et n)
: Lemma
  (requires true)
  (ensures
    Seq.slice (seq_make_sparse pos s) i j ==
    seq_make_sparse #_ #_ #(j - i) (Seq.slice pos i j) s
  )
= assert
    Seq.slice (seq_make_sparse pos s) i j `Seq.equal`
    seq_make_sparse #_ #_ #(j - i) (Seq.slice pos i j) s

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


(* Utils *)

open Kuiper.Bijection

let natlt_refined_bij (m n : nat)
: bijection (a : natlt m {a < n}) (natlt (min m n))
= {
  ff = (fun (a : natlt m {a < n}) -> let a' : natlt (min m n) = a in a');
  gg = (fun (b : natlt (min m n)) -> b);
  ff_gg = (fun b -> ());
  gg_ff = (fun a -> ())
}

let natlt_is_between (n : nat) : Lemma (natlt n == between 0 n)
  =
  FStar.RefinementExtensionality.refext
    nat
    (fun (x:nat) -> x < n)
    (fun (x:nat) -> 0 <= x /\ x < n);
  assert (x:nat{x < n} == x:nat{0 <= x /\ x < n});
  assert (natlt n == x:nat{x < n});
  assert_norm (between 0 n == x:nat{0 <= x /\ x < n});
  ()

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
{
  natlt_is_between n;
  assert pure (natlt n == between 0sz n);
  forevery_rw_type (natlt n) (between 0sz n) p;
  Kuiper.For.for_loop' 0sz n
    p q
    frame
    fn x { f x };
  forevery_rw_type (between 0sz n) (natlt n) q;
}
