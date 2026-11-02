module Kuiper.Sparse.Common

#lang-pulse

open Kuiper
open Kuiper.Sparse.Math { divup }
open Kuiper.Tensor
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
instance has_vec_cpy_sz : has_vec_cpy sz = { _chunk = 4sz; _pf = ez }

(* Secuencias *)

let rec mem_slice
  (#et : eqtype)
  (x : et) (s : seq et)
  (a b : nat {a <= b /\ b <= len s})
  : Pure bool
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

(* Matrices *)

open Kuiper.EMatrix
open Kuiper.Chest

let ematrix_row_chunk_
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (em : chest2 et rows cols)
  (i : natlt rows)
  (j : natlt cols { j + chunk et <= cols })
: GTot (lseq et (chunk et))
= Seq.init_ghost (chunk et) (fun k -> acc2 em i (j + k))

let ematrix_row_chunk
  (#et : Type0) {| sized et, has_vec_cpy et |}
  // usamos pos y no nat porque garantiza que chunk et <= cols
  (#rows #cols : pos { chunk et /? cols })
  (em : chest2 et rows cols)
  (i : natlt rows)
  (j : natlt cols { chunk et /? j })
: GTot (lseq et (chunk et))
= ematrix_row_chunk_ em i j

let ematrix_row_chunk_index
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos { chunk et /? cols })
  (em : chest2 et rows cols)
  (i : natlt rows)
  (j : natlt cols { chunk et /? j })
  (k : natlt (chunk et))
  : Lemma (Seq.index (ematrix_row_chunk em i j) k == acc2 em i (j + k))
  = ()

let is_ematrix_tile_at
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
=
  seq_chunk s (k * chunk et) ==
  ematrix_row_chunk em i (offset_chunk et j k nthr)

let is_ematrix_tile_at_intro
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (em : chest2 et rows cols)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#row_tile : nat { chunk et /? row_tile })
  (s : lseq et row_tile)
  (nthr : nat)
  (k : natlt (row_tile / chunk et))
: Lemma
  (requires
    offset_chunk et j k nthr < cols /\
    seq_chunk s (k * chunk et) ==
      ematrix_row_chunk em i (offset_chunk et j k nthr))
  (ensures is_ematrix_tile_at em i j s nthr k)
=
  ()

let is_ematrix_tile
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (em : chest2 et rows cols)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#row_tile : nat { chunk et /? row_tile })
  (s : lseq et row_tile)
  (nthr : nat)
: prop
=
  forall (k : natlt (row_tile / chunk et)).
    offset_chunk et j k nthr < cols ==>
      is_ematrix_tile_at em i j s nthr k

let is_ematrix_tile_at_eq
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (em : chest2 et rows cols)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#row_tile : nat { chunk et /? row_tile })
  (s : lseq et row_tile)
  (nthr : nat)
  (k : natlt (row_tile / chunk et))
: Lemma
  (requires
    offset_chunk et j k nthr < cols /\
    is_ematrix_tile em i j s nthr)
  (ensures
    seq_chunk s (k * chunk et) ==
    ematrix_row_chunk em i (offset_chunk et j k nthr))
=
  ()

let is_ematrix_tile_intro
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (em : chest2 et rows cols)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#row_tile : nat { chunk et /? row_tile })
  (s : lseq et row_tile)
  (nthr : nat)
: Lemma
  (requires
    forall (k : natlt (row_tile / chunk et)).
      offset_chunk et j k nthr < cols ==>
        is_ematrix_tile_at em i j s nthr k)
  (ensures is_ematrix_tile em i j s nthr)
=
  ()

(* Propiedades sobre las posiciones de un arreglo ralo *)

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
    seq_make_sparse #_ #(j - i) #n (Seq.slice pos i j) s
  )
= assert
    Seq.slice (seq_make_sparse pos s) i j `Seq.equal`
    seq_make_sparse #_ #(j - i) #n (Seq.slice pos i j) s

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

(* SL helpers *)

ghost
fn when__intro_true (p : prop) (q : slprop)
  requires pure p
  requires q
  ensures when__ p (fun _ -> q)
{
  rewrite q as when__ p (fun _ -> q)
}

ghost
fn when__intro_false (p : prop) (q : squash p -> slprop)
  requires pure (~p)
  ensures when__ p q
{
  rewrite emp as when__ p q
}

ghost
fn when__elim_true (p : prop) (q : slprop)
  requires pure p
  requires when__ p (fun _ -> q)
  ensures q
{
  rewrite when__ p (fun _ -> q) as q;
}

ghost
fn when__elim_false (p : prop) (q : squash p -> slprop)
  requires pure (~p)
  requires when__ p q
{
  rewrite when__ p q as emp;
}

ghost
fn forevery_refine_pred'
  (#a:Type0)
  (f: a -> prop)
  (p: (x:a) -> squash (f x) -> slprop)
  requires
    forall+ (x:a). when__ (f x) (p x)
  ensures
    forall+ (x:a { f x }). p x ()
{
  forevery_refine_split (fun x -> when__ (f x) (p x)) f;
  drop_ (forall+ (x:a { ~(f x) }). when__ (f x) (p x));
  forevery_ext (fun (x:a { f x }) -> when__ (f x) (p x)) (fun x -> p x ());
}

let divup_factor (n : nat) (d : pos) =
  (i : natlt (divup n d) & (j : natlt d {i * d + j < n }))

let bij_divup_factor (n : nat) (d : pos)
: Kuiper.Bijection.bijection (natlt n) (divup_factor n d)
=
{
  ff = (fun (i : natlt n) -> (|i / d, i % d|) <: divup_factor n d);
  gg = (fun (|j, k|) -> j * d + k);

  ff_gg = (fun _ -> ());
  gg_ff = (fun _ -> ());
}

ghost
fn forevery_factor_
  (n : nat)
  (d : pos)
  (p : natlt n -> slprop)
  requires forall+ (i:natlt n). p i
  ensures forall+ (i1:natlt (divup n d)) (i2:natlt d {i1 * d + i2 < n}).
    p (i1 * d + i2)
{
  forevery_iso (bij_divup_factor n d) p;
  forevery_ext #(divup_factor n d)
    (fun q -> p ((bij_divup_factor n d).gg q))
    (fun q -> p (q._1 * d + q._2));
  forevery_unflatten_dep
    #(natlt (divup n d)) #(fun i1 -> (i2 : natlt d {i1 * d + i2 < n}))
    (fun i1 i2 -> p (i1 * d + i2));
}

ghost
fn forevery_unfactor_
  (n : nat)
  (d : pos)
  (p : natlt n -> slprop)
  requires forall+ (i1:natlt (divup n d)) (i2:natlt d {i1 * d + i2 < n}).
    p (i1 * d + i2)
  ensures forall+ (i:natlt n). p i
{
  forevery_flatten_dep
    #(natlt (divup n d)) #(fun i1 -> (i2 : natlt d {i1 * d + i2 < n}))
    (fun i1 i2 -> p (i1 * d + i2));
  forevery_iso (Kuiper.Bijection.bij_sym (bij_divup_factor n d )) _;
  forevery_ext _ (fun i -> p i);
}
