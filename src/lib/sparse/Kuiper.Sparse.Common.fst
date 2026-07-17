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

let divup (n : int) (d : pos {n + d > 0}) : Tot nat = (n + d - 1) / d

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
: Lemma (requires d /? a /\ d /? b) (ensures d /? (a + k * b) /\ d /? (k * b + a))
=
  lemma_divides_product_r d k b;
  lemma_divides_sum d a (k * b)

let prod_preserves_divides (c d : pos) (a : nat)
: Lemma (requires c /? a) (ensures (c * d) /? (a * d))
=
  lemma_divides_product_l c a d;
  lemma_divides_exact c (a * d);
  intro_divides (c * d) (a / c) (a * d)

let lemma_divides_leq
  (d : pos)
  (a b : nat)
: Lemma
  (requires d /? a /\ d /? b)
  (ensures b < a <==> b + d <= a)
= ()

// ya esta definido pero pide que c sea pos??
let lemma_divides_chain (a b : pos) (c : nat)
  : Lemma (requires a /? b /\ b /? c)
          (ensures a /? c)
= ()


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

let seq_chunk
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n : nat)
  (s : lseq et n)
  (k : nat { k + chunk et <= n })
: GTot (lseq et (chunk et))
= Seq.slice s k (k + chunk et)



(* Matrices *)

open Kuiper.EMatrix

let ematrix_from_rows
  (#et : Type0)
  (#rows #cols : erased nat)
  (r : natlt rows -> GTot (lseq et cols))
: ematrix et rows cols
= mkM fun i j -> r i @! j

let ematrix_from_rows_lemma
  (#et : Type0)
  (#rows #cols : erased nat)
  (r : natlt rows -> GTot (lseq et cols))
  (i : natlt rows)
: Lemma (requires true) (ensures ematrix_row (ematrix_from_rows r) i == r i)
  [SMTPat (ematrix_row (ematrix_from_rows r) i)]
= assert ematrix_row (ematrix_from_rows r) i `Seq.equal` r i 

let ematrix_rows_equal
  (#et : Type0)
  (#rows #cols : erased nat)
  (m1 m2 : ematrix et rows cols)
: prop = forall i. ematrix_row m1 i == ematrix_row m2 i 

let ematrix_rows_equal_intro
  (#et : Type0)
  (#rows #cols : erased nat)
  (m1 m2 : ematrix et rows cols)
: Lemma (requires ematrix_rows_equal m1 m2) (ensures m1 == m2)
  [SMTPat (ematrix_rows_equal m1 m2)]
=
  introduce forall i j. macc m1 i j == macc m2 i j
  with (
    assert macc m1 i j == ematrix_row m1 i @! j;
    assert macc m2 i j == ematrix_row m2 i @! j
  );
  assert m1 `Kuiper.EMatrix.equal` m2

let ematrix_from_cols
  (#et : Type0)
  (#rows #cols : erased nat)
  (c : natlt cols -> GTot (lseq et rows))
: ematrix et rows cols
= mkM fun i j -> c j @! i

let ematrix_from_cols_lemma
  (#et : Type0)
  (#rows #cols : erased nat)
  (c : natlt cols -> GTot (lseq et rows))
  (j : natlt cols)
: Lemma (requires true) (ensures ematrix_col (ematrix_from_cols c) j == c j)
  [SMTPat (ematrix_col (ematrix_from_cols c) j)]
= assert ematrix_col (ematrix_from_cols c) j `Seq.equal` c j 

let ematrix_cols_equal
  (#et : Type0)
  (#rows #cols : erased nat)
  (m1 m2 : ematrix et rows cols)
: prop = forall j. ematrix_col m1 j == ematrix_col m2 j 

let ematrix_cols_equal_intro
  (#et : Type0)
  (#rows #cols : erased nat)
  (m1 m2 : ematrix et rows cols)
: Lemma (requires ematrix_cols_equal m1 m2) (ensures m1 == m2)
  [SMTPat (ematrix_cols_equal m1 m2)]
=
  introduce forall i j. macc m1 i j == macc m2 i j
  with (
    assert macc m1 i j == ematrix_col m1 j @! i;
    assert macc m2 i j == ematrix_col m2 j @! i
  );
  assert m1 `Kuiper.EMatrix.equal` m2

let ematrix_upd_row_f
  (#et : Type0)
  (#rows #cols : erased nat)
  (em : ematrix et rows cols)
  (i : natlt rows)
  (new_row : lseq et cols)
  (k : natlt rows)
: GTot (lseq et cols)
= if k == i then new_row else ematrix_row em k

let ematrix_upd_row_lemma
  (#et : Type0)
  (#rows #cols : erased nat)
  (em : ematrix et rows cols)
  (i : natlt rows)
  (new_row : lseq et cols)
: Lemma
  (requires true)
  (ensures ematrix_upd_row em i new_row == ematrix_from_rows (ematrix_upd_row_f em i new_row)) 
=
  assert Kuiper.EMatrix.equal
    (ematrix_upd_row em i new_row)
    (ematrix_from_rows (ematrix_upd_row_f em i new_row))

let ematrix_row_chunk_
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (em : ematrix et rows cols)
  (i : natlt rows)
  (j : natlt cols { j + chunk et <= cols })
: GTot (lseq et (chunk et))
= Seq.init_ghost (chunk et) (fun k -> macc em i (j + k)) 

let ematrix_row_chunk
  (#et : Type0) {| sized et, has_vec_cpy et |}
  // usamos pos y no nat porque garantiza que chunk et <= cols
  (#rows #cols : pos { chunk et /? cols })
  (em : ematrix et rows cols)
  (i : natlt rows)
  (j : natlt cols { chunk et /? j })
: GTot (lseq et (chunk et))
= ematrix_row_chunk_ em i j

// TODO estas cosas son medio especificas, mover??
let offset_chunk
  (et : Type0) {| sized et, has_vec_cpy et |}
  (j : nat { chunk et /? j })
  (k : nat)
  (nthr : nat)
: Pure nat (requires true) (ensures divides (chunk et))
=
  lemma_divides_product (chunk et) (k * nthr);
  lemma_divides_sum (chunk et) j (k * nthr * chunk et);
  j + k * nthr * v (chunk et)

let is_ematrix_tile_at
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (em : ematrix et rows cols)
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

let is_ematrix_tile
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (em : ematrix et rows cols)
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
  (#et : Type0)
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
    seq_make_sparse #_ #(j - i) #n (Seq.slice pos i j) s
  )
= assert
    Seq.slice (seq_make_sparse pos s) i j `Seq.equal`
    seq_make_sparse #_ #(j - i) #n (Seq.slice pos i j) s

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

(* Tensors *) // tiene sentido que esté acá?

module A = Kuiper.Array1
module M = Kuiper.Array2
module T = Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Array2.Strided { strided_row_major, aligned_strided_row_major }

let array_cell_of_pos (#n : nat)
  (l : A.layout n) (i : natlt n) : GTot (natlt (A.layout_size l)) =
  l.imap.f (A.adapt_idx_back i)

open FStar.Tactics.Typeclasses { no_method }

inline_for_extraction noextract
class cont_layout (#n : erased nat) (l : A.layout n) = {
  [@@@no_method]
  offset : sz;
  [@@@no_method]
  pf : i:natlt n -> squash (array_cell_of_pos l i == offset + i);
}

let aligned_cont_layout
  (#n : erased nat)
  (#l : A.layout n)
  (k : pos)
  (cl : cont_layout l)
: prop = k /?+ cl.offset

inline_for_extraction noextract
instance cont_layout_l1_forward (#n : erased nat)
  : cont_layout (l1_forward n) = {
    offset  = 0sz;
    pf = ez
  }

inline_for_extraction noextract
instance cont_layout_strided_row_major
  (#rows #cols : erased nat)
  (l : M.layout rows cols { 0 < cols }) {| srm : strided_row_major l |}
  (#_: squash (fits (M.layout_size l)))
  (i : natlt rows)
  {| conc_i : concrete_sz i |}
  : cont_layout #cols (T.tlayout_slice l 0 i) = {
    offset  = (srm.pf i 0; srm.offset +^ srm.stride *^ (concr' conc_i));
    pf = srm.pf i;
  }

let aligned_cont_strided_row_major
  (#rows #cols : erased nat { 0 < cols })
  (l : M.layout rows cols) {| srm : strided_row_major l |}
  (#_: squash (fits (M.layout_size l)))
  (k : pos)
  (i : szlt rows)
: Lemma
  (requires aligned_strided_row_major k srm)
  (ensures aligned_cont_layout k (cont_layout_strided_row_major l i))
=
  srm.pf i 0;
  lineal_divides k srm.offset srm.stride i

// esto es cierto pero no lo puedo probar con la interfaz actual
let row_core_lemma
  (#et:Type0)
  (#m #n : nat)
  (#l : M.layout m n)
  (a : M.array2 et l)
  (i : natlt m)
: Lemma (requires true) (ensures A.core (M.row a i) == M.core a)
= admit()