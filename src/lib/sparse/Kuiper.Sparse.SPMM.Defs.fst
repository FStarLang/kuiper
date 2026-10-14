module Kuiper.Sparse.SPMM.Defs

(* Shared type definitions for the SPMM kernel. *)

#lang-pulse

open Kuiper
module SZ = Kuiper.SizeT
open Kuiper.Sparse
open Kuiper.Math { even, odd }
open Kuiper.Array.Vectorized

inline_for_extraction
type parameters (et : Type0) {| sized et, has_vec_cpy et |} = {
  rows : szp;
  shared : szp;
  cols : (k : szp { chunk et /? k});
  blockItemsK : szp;
  blockItemsX : szp;
  blockWidth : (k : szp {
    (k * chunk et) /? blockItemsK /\
    (k * chunk sz) /? blockItemsK /\
    (k * chunk et) /? blockItemsX
  });
}

let nblocks_
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et)
: GTot pos
= p.rows * (p.cols `divup` p.blockItemsX)

let nthreads_
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et)
: GTot pos
= p.blockWidth

let allthreads_
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et)
: GTot pos
= nblocks_ p * nthreads_ p

inline_for_extraction noextract
let size_req
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et) =
    nblocks_ p <= max_blocks /\
    p.blockWidth <= max_threads /\
    p.rows < 10000 /\
    p.shared < 10000 /\
    p.cols < 10000 /\
    p.blockItemsK < 10000 /\
    p.blockItemsX < 10000

inline_for_extraction noextract
let nblocks
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et)
: Pure (szle max_blocks)
  (requires size_req p)
  (ensures fun r -> SZ.v r == nblocks_ p)
= p.rows *^ (p.cols `divup_` p.blockItemsX)

inline_for_extraction noextract
let nthreads
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et)
: Pure (szle max_threads)
  (requires size_req p)
  (ensures fun r -> SZ.v r == nthreads_ p)
= p.blockWidth

inline_for_extraction noextract
let allthreads
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et)
: Pure sz
  (requires size_req p)
  (ensures fun r -> SZ.v r == allthreads_ p)
= nblocks p *^ nthreads p

let brow
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et) (bid : natlt (nblocks_ p))
: GTot (natlt p.rows)
= bid % p.rows

inline_for_extraction noextract
let brow_
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et) (bid : szlt (nblocks_ p))
  (#_ : squash (fits (p.cols + p.blockItemsX)))
: Tot (m : sz {SZ.v m == brow p bid})
= bid %^ p.rows

(* [bid < rows * divup cols x ==> (bid / rows) * x < cols], with every
   nonlinear step explicit. *)
let bcol_bound (rows cols x : pos) (bid : nat)
: Lemma (requires bid < rows * divup cols x)
        (ensures (bid / rows) * x < cols)
= let q = bid / rows in
  let d = divup cols x in
  FStar.Math.Lemmas.euclidean_division_definition bid rows;
  if q >= d then FStar.Math.Lemmas.lemma_mult_le_left rows d q;
  assert (q + 1 <= d);
  FStar.Math.Lemmas.lemma_mult_le_right x (q + 1) d;
  FStar.Math.Lemmas.distributivity_add_left q 1 x;
  FStar.Math.Lemmas.euclidean_division_definition (cols + (x - 1)) x;
  FStar.Math.Lemmas.swap_mul d x

let bcol
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et) (bid : natlt (nblocks_ p))
: GTot (natlt p.cols) // por que Ghost?
= bcol_bound p.rows p.cols p.blockItemsX bid;
  (bid / p.rows) * p.blockItemsX

inline_for_extraction noextract
let bcol_
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et { size_req p }) (bid : szlt (nblocks_ p))
: Tot (n : sz {SZ.v n == bcol p bid})
= FStar.SizeT.fits_at_least_16 (bcol p bid);
  let q : (q : sz { SZ.v q == SZ.v bid / SZ.v p.rows }) = bid /^ p.rows in
  assert (bcol p (SZ.v bid) < SZ.v p.cols);
  q *^ p.blockItemsX

noextract
let tcol
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (p : parameters et)
  (bid : natlt (nblocks_ p))
  (tid : nat)
: Ghost nat (requires true) (ensures fun c -> chunk et /? c)
=
  lemma_divides_product (chunk et) tid;
  assert chunk et /? (tid * chunk et);
  lemma_divides_product p.blockItemsX (bid / p.rows);
  assert p.blockItemsX /? bcol p bid;
  prod_divides (p.blockWidth) (chunk et) p.blockItemsX;
  assert chunk et /? p.blockItemsX;
  lemma_divides_chain (chunk et) p.blockItemsX (bcol p bid);
  lemma_divides_sum (chunk et) (bcol p bid) (tid * chunk et);
  bcol p bid + tid * chunk et

inline_for_extraction noextract
let tcol_
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (bid : szlt (nblocks_ p))
  (tid : szlt p.blockWidth)
: Pure sz (requires true) (ensures fun c -> SZ.v c == tcol p bid tid)
= assert (SZ.v (chunk et) * SZ.v (size #et) == 16);
  assert (SZ.v (chunk et) <= 16);
  assert (SZ.v tid < SZ.v p.blockWidth /\ SZ.v p.blockWidth <= SZ.v max_threads);
  let c : (c : sz { SZ.v c == SZ.v tid * SZ.v (chunk et) }) = tid *^ chunk et in
  let b : (b : sz { SZ.v b == bcol p bid }) = bcol_ p bid in
  (* [c <= 1024 * 16] and [b < 10000], so [b + c] fits in 16 bits *)
  FStar.Math.Lemmas.lemma_mult_le_right (SZ.v (chunk et)) (SZ.v tid) 1024;
  FStar.Math.Lemmas.lemma_mult_le_left 1024 (SZ.v (chunk et)) 16;
  assert (SZ.v c <= 16384);
  assert_norm (pow2 16 == 65536);
  FStar.SizeT.fits_at_least_16 (SZ.v b + SZ.v c);
  b +^ c

// MAYBE definir threadItemsX?

(* Description of shared memory used in this kernel. *)
inline_for_extraction noextract
let shmems_desc
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et)
: list shmem_desc = [
  SHArray et p.blockItemsK;
  // TODO podemos parametrizar este tipo?
  SHArray sz p.blockItemsK;
]

// esto no tiene sentido, pedir valid_smatrix y listo
unfold
let well_formed
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et)
  (#nnz : sz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
: prop
= valid_smatrix p.rows p.shared (cast_pos col_ind) (cast_pos row_off)

(* Lemas *)

// TODO esto se usa?
let block_lemma whole block k
  : Lemma (requires block /? whole /\ k * block < whole)
          (ensures k * block + block <= whole)
  = if block > 0 then begin
      lemma_divides_exact block whole;
      FStar.Math.Lemmas.multiplication_order_lemma k (whole / block) block;
      FStar.Math.Lemmas.lemma_mult_le_right block (k + 1) (whole / block)
    end

(* [block_lemma] is the same fact minus [off]; chaining it explicitly replaces
   a nonlinear search (from [block /? whole] and [k * block < whole], conclude
   [k * block + block <= whole]) that Z3 no longer completes on its own. *)
let block_lemma_off whole block k off
  : Lemma (requires block /? whole /\ k * block < whole /\ off < block)
          (ensures k * block + off < whole)
  = block_lemma whole block k

(* An offset that is a multiple of [chunk et] is 16-byte aligned, since
   [chunk et * size et == 16]; nonlinear, so proven here with explicit steps. *)
let chunk_offset_aligned
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (x : array et { aligned 16 x })
  (off : nat)
: Lemma
  (requires chunk et /? off)
  (ensures aligned' 16 x off)
=
  let c : pos = SZ.v (chunk et) in
  assert (c * size #et == 16);
  let z = off / c in
  lemma_divides_exact c off;
  assert (off == c * z);
  FStar.Math.Lemmas.paren_mul_right c z (size #et);
  FStar.Math.Lemmas.swap_mul z (size #et);
  FStar.Math.Lemmas.paren_mul_right c (size #et) z;
  assert (off * size #et == 16 * z);
  FStar.Math.Lemmas.swap_mul 16 z;
  FStar.Math.Lemmas.lemma_mod_plus (base_address x) z 16

#push-options "--z3rlimit 30"
let offset_aligned_lemma_et
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (p : parameters et)
  (#n : nat)
  (x : larray et n { aligned 16 x })
  (i k : nat)
: Lemma
  (requires true)
  (ensures aligned' 16 x
    (round2 (max (chunk et) (chunk sz)) i + k * p.blockItemsK)
  )
=
  let i' = round2 (max (chunk et) (chunk sz)) i in
  round2_chunk_lemma et sz i;
  assert chunk et /? i';
  prod_divides p.blockWidth (chunk et) p.blockItemsK;
  lineal_divides (chunk et) i' p.blockItemsK k;
  chunk_offset_aligned x (i' + k * p.blockItemsK)

let offset_aligned_lemma_sz
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (p : parameters et)
  (#n : nat)
  (x : larray sz n { aligned 16 x })
  (i k : nat)
: Lemma
  (requires true)
  (ensures aligned' 16 x
    (round2 (max (chunk et) (chunk sz)) i + k * p.blockItemsK)
  )
=
  let i' = round2 (max (chunk et) (chunk sz)) i in
  round2_chunk_lemma et sz i;
  assert chunk sz /? i';
  prod_divides p.blockWidth (chunk sz) p.blockItemsK;
  lineal_divides (chunk sz) i' p.blockItemsK k;
  chunk_offset_aligned x (i' + k * p.blockItemsK)
#pop-options

// TODO mejores nombress
let offset_aligned_lemma_et'
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (p : parameters et)
  (#n : nat)
  (x : larray et n { aligned 16 x })
  (i : nat)
: Lemma
  (requires true)
  (ensures aligned' 16 x
    (round2 (max (chunk et) (chunk sz)) i)
  )
=
  let i' = round2 (max (chunk et) (chunk sz)) i in
  round2_chunk_lemma et sz i;
  assert chunk et /? i';
  chunk_offset_aligned x i'

let offset_aligned_lemma_sz'
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (p : parameters et)
  (#n : nat)
  (x : larray et n { aligned 16 x })
  (i : nat)
: Lemma
  (requires true)
  (ensures aligned' 16 x
    (round2 (max (chunk et) (chunk sz)) i)
  )
=
  let i' = round2 (max (chunk et) (chunk sz)) i in
  round2_chunk_lemma et sz i;
  assert chunk sz /? i';
  chunk_offset_aligned x i'

open Kuiper.Chest
open Kuiper.EMatrix

let chest2_tile_prop
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#m1 #n1 : nat { chunk et /? n1 })
  (#m2 #n2 : nat {  chunk et /? n2 })
  (em2 : chest2 et m2 n2)
  (row_ind : lseq nat m1 { in_bounds 0 m2 row_ind })
  (j : nat { chunk et /? j })
  (step : nat)
  (em : chest2 et m1 n1)
: GTot prop
=
  forall (k1 : natlt n1).
    let k2 = j + k1 / chunk et * step * chunk et + k1 % chunk et in
    k2 < n2 ==> chest2_col em k1 == seq_to_chest1 (seq_make_sparse row_ind (ematrix_col em2 k2))
