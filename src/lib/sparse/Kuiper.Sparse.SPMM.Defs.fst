module Kuiper.Sparse.SPMM.Defs

(* Shared type definitions for the SPMM kernel. *)

#lang-pulse

open Kuiper
module SZ = Kuiper.SizeT
open Kuiper.Sparse
open Kuiper.Math { even, odd }
open Kuiper.Array.Vectorized

module Math = Kuiper.Sparse.Math

let size_req rows shared cols blockWidth blockItemsK blockItemsX =
  rows * (cols `divup` blockItemsX) <= max_blocks /\
  blockWidth <= max_threads /\
  rows < 10000 /\
  shared < 10000 /\
  cols < 10000 /\
  blockItemsK < 10000 /\
  blockItemsX < 10000

inline_for_extraction
type parameters (et : Type0) {| sized et, has_vec_cpy et |} = {
  // Thread-to-column grouping; B and C keep their native vector widths.
  workChunk : (k : szp { k <= 16 });
  rows : szp;
  shared : szp;
  cols : (k : szp { workChunk /? k});
  blockItemsK : szp;
  blockItemsX : szp;
  blockWidth : (k : szp {
    (k * chunk et) /? blockItemsK /\
    (k * chunk sz) /? blockItemsK /\
    (k * workChunk) /? blockItemsX
  });
  sz_req : squash (size_req rows shared cols blockWidth blockItemsK blockItemsX);
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
let nblocks
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et)
: Pure (szle max_blocks)
  (ensures fun r -> SZ.v r == nblocks_ p)
= p.rows *^ (p.cols `divup_` p.blockItemsX)

inline_for_extraction noextract
let nthreads
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et)
: Pure (szle max_threads)
  (ensures fun r -> SZ.v r == nthreads_ p)
= p.blockWidth

#push-options "--z3rlimit 20"
inline_for_extraction noextract
let allthreads
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et)
: Pure sz
  (ensures fun r -> SZ.v r == allthreads_ p)
= assert fits (max_blocks * max_threads); nblocks p *^ nthreads p
#pop-options

let brow
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et) (bid : natlt (nblocks_ p))
: GTot (natlt p.rows)
= bid % p.rows

(* [%^] is specified via [mod_spec] (a - (a/b)*b); relate it to [%] explicitly *)
let rem_eq_mod (a : sz) (b : szp)
  : Lemma (SZ.v (a %^ b) == SZ.v a % SZ.v b)
  = FStar.Math.Lemmas.euclidean_division_definition (SZ.v a) (SZ.v b)

inline_for_extraction noextract
let brow_
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et) (bid : szlt (nblocks_ p))
  (#_ : squash (fits (p.cols + p.blockItemsX)))
: Tot (m : sz {SZ.v m == brow p bid})
= rem_eq_mod bid p.rows;
  bid %^ p.rows

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
: GTot (natlt p.cols)
= bcol_bound p.rows p.cols p.blockItemsX bid;
  (bid / p.rows) * p.blockItemsX

inline_for_extraction noextract
let bcol_
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et) (bid : szlt (nblocks_ p))
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
: Ghost nat (requires true) (ensures fun c -> p.workChunk /? c)
=
  lemma_divides_product p.workChunk tid;
  assert p.workChunk /? (tid * p.workChunk);
  lemma_divides_product p.blockItemsX (bid / p.rows);
  assert p.blockItemsX /? bcol p bid;
  Math.prod_divides p.blockWidth p.workChunk p.blockItemsX;
  assert p.workChunk /? p.blockItemsX;
  Math.divides_chain p.workChunk p.blockItemsX (bcol p bid);
  lemma_divides_sum p.workChunk (bcol p bid) (tid * p.workChunk);
  bcol p bid + tid * p.workChunk

(* [c * s == 16] implies [c <= 16], with explicit nonlinear steps *)
let chunk_le_16 (c s : nat)
  : Lemma (requires c * s == 16) (ensures c <= 16)
  = if s = 0 then FStar.Math.Lemmas.mul_zero_right_is_zero c
    else FStar.Math.Lemmas.lemma_mult_le_left c 1 s

// Native vectors contain 16 bytes, so their element counts are comparable
// powers of two. Their maximum is therefore a multiple of both counts.
#push-options "--z3rlimit 30"
let native_chunk_cases (c : pos) (s : nat)
  : Lemma (requires c * s == 16)
          (ensures c == 1 \/ c == 2 \/ c == 4 \/ c == 8 \/ c == 16)
  = chunk_le_16 c s;
    if c = 1 then () else if c = 2 then () else if c = 4 then ()
    else if c = 8 then () else if c = 16 then () else ()

inline_for_extraction noextract
let working_chunk
  (tb tc : Type0) {| sized tb, has_vec_cpy tb, sized tc, has_vec_cpy tc |}
  : Tot (w : szp { w <= 16 /\ chunk tb /? w /\ chunk tc /? w /\
                   SZ.v w == max (SZ.v (chunk tb)) (SZ.v (chunk tc)) })
  = native_chunk_cases (chunk tb) (size #tb);
    native_chunk_cases (chunk tc) (size #tc);
    if chunk tb <=^ chunk tc then chunk tc else chunk tb
#pop-options

#push-options "--z3rlimit 15"
inline_for_extraction noextract
let tcol_
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (p : parameters et)
  (bid : szlt (nblocks_ p))
  (tid : szlt p.blockWidth)
: Pure sz (requires true) (ensures fun c -> SZ.v c == tcol p bid tid)
= assert (SZ.v p.workChunk <= 16);
  assert (SZ.v tid < SZ.v p.blockWidth /\ SZ.v p.blockWidth <= SZ.v max_threads);
  let c : (c : sz { SZ.v c == SZ.v tid * SZ.v p.workChunk }) = tid *^ p.workChunk in
  let b : (b : sz { SZ.v b == bcol p bid }) = bcol_ p bid in
  (* [c <= 1024 * 16] and [b < 10000], so [b + c] fits in 16 bits *)
  FStar.Math.Lemmas.lemma_mult_le_right (SZ.v p.workChunk) (SZ.v tid) 1024;
  FStar.Math.Lemmas.lemma_mult_le_left 1024 (SZ.v p.workChunk) 16;
  assert (SZ.v c <= 16384);
  assert_norm (pow2 16 == 65536);
  FStar.SizeT.fits_at_least_16 (SZ.v b + SZ.v c);
  b +^ c
#pop-options

// TODO actually use this definition, and make one for sz too
let threadItemsX
  (#et : Type0) {| d : scalar et, sized et, hvc : has_vec_cpy et |}
  (p : parameters et)
: Ghost nat (requires true) (ensures fun k -> p.workChunk /? k)
=
  Math.prod_cancel_divides p.blockItemsX p.workChunk p.blockWidth;
  p.blockItemsX / p.blockWidth

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

// TODO just use valid_csr_matrix, remove this definition
unfold
let well_formed
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et)
  (#nnz : sz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
: prop
= valid_csr_matrix p.rows p.shared (cast_pos col_ind) (cast_pos row_off)

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
: prop
=
  forall (k1 : natlt n1).
    let k2 = j + k1 / chunk et * step * chunk et + k1 % chunk et in
    k2 < n2 ==> chest2_col em k1 == seq_to_chest1 (seq_make_sparse row_ind (ematrix_col em2 k2))
