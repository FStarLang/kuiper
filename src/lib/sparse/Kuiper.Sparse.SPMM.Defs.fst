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
= bid / (p.cols `divup` p.blockItemsX)

inline_for_extraction noextract
let brow_
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et) (bid : szlt (nblocks_ p))
  (#_ : squash (fits (p.cols + p.blockItemsX)))
: Tot (m : sz {SZ.v m == brow p bid})
= bid /^ (p.cols `divup_` p.blockItemsX)

let bcol
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et) (bid : natlt (nblocks_ p))
: GTot (natlt p.cols) // por que Ghost?
= (bid % (p.cols `divup` p.blockItemsX)) * p.blockItemsX

inline_for_extraction noextract
let bcol_
  #et {| sized et, has_vec_cpy et |}
  (p : parameters et { size_req p }) (bid : szlt (nblocks_ p))
: Tot (n : sz {SZ.v n == bcol p bid})
= (bid %^ (p.cols `divup_` p.blockItemsX)) *^ p.blockItemsX

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
  lemma_divides_product p.blockItemsX (bid % (p.cols `divup` p.blockItemsX));
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
= bcol_ p bid +^ tid *^ chunk et

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
  = ()

let block_lemma_off whole block k off
  : Lemma (requires block /? whole /\ k * block < whole /\ off < block)
          (ensures k * block + off < whole)
  = ()

#push-options "--z3rlimit 10"
let offset_aligned_lemma_et
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (p : parameters et)
  (#n : nat)
  (x : gpu_array et n { aligned 16 x })
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
  prod_divides (chunk et) p.blockWidth p.blockItemsK;
  lineal_divides (chunk et) i' p.blockItemsK k;
  ()
#pop-options

let offset_aligned_lemma_sz
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (p : parameters et)
  (#n : nat)
  (x : gpu_array sz n { aligned 16 x })
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
  prod_divides (chunk sz) p.blockWidth p.blockItemsK;
  lineal_divides (chunk sz) i' p.blockItemsK k;
  ()

// TODO mejores nombress
let offset_aligned_lemma_et'
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (p : parameters et)
  (#n : nat)
  (x : gpu_array et n { aligned 16 x })
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
  ()

let offset_aligned_lemma_sz'
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (p : parameters et)
  (#n : nat)
  (x : gpu_array et n { aligned 16 x })
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
  ()

open Kuiper.EMatrix

let ematrix_tile_prop
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#m1 #n1 : nat { chunk et /? n1 })
  (#m2 #n2 : nat {  chunk et /? n2 })
  (em2 : ematrix et m2 n2)
  (row_ind : lseq nat m1 { in_bounds 0 m2 row_ind })
  (j : nat { chunk et /? j })
  (step : nat)
  (em : ematrix et m1 n1)
: GTot prop
=
  forall (k1 : natlt n1).
    let k2 = j + k1 / chunk et * step * chunk et + k1 % chunk et in
    k2 < n2 ==> ematrix_col em k1 == seq_make_sparse row_ind (ematrix_col em2 k2)