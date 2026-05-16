module Kuiper.Sparse.SPMM.Defs

(* Shared type definitions for the SPMM kernel. *)

#lang-pulse

open Kuiper
module SZ = Kuiper.SizeT
open Kuiper.Sparse
open Kuiper.Math { even, odd }

let slice_live
  (#et : Type0)
  (#l : nat)
  (a : gpu_array et l)
  (#[FStar.Tactics.exact (`1.0R)] f : perm)
  (i j : nat)
  : slprop
  = exists* s. gpu_pts_to_slice a #f i j s

let array_live_cell
  (#et : Type0)
  (#l : nat)
  (a :gpu_array et l)
  (#[FStar.Tactics.exact (`1.0R)] f : perm)
  (i : natlt l)
  : slprop
  = exists* v. gpu_pts_to_cell a #f i v

inline_for_extraction
type parameters = {
  rows : szp;
  shared : szp;
  cols : szp;
  blockItemsK : szp;
  blockItemsX : szp;
  blockWidth : (k : szp {k /? blockItemsK /\ k /? blockItemsX});
}

(* Shadow lseq to make it erased. *)
let lseq (a:Type) (n:nat) = erased (Seq.lseq a n)


let nblocks_ (p : parameters) : GTot pos
// = p.rows * ((p.cols + p.blockItemsX - 1) / p.blockItemsX)
= p.rows * (p.cols `divup` p.blockItemsX)

let nthreads_ (p : parameters) : GTot pos
= p.blockWidth

let allthreads_ (p : parameters) : GTot pos
= nblocks_ p * nthreads_ p

inline_for_extraction noextract
let size_req (p : parameters) =
    nblocks_ p <= max_blocks /\
    p.blockWidth <= max_threads /\
    p.rows < 10000 /\
    p.shared < 10000 /\
    p.cols < 10000 /\
    p.blockItemsK < 10000 /\
    p.blockItemsX < 10000

inline_for_extraction noextract
let nblocks (p : parameters)
: Pure (szle max_blocks)
  (requires size_req p)
  (ensures fun r -> SZ.v r == nblocks_ p)
= p.rows *^ (p.cols `divup_` p.blockItemsX)

inline_for_extraction noextract
let nthreads (p : parameters{size_req p})
: Pure (szle max_threads)
  (requires size_req p)
  (ensures fun r -> SZ.v r == nthreads_ p)
= p.blockWidth

inline_for_extraction noextract
let allthreads (p : parameters{size_req p})
: Pure sz
  (requires size_req p)
  (ensures fun r -> SZ.v r == allthreads_ p)
= nblocks p *^ nthreads p

let brow (p : parameters) (bid : natlt (nblocks_ p))
: GTot (natlt p.rows)
= bid / (p.cols `divup` p.blockItemsX)

inline_for_extraction noextract
let brow_ (p : parameters) (bid : szlt (nblocks_ p))
  (#_ : squash (fits (p.cols + p.blockItemsX)))
: Tot (m : sz {SZ.v m == brow p bid})
= bid /^ (p.cols `divup_` p.blockItemsX)

let bcol (p : parameters) (bid : natlt (nblocks_ p))
: GTot (natlt p.cols)
= (bid % (p.cols `divup` p.blockItemsX)) * p.blockItemsX

inline_for_extraction noextract
let bcol_ (p : parameters { size_req p }) (bid : szlt (nblocks_ p))
: Tot (n : sz {SZ.v n == bcol p bid})
= (bid %^ (p.cols `divup_` p.blockItemsX)) *^ p.blockItemsX

// MAYBE definir threadItemsX?

(* Description of shared memory used in this kernel. *)
inline_for_extraction noextract
let shmems_desc (et:Type0) {| sized et |}
  (p : parameters) : list shmem_desc = [
  SHArray et p.blockItemsK;
  // TODO podemos parametrizar este tipo?
  SHArray sz p.blockItemsK;
]

unfold
let well_formed
  (p : parameters)
  (#nnz : sz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
: prop
= valid_smatrix p.rows p.shared (cast_pos col_ind) (cast_pos row_off)

let block_lemma whole block k
  : Lemma (requires block /? whole /\ k * block < whole)
          (ensures k * block + block <= whole)
  = ()

let block_lemma_off whole block k off
  : Lemma (requires block /? whole /\ k * block < whole /\ off < block)
          (ensures k * block + off < whole)
  = ()
