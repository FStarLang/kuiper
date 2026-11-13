module KPlay.Proof.Optimized.StorageAdapters

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Core
open FStar.Ghost

module A = Pulse.Lib.Array
module Mem = Kuiper.Array.Core
module Every = Kuiper.ForEvery
module B = Kuiper.Bijection
module S = FStar.Seq
module P = KPlay.Proof.Shared.Projection
module O = KPlay.Proof.Shared.Observations
module Storage = KPlay.Proof.Shared.Storage

inline_for_extraction noextract
let vector_binding (#a:Type0) (#n:pos) (array:A.array a)
  (offset:nat{offset + n <= A.length array}) (backing:lseq a (A.length array))
  (at:O.vector (Storage.cell a) n) (values:O.vector a n) : prop =
  forall (i:P.index n).
    (at i).Storage.buffer == array /\ (at i).Storage.offset == offset + i /\
    values i == S.index backing (offset + i)

ghost
fn vector_to_slice
  (#a:Type0) (#n:pos) (array:A.array a)
  (offset:nat{offset + n <= A.length array}) (backing:lseq a (A.length array))
  (at:O.vector (Storage.cell a) n) (values:O.vector a n) (fraction:perm)
  requires Storage.vector_owns fraction at values
  requires pure (vector_binding array offset backing at values)
  ensures Mem.pts_to_slice array #fraction offset (offset + n) (S.slice backing offset (offset + n))
{
  unfold Storage.vector_owns #a #n fraction at values;
  Every.forevery_ext #(P.index n)
    (fun i -> Storage.cell_owns fraction (at i) (values i))
    (fun i -> Mem.pts_to_cell array #fraction (offset + i) (S.index backing (offset + i)));
  Every.forevery_ext #(P.index n)
    (fun i -> Mem.pts_to_cell array #fraction (offset + i) (S.index backing (offset + i)))
    (fun i -> Mem.pts_to_cell array #fraction (offset + i)
      (S.index (S.slice backing offset (offset + n)) i));
  Every.forevery_rw_type (P.index n) (Kuiper.Common.natlt (offset + n - offset))
    (fun i -> Mem.pts_to_cell array #fraction (offset + i)
      (S.index (S.slice backing offset (offset + n)) i));
  Mem.cells_to_nonempty_slice array #fraction offset (offset + n)
    #(S.slice backing offset (offset + n));
}

inline_for_extraction noextract
let matrix_binding (#a:Type0) (#m #n:pos) (array:A.array a)
  (offset:nat{offset + m * n <= A.length array}) (backing:lseq a (A.length array))
  (at:O.matrix (Storage.cell a) m n) (values:O.matrix a m n) : prop =
  forall (i:P.index m) (j:P.index n).
    (at i j).Storage.buffer == array /\ (at i j).Storage.offset == offset + i * n + j /\
    values i j == S.index backing (offset + i * n + j)

ghost
fn matrix_to_slice
  (#a:Type0) (#m #n:pos) (array:A.array a)
  (offset:nat{offset + m * n <= A.length array}) (backing:lseq a (A.length array))
  (at:O.matrix (Storage.cell a) m n) (values:O.matrix a m n) (fraction:perm)
  requires Storage.matrix_owns fraction at values
  requires pure (matrix_binding array offset backing at values)
  ensures Mem.pts_to_slice array #fraction offset (offset + m * n)
    (S.slice backing offset (offset + m * n))
{
  unfold Storage.matrix_owns #a #m #n fraction at values;
  Every.forevery_map #(P.index m)
    (fun i -> Storage.vector_owns #a #n fraction (at i) (values i))
    (fun i -> Every.forevery (P.index n) (fun j -> Storage.cell_owns fraction (at i j) (values i j)))
    fn i { unfold Storage.vector_owns #a #n fraction (at i) (values i); };
  Every.forevery_flatten #(P.index m) #(P.index n)
    (fun i j -> Storage.cell_owns fraction (at i j) (values i j));
  Every.forevery_iso (B.bij_nat_prod #m #n)
    (fun (ij:P.index m & P.index n) -> Storage.cell_owns fraction (at ij._1 ij._2) (values ij._1 ij._2));
  Every.forevery_ext #(P.index (m * n))
    (fun i -> Storage.cell_owns fraction
      (at ((B.bij_nat_prod #m #n).B.gg i)._1 ((B.bij_nat_prod #m #n).B.gg i)._2)
      (values ((B.bij_nat_prod #m #n).B.gg i)._1 ((B.bij_nat_prod #m #n).B.gg i)._2))
    (fun i -> Mem.pts_to_cell array #fraction (offset + i) (S.index backing (offset + i)));
  Every.forevery_ext #(P.index (m * n))
    (fun i -> Mem.pts_to_cell array #fraction (offset + i) (S.index backing (offset + i)))
    (fun i -> Mem.pts_to_cell array #fraction (offset + i)
      (S.index (S.slice backing offset (offset + m * n)) i));
  Every.forevery_rw_type (P.index (m * n)) (Kuiper.Common.natlt (offset + m * n - offset))
    (fun i -> Mem.pts_to_cell array #fraction (offset + i)
      (S.index (S.slice backing offset (offset + m * n)) i));
  Mem.cells_to_nonempty_slice array #fraction offset (offset + m * n)
    #(S.slice backing offset (offset + m * n));
}

ghost
fn join_slices
  (#a:Type0) (array:A.array a) (backing:lseq a (A.length array)) (fraction:perm)
  (lo mid hi:nat{lo <= mid /\ mid <= hi /\ hi <= A.length array})
  requires Mem.pts_to_slice array #fraction lo mid (S.slice backing lo mid)
  requires Mem.pts_to_slice array #fraction mid hi (S.slice backing mid hi)
  ensures Mem.pts_to_slice array #fraction lo hi (S.slice backing lo hi)
{
  Mem.slice_concat array #fraction lo mid hi;
  assert pure (S.equal (S.append (S.slice backing lo mid) (S.slice backing mid hi))
    (S.slice backing lo hi));
  S.lemma_eq_elim (S.append (S.slice backing lo mid) (S.slice backing mid hi))
    (S.slice backing lo hi);
  rewrite Mem.pts_to_slice array #fraction lo hi
    (S.append (S.slice backing lo mid) (S.slice backing mid hi))
    as Mem.pts_to_slice array #fraction lo hi (S.slice backing lo hi);
}

ghost
fn split_slices
  (#a:Type0) (array:A.array a) (backing:lseq a (A.length array)) (fraction:perm)
  (lo mid hi:nat{lo <= mid /\ mid <= hi /\ hi <= A.length array})
  requires Mem.pts_to_slice array #fraction lo hi (S.slice backing lo hi)
  ensures Mem.pts_to_slice array #fraction lo mid (S.slice backing lo mid)
  ensures Mem.pts_to_slice array #fraction mid hi (S.slice backing mid hi)
{
  assert pure (S.equal (S.slice backing lo hi)
    (S.append (S.slice backing lo mid) (S.slice backing mid hi)));
  S.lemma_eq_elim (S.slice backing lo hi)
    (S.append (S.slice backing lo mid) (S.slice backing mid hi));
  rewrite Mem.pts_to_slice array #fraction lo hi (S.slice backing lo hi)
    as Mem.pts_to_slice array #fraction lo hi
      (S.append (S.slice backing lo mid) (S.slice backing mid hi));
  Mem.slice_split array #fraction lo mid hi;
}

ghost
fn slice_to_vector
  (#a:Type0) (#n:pos) (array:A.array a)
  (offset:nat{offset + n <= A.length array}) (backing:lseq a (A.length array))
  (at:O.vector (Storage.cell a) n) (values:O.vector a n) (fraction:perm)
  requires Mem.pts_to_slice array #fraction offset (offset + n) (S.slice backing offset (offset + n))
  requires pure (vector_binding array offset backing at values)
  ensures Storage.vector_owns fraction at values
{
  Mem.slice_to_cells array #fraction offset (offset + n) #(S.slice backing offset (offset + n));
  Every.forevery_rw_type (Kuiper.Common.natlt (offset + n - offset)) (P.index n)
    (fun i -> Mem.pts_to_cell array #fraction (offset + i)
      (S.index (S.slice backing offset (offset + n)) i));
  Every.forevery_ext #(P.index n)
    (fun i -> Mem.pts_to_cell array #fraction (offset + i)
      (S.index (S.slice backing offset (offset + n)) i))
    (fun i -> Storage.cell_owns fraction (at i) (values i));
  fold Storage.vector_owns #a #n fraction at values;
  drop_ (Mem.array_exists array);
}

ghost
fn slice_to_matrix
  (#a:Type0) (#m #n:pos) (array:A.array a)
  (offset:nat{offset + m * n <= A.length array}) (backing:lseq a (A.length array))
  (at:O.matrix (Storage.cell a) m n) (values:O.matrix a m n) (fraction:perm)
  requires Mem.pts_to_slice array #fraction offset (offset + m * n)
    (S.slice backing offset (offset + m * n))
  requires pure (matrix_binding array offset backing at values)
  ensures Storage.matrix_owns fraction at values
{
  Mem.slice_to_cells array #fraction offset (offset + m * n)
    #(S.slice backing offset (offset + m * n));
  Every.forevery_rw_type (Kuiper.Common.natlt (offset + m * n - offset)) (P.index (m * n))
    (fun i -> Mem.pts_to_cell array #fraction (offset + i)
      (S.index (S.slice backing offset (offset + m * n)) i));
  Every.forevery_ext #(P.index (m * n))
    (fun i -> Mem.pts_to_cell array #fraction (offset + i)
      (S.index (S.slice backing offset (offset + m * n)) i))
    (fun i -> Storage.cell_owns fraction
      (at ((B.bij_nat_prod #m #n).B.gg i)._1 ((B.bij_nat_prod #m #n).B.gg i)._2)
      (values ((B.bij_nat_prod #m #n).B.gg i)._1 ((B.bij_nat_prod #m #n).B.gg i)._2));
  Every.forevery_iso_back (B.bij_nat_prod #m #n)
    (fun (ij:P.index m & P.index n) -> Storage.cell_owns fraction (at ij._1 ij._2) (values ij._1 ij._2));
  Every.forevery_unflatten #(P.index m) #(P.index n)
    (fun i j -> Storage.cell_owns fraction (at i j) (values i j));
  Every.forevery_map #(P.index m)
    (fun i -> Every.forevery (P.index n) (fun j -> Storage.cell_owns fraction (at i j) (values i j)))
    (fun i -> Storage.vector_owns #a #n fraction (at i) (values i))
    fn i { fold Storage.vector_owns #a #n fraction (at i) (values i); };
  fold Storage.matrix_owns #a #m #n fraction at values;
  drop_ (Mem.array_exists array);
}
