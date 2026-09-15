module Kuiper.Kernel.GEMM.BlockTiling1D.Compute

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Chest
open Kuiper.Tensor.Tiling
open Kuiper.Kernel.GEMMGPU.Type
open Kuiper.Kernel.GEMM.FlipFlopColumns
module T = Kuiper.Tensor
module MS = Kuiper.Spec.GEMM

(* Supply tile-index bounds while checking refined specifications. *)
val tile_quotient (n : nat) (tile : pos)
  : Lemma (n * tile / tile == n) [SMTPat (n * tile / tile)]

(* Updating one index leaves every other prefix predicate unchanged. *)
val prefix_frame (n : nat) (done : nat)
  (p : natlt n -> bool -> GTot slprop)
  : Lemma (forall (i : natlt n{i =!= done}).
      p i (i < done + 1) == p i (i < done))

val advance_columns
  (#ta #tb #tacc : Type0) {| scalar tacc |}
  (tile : valid_tile) (mapA : ta -> tacc) (mapB : tb -> tacc)
  (#m #k #n : nat)
  (eA : chest2 ta (m * tile) (k * tile))
  (eB : chest2 tb (k * tile) (n * tile))
  (row : natlt m) (col : natlt n) (j : natlt tile) (bk : natlt k)
  (s0 s1 : lseq tacc tile)
  : Lemma
    (requires
      (forall (i : natlt tile). s0 @! i ==
        MS.__gmatmul_single zero mul add (chest_map mapA eA) (chest_map mapB eB)
          (row * tile + i) (col * tile + j) (bk * tile)) /\
      (forall (i : natlt tile). s1 @! i ==
        MS.__gmatmul_single (s0 @! i) mul add
          (chest_map mapA (ematrix_subtile eA tile tile row bk))
          (chest_map mapB (ematrix_subtile eB tile tile bk col)) i j tile))
    (ensures forall (i : natlt tile). s1 @! i ==
      MS.__gmatmul_single zero mul add (chest_map mapA eA) (chest_map mapB eB)
        (row * tile + i) (col * tile + j) ((bk + 1) * tile))

inline_for_extraction noextract
fn bring_2cols
  (tile : valid_tile)
  (#ta #tb #tacc : Type0) {| scalar ta, scalar tb, scalar tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (#m #n #k : erased nat)
  (#lA : layout2 (m * tile) (k * tile))
  (#lB : layout2 (k * tile) (n * tile))
  {| T.ctlayout lA, T.ctlayout lB |}
  (gA : array2 ta lA)
  (gB : array2 tb lB)
  (#l1 #l2 : layout2 tile tile)
  {| T.ctlayout l1, T.ctlayout l2 |}
  (sa1 : array2 tacc l1) (sa2 : array2 tacc l2)
  (mm : szlt m)
  (kk : szlt k)
  (nn : szlt n)
  (tid : szlt tile)
  (#fA #fB : perm)
  (#eA : chest2 ta (m * tile) (k * tile))
  (#eB : chest2 tb (k * tile) (n * tile))
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    own_1_col sa1 tid **
    own_1_col sa2 tid
  ensures
    own_1_col_at sa1 (chest_map mapA (ematrix_subtile eA tile tile mm kk)) tid **
    own_1_col_at sa2 (chest_map mapB (ematrix_subtile eB tile tile kk nn)) tid

inline_for_extraction noextract
fn subproduct_cols
  (#et : Type0) {| scalar et |}
  (tile : sz)
  (acc : array et)
  (#l1 #l2 : layout2 tile tile)
  {| T.ctlayout l1, T.ctlayout l2 |}
  (m1 : array2 et l1)
  (m2 : array2 et l2)
  (j : szlt tile)
  (#acc0 : erased (lseq et tile))
  (#em1 #em2 : chest2 et tile tile)
  (#f : perm)
  preserves
    gpu **
    m1 |-> Frac f em1 **
    m2 |-> Frac f em2
  requires
    acc |-> acc0
  ensures
    exists* (acc' : lseq et tile).
      acc |-> acc' ** pure (forall (i : natlt tile).
        acc' @! i == MS.__gmatmul_single (acc0 @! i) mul add em1 em2 i j tile)
