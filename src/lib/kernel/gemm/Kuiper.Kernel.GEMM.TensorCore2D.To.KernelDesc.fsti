module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Kernel.GEMM.Tiled.Common.Vec
open Kuiper.TensorCore

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module MS = Kuiper.Spec.GEMM

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc

let in_lane
  (rows cols : nat)
  (lane : natlt warp_size)
  (ij : natlt rows & natlt cols)
  : prop
= (ij._1 * cols + ij._2) % warp_size == lane

let own_lane_cells
  (#et : Type0) {| scalar et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  ([@@@mkey] m : array2 et l)
  (em : chest2 et rows cols)
  (lane : natlt warp_size)
  : slprop
= forall+ (ij : (natlt rows & natlt cols){in_lane rows cols lane ij}).
    tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2)

let live_lane_cells
  (#et : Type0) {| scalar et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  ([@@@mkey] m : array2 et l)
  (lane : natlt warp_size)
  : slprop
= exists* (em : chest2 et rows cols). own_lane_cells m em lane

inline_for_extraction noextract
let output_fragment
  (#et : Type0) {| scalar et |}
  (#m #n : nat)
  (gD : array2 et (rm m n))
  (bm bn tm tn wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (bid : natlt (m / bm * (n / bn)))
  (wid : natlt (bm / (wm * tm) * (bn / (wn * tn))))
  (mi : natlt wm)
  (nj : natlt wn)
= array2_subtile
    (warp_tile (block_tile gD bm bn bid) (wm * tm) (wn * tn) wid)
    tm tn mi nj

let epilogue_chest
  (#et_cd #et_acc : Type0)
  {| scalar et_cd, scalar et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
  (#rows #cols : nat)
  (eC : chest2 et_cd rows cols)
  (eAcc : chest2 et_acc rows cols)
  : chest2 et_cd rows cols
= mk2 (fun i j -> comb (acc2 eC i j) (acc2 eAcc i j))

inline_for_extraction noextract
fn epilogue_fragment_from_warp
  (#et_cd #et_acc : Type0)
  {| scd : scalar et_cd, real_like et_cd,
     sacc : scalar et_acc, real_like et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #n : szp)
  (c : array2 et_cd (rm m n))
  (#_ : squash (SZ.fits (m * n)))
  (bm bn rows cols wm wn : szp)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * rows /?+ bm /\ wn * cols /?+ bn))
  (mrow : szlt (m / bm))
  (mcol : szlt (n / bn))
  (warpRow : szlt (bm / (wm * rows)))
  (warpCol : szlt (bn / (wn * cols)))
  (bid : szlt (m / bm * (n / bn)))
  (wid : szlt (bm / (wm * rows) * (bn / (wn * cols))))
  (#_ : squash (
    SZ.v mrow == SZ.v bid / (SZ.v n / SZ.v bn) /\
    SZ.v mcol == SZ.v bid % (SZ.v n / SZ.v bn) /\
    SZ.v warpRow == SZ.v wid / (SZ.v bn / (SZ.v wn * SZ.v cols)) /\
    SZ.v warpCol == SZ.v wid % (SZ.v bn / (SZ.v wn * SZ.v cols))))
  (#fC : perm)
  (#eC : chest2 et_cd m n)
  (#rC : chest2 real m n)
  (#_ : squash (eC %~ rC))
  (#lAcc : layout2 rows cols) {| T.ctlayout lAcc |}
  (acc : array2 et_acc lAcc)
  (#eAcc : chest2 et_acc rows cols)
  (#rAcc : chest2 real rows cols)
  (#_ : squash (eAcc %~ rAcc))
  (d : array2 et_cd (rm m n))
  (idx : szlt (wm * wn))
  (lane : szlt warp_size)
  (#_ : squash (SZ.fits (rows * cols + warp_size)))
  preserves
    gpu **
    c |-> Frac fC eC **
    acc |-> Frac (1.0R /. warp_size) eAcc
  requires
    live_lane_cells
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      lane
  ensures
    own_lane_cells
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (epilogue_chest comb
        (ematrix_subtile
          (ematrix_subtile
            (ematrix_subtile eC bm bn (SZ.v mrow) (SZ.v mcol))
            (wm * rows) (wn * cols) (SZ.v warpRow) (SZ.v warpCol))
          rows cols (SZ.v idx / wn) (SZ.v idx % wn))
        eAcc)
      lane **
    pure (
      epilogue_chest comb
        (ematrix_subtile
          (ematrix_subtile
            (ematrix_subtile eC bm bn (SZ.v mrow) (SZ.v mcol))
            (wm * rows) (wn * cols) (SZ.v warpRow) (SZ.v warpCol))
          rows cols (SZ.v idx / wn) (SZ.v idx % wn))
        eAcc
      %~
      chest_comb comb_r
        (ematrix_subtile
          (ematrix_subtile
            (ematrix_subtile rC bm bn (SZ.v mrow) (SZ.v mcol))
            (wm * rows) (wn * cols) (SZ.v warpRow) (SZ.v warpCol))
          rows cols (SZ.v idx / wn) (SZ.v idx % wn))
        rAcc)

inline_for_extraction noextract
unfold let shmems_desc_to
  (et_ab et_acc : Type0) {| sized et_ab, sized et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  : list shmem_desc
=
  assert (SZ.fits ((nthr / warp_size) * tm));
  [
    SHArray et_ab (bm *^ bk);
    SHArray et_ab (bk *^ bn);
    SHArray et_acc ((nthr /^ warp_size) *^ tm *^ tn);
  ]

inline_for_extraction noextract
let scratch_layout
  (tm tn nthr : szp)
  (#_ : squash (warp_size /?+ nthr))
  : layout2 ((nthr / warp_size) * tm) tn
= rm ((SZ.v nthr / warp_size) * SZ.v tm) (SZ.v tn)

inline_for_extraction noextract
let scratch_matrix
  (#et_ab #et_acc : Type0)
  {| sized et_ab, sized et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  : array2 et_acc (scratch_layout tm tn nthr)
= from_array (scratch_layout tm tn nthr) (fst (snd (snd sh)))

inline_for_extraction noextract
let scratch_tile
  (#et_ab #et_acc : Type0)
  {| sized et_ab, sized et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (wid : natlt (nthr / warp_size))
= array2_subtile (scratch_matrix bm bn bk tm tn nthr sh)
    (SZ.v tm) (SZ.v tn) wid 0

inline_for_extraction noextract
let scratch_tile_st
  (#et_ab #et_acc : Type0)
  {| sized et_ab, sized et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (wid : szlt (nthr /^ warp_size))
= scratch_tile bm bn bk tm tn nthr sh (SZ.v wid)

let output_lane_live
  (#et : Type0) {| scalar et |}
  (#m #n : nat)
  (gD : array2 et (rm m n))
  (bm bn tm tn wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (bid : natlt (m / bm * (n / bn)))
  (tid : natlt (bm / (wm * tm) * (bn / (wn * tn)) * warp_size))
  : slprop
= forall+ (mi : natlt wm) (nj : natlt wn).
    live_lane_cells
      (output_fragment gD bm bn tm tn wm wn bid (tid / warp_size) mi nj)
      (tid % warp_size)

let output_lane_approximates
  (#et : Type0) {| scalar et, real_like et |}
  (#m #n : nat)
  (gD : array2 et (rm m n))
  (bm bn tm tn wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (bid : natlt (m / bm * (n / bn)))
  (tid : natlt (bm / (wm * tm) * (bn / (wn * tn)) * warp_size))
  (rD : chest2 real (wm * tm) (wn * tn))
  : slprop
= forall+ (mi : natlt wm) (nj : natlt wn).
    exists* (eD : chest2 et tm tn).
      own_lane_cells
        (output_fragment gD bm bn tm tn wm wn bid (tid / warp_size) mi nj)
        eD
        (tid % warp_size) **
      pure (eD %~ ematrix_subtile rD tm tn mi nj)

let scratch_tile_live
  (#et_ab #et_acc : Type0)
  {| sized et_ab, scalar et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (tid : natlt nthr)
  : slprop
= exists* (eAcc : chest2 et_acc tm tn).
    scratch_tile bm bn bk tm tn nthr sh (tid / warp_size)
      |-> Frac (1.0R /. warp_size) eAcc

unfold
let kpre1_to
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, real_like et_ab,
     scalar et_cd, real_like et_cd |}
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n))
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp{
    constraints bm bn bk tm tn tk wm wn})
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (nblk : szp{SZ.v nblk == m / bm * (n / bn)})
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (bid : natlt nblk)
  (tid : natlt nthr)
  : slprop
= gA |-> Frac (fA /. (nblk * nthr)) eA **
  gB |-> Frac (fB /. (nblk * nthr)) eB **
  gC |-> Frac (fC /. (nblk * nthr)) eC **
  output_lane_live gD bm bn tm tn wm wn bid tid **
  pure (aligned 16 (core gA)) **
  pure (aligned 16 (core gB)) **
  pure (eA %~ rA) **
  pure (eB %~ rB) **
  pure (eC %~ rC)

unfold
let kpost1_to
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n))
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp{
    constraints bm bn bk tm tn tk wm wn})
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (nblk : szp{SZ.v nblk == m / bm * (n / bn)})
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (bid : natlt nblk)
  (tid : natlt nthr)
  : slprop
= gA |-> Frac (fA /. (nblk * nthr)) eA **
  gB |-> Frac (fB /. (nblk * nthr)) eB **
  gC |-> Frac (fC /. (nblk * nthr)) eC **
  output_lane_approximates
    gD bm bn tm tn wm wn bid tid
    (ematrix_subtile
      (ematrix_subtile
        (MS.mmcomb comb_r rC rA rB)
        bm bn
        (bid / (n / bn))
        (bid % (n / bn)))
      (wm * tm) (wn * tn)
      ((tid / warp_size) / (bn / (wn * tn)))
      ((tid / warp_size) % (bn / (wn * tn))))

let shared_thread_live
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, scalar et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (tid : natlt nthr)
  : slprop
= live_c_shmem (fst sh) #(1.0R /. nthr) **
  live_c_shmem (fst (snd sh)) #(1.0R /. nthr) **
  scratch_tile_live bm bn bk tm tn nthr sh tid

unfold
let kpre_to
  (#et_ab #et_cd #et_acc : Type0)
  {| scalar et_ab, real_like et_ab,
     scalar et_cd, real_like et_cd, scalar et_acc |}
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n))
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp{
    constraints bm bn bk tm tn tk wm wn})
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits (
    ((bm / (wm * tm) * (bn / (wn * tn)) * warp_size) / warp_size)
      * tm * tn)))
  (#_ : squash (
    warp_size /?+ (bm / (wm * tm) * (bn / (wn * tn)) * warp_size)))
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (nblk : szp{SZ.v nblk == m / bm * (n / bn)})
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (bid : natlt nblk)
  (tid : natlt nthr)
  : slprop
= kpre1_to gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn fA fB fC rA rB rC nblk nthr bid tid **
  shared_thread_live bm bn bk tm tn nthr sh tid

unfold
let kpost_to
  (#et_ab #et_cd #et_acc : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd, scalar et_acc |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n))
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp{
    constraints bm bn bk tm tn tk wm wn})
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits (
    ((bm / (wm * tm) * (bn / (wn * tn)) * warp_size) / warp_size)
      * tm * tn)))
  (#_ : squash (
    warp_size /?+ (bm / (wm * tm) * (bn / (wn * tn)) * warp_size)))
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (nblk : szp{SZ.v nblk == m / bm * (n / bn)})
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (bid : natlt nblk)
  (tid : natlt nthr)
  : slprop
= kpost1_to comb_r gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn fA fB fC rA rB rC nblk nthr bid tid **
  shared_thread_live bm bn bk tm tn nthr sh tid

ghost
fn setup_to
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, real_like et_ab,
     scalar et_cd, real_like et_cd |}
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (gC : array2 et_cd (rm m n))
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (#_ : squash (SZ.fits (m * n)))
  (bm bn bk tm tn tk wm wn : szp{
    constraints bm bn bk tm tn tk wm wn})
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp{SZ.v nblk == m / bm * (n / bn)})
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  ()
  norewrite
  requires
    gA |-> Frac fA eA ** pure (eA %~ rA) **
    gB |-> Frac fB eB ** pure (eB %~ rB) **
    gC |-> Frac fC eC ** pure (eC %~ rC) **
    live gD
  ensures
    (forall+ (bid : natlt nblk) (tid : natlt nthr).
      kpre1_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid) **
    pure (SZ.fits ((rm m n).ulen))

ghost
fn block_setup_to
  (#et_ab #et_cd #et_acc : Type0)
  {| scalar et_ab, real_like et_ab,
     scalar et_cd, real_like et_cd, scalar et_acc |}
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n))
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp{
    constraints bm bn bk tm tn tk wm wn})
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits (
    ((bm / (wm * tm) * (bn / (wn * tn)) * warp_size) / warp_size)
      * tm * tn)))
  (#_ : squash (
    warp_size /?+ (bm / (wm * tm) * (bn / (wn * tn)) * warp_size)))
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (nblk : szp{SZ.v nblk == m / bm * (n / bn)})
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpre1_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid)
  ensures
    (forall+ (tid : natlt nthr).
      kpre_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr sh bid tid) **
    emp

ghost
fn block_teardown_to
  (#et_ab #et_cd #et_acc : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd, scalar et_acc |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n))
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp{
    constraints bm bn bk tm tn tk wm wn})
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits (
    ((bm / (wm * tm) * (bn / (wn * tn)) * warp_size) / warp_size)
      * tm * tn)))
  (#_ : squash (
    warp_size /?+ (bm / (wm * tm) * (bn / (wn * tn)) * warp_size)))
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (nblk : szp{SZ.v nblk == m / bm * (n / bn)})
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
      kpost_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr sh bid tid) **
    emp
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid)

ghost
fn teardown_to
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n))
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (#_ : squash (SZ.fits (m * n)))
  (bm bn bk tm tn tk wm wn : szp{
    constraints bm bn bk tm tn tk wm wn})
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp{SZ.v nblk == m / bm * (n / bn)})
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  ()
  norewrite
  requires
    (forall+ (bid : natlt nblk) (tid : natlt nthr).
      kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid) **
    pure (SZ.fits ((rm m n).ulen))
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> Frac fC eC **
    (exists* (eD : chest2 et_cd m n).
      gD |-> eD ** pure (eD %~ MS.mmcomb comb_r rC rA rB))
