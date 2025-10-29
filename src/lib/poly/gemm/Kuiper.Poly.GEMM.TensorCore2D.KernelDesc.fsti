module Kuiper.Poly.GEMM.TensorCore2D.KernelDesc

#lang-pulse

#set-options "--z3rlimit 40"

open Kuiper

open Kuiper.Matrix.Reprs
module R = Kuiper.Matrix.Reprs
open Kuiper.TensorCore

module SZ = Kuiper.SizeT
open Kuiper.Matrix.Reprs.Type
open Kuiper.Math { even, odd, even_2x, odd_2x1 }

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Matrix

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier

open Kuiper.Matrix.Reprs
module R = Kuiper.Matrix.Reprs

open Kuiper.EMatrix
open Kuiper.VArray {
  varray,
  varray_pts_to,
  varray_pts_to_cell
}
open Kuiper.TensorCore
open Kuiper.Float16
open Kuiper.Matrix.Tiling

open Kuiper.Poly.GEMM.Copy.Vec
open Kuiper.Poly.GEMM.Tiled.Common.Vec

open Pulse.Lib.Array
open Pulse.Lib.Trade

open Kuiper.Bijection

type constraints (bm bn bk tm tn tk wm wn : szp) : prop =
  tm /?+ bm /\
  tn /?+ bn /\
  tk /?+ bk /\
  wm * tm /?+ bm /\
  wn * tn /?+ bn /\
  SZ.fits (wm * wn)

let live_warp_tile
  (#et : Type0) {| scalar et |}
  // Since this is an slprop, I would like to not erase the nat.
  // Unfortunately, when unfolding live_warp_tile, after passing
  // a (reveal x) as argument, this leads to (reveal (hide (reveal x)))
  // which creates problems with type equalities.
  (#rows : erased nat)
  (#cols : nat)
  (#lC : mlayout rows cols)
  (gC : gpu_matrix et lC)
  (bm : pos{bm /?+ rows})
  (bn : pos{bn /?+ cols})
  (tm : pos{tm /?+ bm})
  (tn : pos{tn /?+ bn})
  (wm : pos{wm * tm /?+ bm})
  (wn : pos{wn * tn /?+ bn})
  (bid : natlt ((rows/bm) * (cols/bn)))
  (wid : natlt (bm/(wm*tm) * (bn/(wn*tn))))
  : slprop
  =
  live (warp_tile (block_tile gC bm bn bid) (wm*tm) (wn*tn) wid) #(1.0R /. warp_size)

let barrier_p
  (#et : Type0) {| has_vec_cpy et |}
  (#bm #bn #bk : szp)
  (#l1 : mlayout bm bk)
  (#l2 : mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  (nthr : pos)
  : B.barrier_side nthr =
  fun it tid ->
    if even it then
      (exists* (x : ematrix _ _ _). m1 |-> Frac (1.0R /. nthr) x) **
      (exists* (x : ematrix _ _ _). m2 |-> Frac (1.0R /. nthr) x)
    else
      live_tile_stride_cells m1 nthr tid **
      live_tile_stride_cells m2 nthr tid

let barrier_q
  (#et : Type0) {| has_vec_cpy et |}
  (#bm #bn #bk : szp)
  (#l1 : mlayout bm bk)
  (#l2 : mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  (nthr : pos)
  : B.barrier_side nthr =
  fun it tid -> barrier_p m1 m2 nthr (it+1) tid (* flip flop *)

let barrier_tok
  (#et : Type0) {| has_vec_cpy et |}
  (#bm #bn #bk : szp)
  (* This is defined over the base shared gpu_arrays, as
  this spec must make sense before the arrays are viewed as
  a matrix. *)
  (l1 : full_mlayout bm bk)
  (l2 : full_mlayout bk bn)
  (sar1 : gpu_array et (bm * bk))
  (sar2 : gpu_array et (bk * bn))
  (it : nat)
  (nthr : pos)
  (tid : natlt nthr)
  : slprop
  =
  B.barrier_tok (barrier_p (from_array l1 sar1) (from_array l2 sar2) nthr)
                (barrier_q (from_array l1 sar1) (from_array l2 sar2) nthr)
                it tid

unfold
let kpre1
  (#et_ab #et_c : Type0) {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (fA fB : perm)
  (nthr : nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (bid : enatlt (rows/bm * (cols/bn)))
  (tid : enatlt nthr)
  : slprop
  =
  gA |-> Frac (fA /. (rows/bm * (cols/bn) * nthr)) eA **
  gB |-> Frac (fB /. (rows/bm * (cols/bn) * nthr)) eB **
  live_warp_tile gC bm bn tm tn wm wn bid (tid/warp_size) **
  pure (aligned 16 (core gA)) **
  pure (aligned 16 (core gB))

unfold
let kpre
  (#et_ab #et_c : Type0) {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (fA fB : perm)
  (nthr : nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (#_ : squash (1.0R /. nthr >=. 0.0R)) // to help SMT, this is obvious
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt nthr)
  : slprop
  =
  kpre1 gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr bid tid **
  (exists* (x : seq et_ab). gpu_pts_to_array (fst sh)       #(1.0R /. nthr) x) **
  (exists* (x : seq et_ab). gpu_pts_to_array (fst (snd sh)) #(1.0R /. nthr) x) **
  barrier_tok (R.row_major bm bk) (R.row_major bk bn) (fst sh) (fst (snd sh)) 0 nthr tid

ghost
fn setup
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c rows cols)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /? rows))
  (#_ : squash (bn /? cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (fA fB : perm)
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (bid : natlt nblk)
             (tid : natlt nthr).
      kpre1 gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr bid tid) **
    emp (* frame *)

ghost
fn block_setup
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c rows cols)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size /\ nthr <= 1024})
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    block_setup_tok nthr **
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpre1 gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr bid tid)
  ensures
    block_setup_tok nthr **
    (forall+ (tid : natlt nthr).
      kpre gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr sh bid tid) **
    emp (* frame *)

unfold
let kpost1
  (#et_ab #et_c : Type0) {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (fA fB : perm)
  (nthr : nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (bid : enatlt (rows/bm * (cols/bn)))
  (tid : enatlt nthr)
  : slprop
  =
  gA |-> Frac (fA /. (rows/bm * (cols/bn) * nthr)) eA **
  gB |-> Frac (fB /. (rows/bm * (cols/bn) * nthr)) eB **
  live_warp_tile gC bm bn tm tn wm wn bid (tid/warp_size)

unfold
let kpost
  (#et_ab #et_c : Type0) {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn
                 /\ 2 * (shared / bk) >= 0 // obvious, but SMT is flaky
                 })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (fA fB : perm)
  (nthr : nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt nthr)
  : slprop
  =
  kpost1 gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr bid tid **
  (exists* (x : seq et_ab). (fst sh) |-> Frac (1.0R /. nthr) x) **
  (exists* (x : seq et_ab). (fst (snd sh)) |-> Frac (1.0R /. nthr) x) **
  barrier_tok (R.row_major bm bk) (R.row_major bk bn) (fst sh) (fst (snd sh)) (2 * (shared/bk)) nthr tid

ghost
fn block_teardown
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c rows cols)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
      kpost (* comb *) gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpost1 (* comb *) gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr bid tid)

ghost
fn teardown
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c rows cols)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (fA fB : perm)
  ()
  norewrite
  requires
    (forall+ (bid : natlt nblk)
             (tid : natlt nthr).
      kpost1 gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr bid tid) **
    emp
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    // underspec not implemented anyway
    (exists* eC'. gC |-> eC')
