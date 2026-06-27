module Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc

#lang-pulse

#set-options "--z3rlimit 60"

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.Math { even, odd }
open Kuiper.Matrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Tiling
open Kuiper.Kernel.GEMM.Copy.Vec
open Kuiper.Kernel.GEMM.Tiled.Common.Vec

module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT

// Using 1.0R /. x can lead to many odd SMT failures...
// work around it. We should investigate why and fix it.
[@@pulse_unfold]
let precip (x : pos) : y:Real.real{y >. 0.0R} = 1.0R /. x

type constraints (bm bn bk tm tn tk wm wn : pos) : prop =
  tm /?+ bm /\
  tn /?+ bn /\
  tk /?+ bk /\
  wm * tm /?+ bm /\
  wn * tn /?+ bn /\
  SZ.fits (wm * wn)

let warp_tile_pts_to
  (#et : Type0) {| scalar et |}
  (#m : nat)
  (#n : nat)
  (#lC : mlayout m n)
  (gC : gpu_matrix et lC)
  (bm : pos{bm /?+ m})
  (bn : pos{bn /?+ n})
  (tm : pos{tm /?+ bm})
  (tn : pos{tn /?+ bn})
  (wm : pos{wm * tm /?+ bm})
  (wn : pos{wn * tn /?+ bn})
  (bid : natlt ((m/bm) * (n/bn)))
  (wid : natlt (bm/(wm*tm) * (bn/(wn*tn))))
  (em : ematrix et (wm * tm) (wn * tn))
  : slprop
  =
  gpu_matrix_pts_to
    (warp_tile (block_tile gC bm bn bid) (wm*tm) (wn*tn) wid)
    #(precip warp_size)
    em

let warp_tile_pts_to_full
  (#et : Type0) {| scalar et |}
  (#m : nat)
  (#n : nat)
  (#lC : mlayout m n)
  (gC : gpu_matrix et lC)
  (bm : pos{bm /?+ m})
  (bn : pos{bn /?+ n})
  (tm : pos{tm /?+ bm})
  (tn : pos{tn /?+ bn})
  (wm : pos{wm * tm /?+ bm})
  (wn : pos{wn * tn /?+ bn})
  (bid : natlt ((m/bm) * (n/bn)))
  (wid : natlt (bm/(wm*tm) * (bn/(wn*tn))))
  (em : ematrix et (wm * tm) (wn * tn))
  : slprop
  =
  gpu_matrix_pts_to
    (warp_tile (block_tile gC bm bn bid) (wm*tm) (wn*tn) wid)
    em

let warp_tile_approximates
  (#et : Type0) {| scalar et, real_like et |}
  (#m : nat)
  (#n : nat)
  (#lC : mlayout m n)
  (gC : gpu_matrix et lC)
  (bm : pos{bm /?+ m})
  (bn : pos{bn /?+ n})
  (tm : pos{tm /?+ bm})
  (tn : pos{tn /?+ bn})
  (wm : pos{wm * tm /?+ bm})
  (wn : pos{wn * tn /?+ bn})
  (bid : natlt ((m/bm) * (n/bn)))
  (wid : natlt (bm/(wm*tm) * (bn/(wn*tn))))
  (rm : ematrix real (wm * tm) (wn * tn))
  : slprop
  =
  exists* em.
    warp_tile_pts_to gC bm bn tm tn wm wn bid wid em **
    pure (em %~ rm)

unfold
let kpre1
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#m #n #k : szp)
  (#lA : mlayout m k)
  (#lB : mlayout k n)
  (#lC : mlayout m n)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab m k)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab k n)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c m n)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bk /?+ k))
  (#_ : squash (bn /?+ n))
  (fA fB : perm)
  (rA : ematrix real m k)
  (rB : ematrix real k n)
  (rC : ematrix real m n)
  (nthr : nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (bid : natlt (m/bm * (n/bn)))
  (tid : natlt nthr)
  : slprop
  =
  gA |-> Frac (fA /. (m/bm * (n/bn) * nthr)) eA **
  gB |-> Frac (fB /. (m/bm * (n/bn) * nthr)) eB **
  (exists* tC.
    warp_tile_pts_to gC bm bn tm tn wm wn bid (tid/warp_size) tC) **
  // ^ Missing functional spec, but not a problem until
  // we make this an actual GEMM instead of a matmul.
  pure (aligned 16 (core gA)) **
  pure (aligned 16 (core gB)) **
  pure (eA %~ rA) **
  pure (eB %~ rB) **
  pure (eC %~ rC)

unfold
let kpre
  (#et_ab #et_c : Type0)
  {| scalar et_ab, v : has_vec_cpy et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#m #n #k : szp)
  (#lA : mlayout m k)
  (#lB : mlayout k n)
  (#lC : mlayout m n)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab m k)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab k n)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c m n)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bk /?+ k))
  (#_ : squash (bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (fA fB : perm)
  (rA : ematrix real m k)
  (rB : ematrix real k n)
  (rC : ematrix real m n)
  (nthr : nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt (m/bm * (n/bn)))
  (tid : natlt nthr)
  : slprop
  =
  kpre1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid **
  live_c_shmems sh #(precip nthr)

ghost
fn setup
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#m #n #k : szp)
  (#lA : mlayout m k)
  (#lB : mlayout k n)
  (#lC : mlayout m n)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab m k)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab k n)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c m n)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /? m))
  (#_ : squash (bk /?+ k))
  (#_ : squash (bn /? n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (nblk : szp{SZ.v nblk == m/bm * (n/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (fA fB : perm)
  (rA : ematrix real m k)
  (rB : ematrix real k n)
  (rC : ematrix real m n)
  ()
  norewrite
  requires
    gA |-> Frac fA eA ** pure (eA %~ rA) **
    gB |-> Frac fB eB ** pure (eB %~ rB) **
    gC |-> eC ** pure (eC %~ rC)
  ensures
    (forall+ (bid : natlt nblk)
             (tid : natlt nthr).
      kpre1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid) **
    pure (SZ.fits (mlayout_size lC)) // frame

ghost
fn block_setup
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#m #n #k : szp)
  (#lA : mlayout m k)
  (#lB : mlayout k n)
  (#lC : mlayout m n)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab m k)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab k n)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c m n)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bk /?+ k))
  (#_ : squash (bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : szp{SZ.v nblk == m/bm * (n/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size /\ nthr <= 1024})
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (fA fB : perm)
  (rA : ematrix real m k)
  (rB : ematrix real k n)
  (rC : ematrix real m n)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpre1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid)
  ensures
    (forall+ (tid : natlt nthr).
      kpre gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr sh bid tid) **
    emp (* frame *)


let block_tile_ematrix
  (#et : Type0) {| scalar et |}
  (#m #n : erased nat)
  (em : ematrix et m n)
  (trows : erased nat{trows > 0 /\ trows /? m})
  (tcols : erased nat{tcols > 0 /\ tcols /? n})
  (bid : enatlt (m/trows * (n/tcols)))
  : ematrix et trows tcols
  = ematrix_subtile em trows tcols
      (block_tile_idx_rows m n trows tcols bid)
      (block_tile_idx_cols m n trows tcols bid)

let warp_tile_ematrix
  (#et : Type0) {| scalar et |}
  (#m #n : erased nat)
  (em : ematrix et m n)
  (trows : erased nat{trows > 0 /\ trows /? m})
  (tcols : erased nat{tcols > 0 /\ tcols /? n})
  (wid : enatlt (m/trows * (n/tcols)))
  : ematrix et trows tcols
  = ematrix_subtile em trows tcols
      (warp_tile_idx_rows m n trows tcols wid)
      (warp_tile_idx_cols m n trows tcols wid)

let warp_tile_i
  (#m #n : pos)
  (bm bn bk
   tm tn tk
   wm wn : pos { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (nthr : nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (bid : natlt (m/bm * (n/bn)))
  (wid : natlt (nthr / warp_size)) // warp ID
  : GTot (natlt (m / (wm*tm)))
  =
    let tile_i = bid / (n/bn) in
    let tile_j = bid % (n/bn) in
    assert (wid < (bm/(wm*tm)) * (bn/(wn*tn)));
    let subtile_i = wid / (bn/(wn*tn)) in
    let subtile_j = wid % (bn/(wn*tn)) in
    (* Z3 takes some convincing.... *)
    assert (subtile_i < (bm/(wm*tm)));
    assert (tile_i < m/bm);
    assert (tile_i * (bm / (wm*tm)) < m/(wm*tm));
    tile_i * (bm / (wm*tm)) + subtile_i

let warp_tile_j
  (#m #n : pos)
  (bm bn bk
   tm tn tk
   wm wn : pos { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (nthr : nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (bid : natlt (m/bm * (n/bn)))
  (wid : natlt (nthr / warp_size)) // warp ID
  : GTot (natlt (n / (wn*tn)))
  =
    let tile_i = bid / (n/bn) in
    let tile_j = bid % (n/bn) in
    let subtile_i = wid / (bn/(wn*tn)) in
    let subtile_j = wid % (bn/(wn*tn)) in
    tile_j * (bn / (wn*tn)) + subtile_j

unfold
let kpost1
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#m #n #k : szp)
  (#lA : mlayout m k)
  (#lB : mlayout k n)
  (#lC : mlayout m n)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab m k)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab k n)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c m n)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bk /?+ k))
  (#_ : squash (bn /?+ n))
  (fA fB : perm)
  (rA : ematrix real m k)
  (rB : ematrix real k n)
  (rC : ematrix real m n)
  (#_ : squash (wm * tm /?+ m)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ n)) // idem
  (nthr : nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (bid : natlt (m/bm * (n/bn)))
  (tid : natlt nthr)
  : slprop
  =
  gA |-> Frac (fA /. (m/bm * (n/bn) * nthr)) eA **
  gB |-> Frac (fB /. (m/bm * (n/bn) * nthr)) eB **
  warp_tile_approximates gC bm bn tm tn wm wn bid (tid / warp_size)
    (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
               (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j bm bn bk tm tn tk wm wn nthr bid (tid / warp_size))))

unfold
let kpost
  (#et_ab #et_c : Type0)
  {| scalar et_ab, v : has_vec_cpy et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#m #n #k : szp)
  (#lA : mlayout m k)
  (#lB : mlayout k n)
  (#lC : mlayout m n)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab m k)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab k n)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c m n)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn
                 /\ 2 * (k / bk) >= 0 // obvious, but SMT is flaky
                 /\ bm * bk > 0 // idem
                  /\ bk * bn > 0 // idem
                 })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bk /?+ k))
  (#_ : squash (bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (fA fB : perm)
  (rA : ematrix real m k)
  (rB : ematrix real k n)
  (rC : ematrix real m n)
  (#_ : squash (wm * tm /?+ m)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ n)) // idem
  (nthr : nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt (m/bm * (n/bn)))
  (tid : natlt nthr)
  : slprop
  =
  kpost1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid **
  live_c_shmems sh #(precip nthr)

ghost
fn block_teardown
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#m #n #k : szp)
  (#lA : mlayout m k)
  (#lB : mlayout k n)
  (#lC : mlayout m n)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab m k)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab k n)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c m n)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bk /?+ k))
  (#_ : squash (bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : szp{SZ.v nblk == m/bm * (n/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (fA fB : perm)
  (rA : ematrix real m k)
  (rB : ematrix real k n)
  (rC : ematrix real m n)
  (#_ : squash (wm * tm /?+ m)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ n)) // idem
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
      kpost gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr sh bid tid) **
    emp
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpost1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid)

ghost
fn teardown
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#m #n #k : szp)
  (#lA : mlayout m k)
  (#lB : mlayout k n)
  (#lC : mlayout m n)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab m k)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab k n)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c m n)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bk /?+ k))
  (#_ : squash (bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : szp{SZ.v nblk == m/bm * (n/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (fA fB : perm)
  (rA : ematrix real m k)
  (rB : ematrix real k n)
  (rC : ematrix real m n)
  (#_ : squash (wm * tm /?+ m)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ n)) // idem
  ()
  norewrite
  requires
    (forall+ (bid : natlt nblk)
             (tid : natlt nthr).
      kpost1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid) **
    pure (SZ.fits (mlayout_size lC)) // frame
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    (exists* (eC' : ematrix et_c m n).
      gC |-> eC' ** pure (eC' %~ MS.matmul rA rB))
