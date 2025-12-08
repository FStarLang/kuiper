module Kuiper.Poly.GEMM.TensorCore2D.KernelDesc

#lang-pulse

#set-options "--z3rlimit 60"

open Kuiper
open Kuiper.Approximates
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.Math { even, odd }
open Kuiper.Matrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Tiling
open Kuiper.Poly.GEMM.Copy.Vec
open Kuiper.Poly.GEMM.Tiled.Common.Vec

module MS = Kuiper.Spec.GEMM
module R  = Kuiper.Matrix.Reprs
module SZ = Kuiper.SizeT
module FlipFlopBarrier = Kuiper.Poly.GEMM.FlipFlopBarrier
module B = Kuiper.Barrier

// Using 1.0R /. x can lead to many odd SMT failures...
// work around it. We should investigate why and fix it.
[@@pulse_unfold]
let recip (x : pos) : y:Real.real{y >. 0.0R} = 1.0R /. x

type constraints (bm bn bk tm tn tk wm wn : pos) : prop =
  tm /?+ bm /\
  tn /?+ bn /\
  tk /?+ bk /\
  wm * tm /?+ bm /\
  wn * tn /?+ bn /\
  SZ.fits (wm * wn)

let warp_tile_pts_to
  (#et : Type0) {| scalar et |}
  (#rows : nat)
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
  (em : ematrix et (wm * tm) (wn * tn))
  : slprop
  =
  gpu_matrix_pts_to
    (warp_tile (block_tile gC bm bn bid) (wm*tm) (wn*tn) wid)
    #(recip warp_size)
    em

let warp_tile_pts_to_full
  (#et : Type0) {| scalar et |}
  (#rows : nat)
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
  (em : ematrix et (wm * tm) (wn * tn))
  : slprop
  =
  gpu_matrix_pts_to
    (warp_tile (block_tile gC bm bn bid) (wm*tm) (wn*tn) wid)
    em

let warp_tile_approximates
  (#et : Type0) {| scalar et, real_like et |}
  (#rows : nat)
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
  (#_ : squash (bk /?+ shared))
  (#_ : squash (bn /?+ cols))
  (fA fB : perm)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
  (nthr : nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt nthr)
  : slprop
  =
  gA |-> Frac (fA /. (rows/bm * (cols/bn) * nthr)) eA **
  gB |-> Frac (fB /. (rows/bm * (cols/bn) * nthr)) eB **
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
  (#_ : squash (bk /?+ shared))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (fA fB : perm)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
  (nthr : nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt nthr)
  : slprop
  =
  kpre1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid **
  live_c_shmems sh #(recip nthr)

// #push-options "--z3rlimit_factor 4 --split_queries no --fuel 0 --ifuel 0 --query_stats"
// instance kpre_block_sendable
//   (#et_ab #et_c : Type0)
//   (_: scalar et_ab)
//   (v : has_vec_cpy et_ab)
//   (_: scalar et_c)
//   (_:real_like et_ab)
//   (_:real_like et_c)
//   (#rows #shared #cols : szp)
//   (#lA : mlayout rows shared)
//   (#lB : mlayout shared cols)
//   (#lC : mlayout rows cols)
//   (gA : gpu_matrix et_ab lA { is_global_matrix gA })
//   (eA : ematrix et_ab rows shared)
//   (gB : gpu_matrix et_ab lB { is_global_matrix gB })
//   (eB : ematrix et_ab shared cols)
//   (gC : gpu_matrix et_c lC { is_global_matrix gC })
//   (eC : ematrix et_c rows cols)
//   (bm bn bk
//    tm tn tk
//    wm wn : szp { constraints bm bn bk tm tn tk wm wn })
//   (#_ : squash (bm /?+ rows))
//   (#_ : squash (bk /?+ shared))
//   (#_ : squash (bn /?+ cols))
//   (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
//   (fA fB : perm)
//   (rA : ematrix real rows shared)
//   (rB : ematrix real shared cols)
//   (rC : ematrix real rows cols)
//   (nblk: SZ.t { SZ.v nblk == (rows/bm * (cols/bn)) })
//   (nthr: SZ.t { SZ.v nthr == (bm/(wm*tm)*(bn/(wn*tn))*warp_size) })
//   (sh : c_shmems (shmems_desc et_ab bm bn bk))
//   (pf : c_shmems_inv sh)
//   (i : natlt nblk)
//   (j : natlt nthr)
// : is_send_across block_of
//   (kpre gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC (SZ.v nthr) sh i j)
// = solve //checking to see that it is provable
// #pop-options

ghost
fn setup
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
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
  (#_ : squash (bk /?+ shared))
  (#_ : squash (bn /? cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (fA fB : perm)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
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
  (#_ : squash (bk /?+ shared))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size /\ nthr <= 1024})
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (fA fB : perm)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
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
  (#rows #cols : erased nat)
  (em : ematrix et rows cols)
  (trows : erased nat{trows > 0 /\ trows /? rows})
  (tcols : erased nat{tcols > 0 /\ tcols /? cols})
  (bid : enatlt (rows/trows * (cols/tcols)))
  : ematrix et trows tcols
  = ematrix_subtile em trows tcols
      (block_tile_idx_rows rows cols trows tcols bid)
      (block_tile_idx_cols rows cols trows tcols bid)

let warp_tile_ematrix
  (#et : Type0) {| scalar et |}
  (#rows #cols : erased nat)
  (em : ematrix et rows cols)
  (trows : erased nat{trows > 0 /\ trows /? rows})
  (tcols : erased nat{tcols > 0 /\ tcols /? cols})
  (wid : enatlt (rows/trows * (cols/tcols)))
  : ematrix et trows tcols
  = ematrix_subtile em trows tcols
      (warp_tile_idx_rows rows cols trows tcols wid)
      (warp_tile_idx_cols rows cols trows tcols wid)

let warp_tile_i
  (#rows #cols : pos)
  (bm bn bk
   tm tn tk
   wm wn : pos { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (nthr : nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (bid : natlt (rows/bm * (cols/bn)))
  (wid : natlt (nthr / warp_size)) // warp ID
  : GTot (natlt (rows / (wm*tm)))
  =
    let tile_i = bid / (cols/bn) in
    let tile_j = bid % (cols/bn) in
    assert (wid < (bm/(wm*tm)) * (bn/(wn*tn)));
    let subtile_i = wid / (bn/(wn*tn)) in
    let subtile_j = wid % (bn/(wn*tn)) in
    (* Z3 takes some convincing.... *)
    assert (subtile_i < (bm/(wm*tm)));
    assert (tile_i < rows/bm);
    assert (tile_i * (bm / (wm*tm)) < rows/(wm*tm));
    tile_i * (bm / (wm*tm)) + subtile_i

let warp_tile_j
  (#rows #cols : pos)
  (bm bn bk
   tm tn tk
   wm wn : pos { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (nthr : nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (bid : natlt (rows/bm * (cols/bn)))
  (wid : natlt (nthr / warp_size)) // warp ID
  : GTot (natlt (cols / (wn*tn)))
  =
    let tile_i = bid / (cols/bn) in
    let tile_j = bid % (cols/bn) in
    let subtile_i = wid / (bn/(wn*tn)) in
    let subtile_j = wid % (bn/(wn*tn)) in
    tile_j * (bn / (wn*tn)) + subtile_j

unfold
let kpost1
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
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
  (#_ : squash (bk /?+ shared))
  (#_ : squash (bn /?+ cols))
  (fA fB : perm)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
  (#_ : squash (wm * tm /?+ rows)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ cols)) // idem
  (nthr : nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt nthr)
  : slprop
  =
  gA |-> Frac (fA /. (rows/bm * (cols/bn) * nthr)) eA **
  gB |-> Frac (fB /. (rows/bm * (cols/bn) * nthr)) eB **
  warp_tile_approximates gC bm bn tm tn wm wn bid (tid / warp_size)
    (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
               (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j bm bn bk tm tn tk wm wn nthr bid (tid / warp_size))))

unfold
let kpost
  (#et_ab #et_c : Type0)
  {| scalar et_ab, v : has_vec_cpy et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
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
   wm wn : szp { constraints bm bn bk tm tn tk wm wn
                 /\ 2 * (shared / bk) >= 0 // obvious, but SMT is flaky
                 /\ bm * bk > 0 // idem
                  /\ bk * bn > 0 // idem
                 })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bk /?+ shared))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (fA fB : perm)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
  (#_ : squash (wm * tm /?+ rows)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ cols)) // idem
  (nthr : nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt nthr)
  : slprop
  =
  kpost1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid **
  live_c_shmems sh #(recip nthr)

// #push-options "--z3rlimit_factor 4 --split_queries no --fuel 0 --ifuel 0 --query_stats"
// #restart-solver
// instance kpost_block_sendable
//   (#et_ab #et_c : Type0)
//   (_: scalar et_ab)
//   (v : has_vec_cpy et_ab)
//   (_: scalar et_c)
//   (_:real_like et_ab)
//   (_:real_like et_c)
//   (#rows #shared #cols : szp)
//   (#lA : mlayout rows shared)
//   (#lB : mlayout shared cols)
//   (#lC : mlayout rows cols)
//   (gA : gpu_matrix et_ab lA { is_global_matrix gA })
//   (eA : ematrix et_ab rows shared)
//   (gB : gpu_matrix et_ab lB { is_global_matrix gB })
//   (eB : ematrix et_ab shared cols)
//   (gC : gpu_matrix et_c lC { is_global_matrix gC })
//   (eC : ematrix et_c rows cols)
//   (bm bn bk
//    tm tn tk
//    wm wn : szp { constraints bm bn bk tm tn tk wm wn })
//   (#_ : squash (bm /?+ rows))
//   (#_ : squash (bk /?+ shared))
//   (#_ : squash (bn /?+ cols))
//   (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
//   (fA fB : perm)
//   (rA : ematrix real rows shared)
//   (rB : ematrix real shared cols)
//   (rC : ematrix real rows cols)
//   (nblk: SZ.t { SZ.v nblk == (rows/bm * (cols/bn)) })
//   (nthr: SZ.t { SZ.v nthr == (bm/(wm*tm)*(bn/(wn*tn))*warp_size) })
//   (sh : c_shmems (shmems_desc et_ab bm bn bk))
//   (pf : c_shmems_inv sh)
//   (i : natlt nblk)
//   (j : natlt nthr)
// : is_send_across block_of
//   (kpost gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC (SZ.v nthr) sh i j)
// = solve //this takes forever! not sure  it helps to prove it here, rather than letting it be proven in place at the kernel desc
// #pop-options

ghost
fn block_teardown
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
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
  (#_ : squash (bk /?+ shared))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (fA fB : perm)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
  (#_ : squash (wm * tm /?+ rows)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ cols)) // idem
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
  (#_ : squash (bk /?+ shared))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (fA fB : perm)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
  (#_ : squash (wm * tm /?+ rows)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ cols)) // idem
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
    (exists* (eC' : ematrix et_c rows cols).
      gC |-> eC' ** pure (eC' %~ MS.matmul rA rB))
