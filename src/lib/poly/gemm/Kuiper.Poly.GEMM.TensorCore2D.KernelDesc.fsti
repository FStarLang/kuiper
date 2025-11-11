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

module B  = Kuiper.Barrier
module MS = Kuiper.Spec.GEMM
module R  = Kuiper.Matrix.Reprs
module SZ = Kuiper.SizeT

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

let bp_sharing
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (m : gpu_matrix et l)
  (em : ematrix et rows cols)
  (nthr : pos)
  : slprop
  = m |-> Frac (1.0R /. nthr) em

let bp_exclusive
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (m : gpu_matrix et l)
  (em : ematrix et rows cols)
  (nthr : pos)
  (tid : natlt nthr)
  : slprop
  = own_strided_chunks m em nthr tid

let barrier_p
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : mlayout bm bk)
  (#l2 : mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  : B.barrier_side nthr =
  fun it tid ->
    let mrow = bid / (cols/bn) in
    let mcol = bid % (cols/bn) in
    (* Barrier contract must be infinite, currently, but we will
       stop after this amount of steps. *)
    if it >= 2 * shared / bk then
      emp
    else if even it then
      (* On even iterations, we give back shared access over the matrix,
         pointing to any value, as we don't care about the content which
         will be overwritten. This is in fact important for the first
         iteration of the loop which starts from uninitialized shared memory. *)
      (exists* em1. bp_sharing m1 em1 nthr) **
      (exists* em2. bp_sharing m2 em2 nthr)
    else
      (* After populating a bit of this matrix, we will give back
         exclusive access to the properly filled strided chunks. *)
      bp_exclusive m1 (ematrix_subtile eA bm bk mrow (it / 2)) nthr tid **
      bp_exclusive m2 (ematrix_subtile eB bk bn (it / 2) mcol) nthr tid

let barrier_q
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : mlayout bm bk)
  (#l2 : mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  : B.barrier_side nthr =
  fun it tid ->
    let mrow = bid / (cols/bn) in
    let mcol = bid % (cols/bn) in
    (* Barrier contract must be infinite, currently, but we will
       stop after this amount of steps. *)
    if it >= 2 * shared / bk then
      emp
    else if even it then
      (* We get back exclusive, strided acess to the matrix. Over unspecified
         contents. *)
      live_strided_chunks m1 nthr tid **
      live_strided_chunks m2 nthr tid
    else
      (* We get back shared, read-only access to the matrix. Over the
         *proper* contents. *)
      bp_sharing m1 (ematrix_subtile eA bm bk mrow (it / 2)) nthr **
      bp_sharing m2 (ematrix_subtile eB bk bn (it / 2) mcol) nthr

let barrier_tok
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (* This is defined over the base shared gpu_arrays, as
  this spec must make sense before the arrays are viewed as
  a matrix. *)
  (l1 : full_mlayout bm bk)
  (l2 : full_mlayout bk bn)
  (sar1 : gpu_array et (bm * bk))
  (sar2 : gpu_array et (bk * bn))
  (it : nat)
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt nthr)
  : slprop
  =
  B.barrier_tok (barrier_p eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid)
                (barrier_q eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid)
                it tid

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
  live_c_shmems sh #(recip nthr) **
  barrier_tok #_ #_ #v #rows #shared #cols eA eB #bm #bk #bn
    (R.row_major bm bk) (R.row_major bk bn) (fst sh) (fst (snd sh)) 0 nthr bid tid

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
    can_create_barrier nthr **
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpre1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid)
  ensures
    consumed_can_create_barrier **
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
  (exists* (x : seq et_ab). (fst sh) |-> Frac (recip nthr) x) **
  (exists* (x : seq et_ab). (fst (snd sh)) |-> Frac (recip nthr) x) **
  barrier_tok #_ #_ #v eA eB (R.row_major bm bk) (R.row_major bk bn) (fst sh) (fst (snd sh)) (2 * (shared/bk)) nthr bid tid

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
