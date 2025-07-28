module Kuiper.Poly.GEMM.BlockTiling2D

#lang-pulse

open Kuiper

module EM = Kuiper.EMatrix

open Kuiper.EMatrix4
open Kuiper.Matrix.Reprs.Type
open Kuiper.Math { even, odd, even_2x, odd_2x1 }

module M = Kuiper.Matrix
open Kuiper.Matrix {
  gpu_matrix,
  gpu_matrix_pts_to,
  gpu_matrix_pts_to_cell,
}

open Kuiper.Matrix4 {
  gpu_matrix as gpu_matrix4,
  gpu_matrix_pts_to as m4_pts_to,
  gpu_matrix_pts_to_cell as m4_pts_to_cell,
  mlayout4,
  clayout4
}

module M4 = Kuiper.Matrix4
module MS = Kuiper.Spec.GEMM
module SZ = FStar.SizeT
module B = Kuiper.Barrier

module R = Kuiper.Matrix.Reprs

open Kuiper.EMatrix { ematrix }
open Kuiper.ArrayView {
  varray,
  varray_pts_to,
  varray_pts_to_cell
}

(* Description of shared memory used in this kernel. *)
inline_for_extraction noextract
let shmems_desc
  (et:Type0) {| sized et |}
  (bm bn bk: szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  : list shmem_desc = [
  SHArray et (bm *^ bk);
  SHArray et (bk *^ bn);
]

// let lemma0 (a c : int) (b : nat): Lemma (a % c * b == a * b % c)

(* without shmem ownership *)
#push-options "--retry 3"
unfold
let kpre1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (#lA : mlayout4 mrows   mshared bm bk)
  (#lB : mlayout4 mshared mcols   bk bn)
  (#lC : mlayout4 mrows   mcols   bm bn)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (eA : ematrix4 et mrows mshared bm bk)
  (eB : ematrix4 et mshared mcols bk bn)
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (bm/tm * bn/tn))
  : slprop
  =
  (gA |-> Frac (fA /. (mrows * mcols * (bm/tm * bn/tn))) eA) **
  (gB |-> Frac (fB /. (mrows * mcols * (bm/tm * bn/tn))) eB) **
  (forall+ (i : natlt tm) (j : natlt tn).
    (exists* v.
      m4_pts_to_cell gC
        (bid / mcols) (bid % mcols)
        ((tid / (bn/tn) * tm) + i)
        ((tid % (bn/tn) * tn) + j)
        v))

(* NO FUNCTIONAL SPEC RIGHT NOW *)
unfold
let kpost1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (#lA : mlayout4 mrows   mshared bm bk)
  (#lB : mlayout4 mshared mcols   bk bn)
  (#lC : mlayout4 mrows   mcols   bm bn)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (eA : ematrix4 et mrows mshared bm bk)
  (eB : ematrix4 et mshared mcols bk bn)
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (bm/tm * bn/tn))
  : slprop
  =
  (gA |-> Frac (fA /. (mrows * mcols * (bm/tm * bn/tn))) eA) **
  (gB |-> Frac (fB /. (mrows * mcols * (bm/tm * bn/tn))) eB) **
  (forall+ (i : natlt tm) (j : natlt tn).
    (exists* v.
      m4_pts_to_cell gC
        (bid / mcols) (bid % mcols)
        ((tid / (bn/tn) * tm) + i)
        ((tid % (bn/tn) * tn) + j)
        v))
#pop-options

(* The barrier flip-flops between an initial state
where every threads shares all of the two arrays, and
a second state where every thread owns a single cell
in each array, related to their tid.

So in even steps, they give their shared ownership,
and receive their cells. (p)
*)
let div_ceil (a : nat) (b : pos) = (a + (b-1))/b
let own_tile_stride_cells
  (#et : Type0)
  (#rows #cols : nat)
  (#lm : mlayout rows cols)
  (m : gpu_matrix et lm)
  (em : ematrix et rows cols)
  // cannot infer if implicit
  (nthr : nat)
  (tid : natlt nthr)
  : slprop
  =
  forall+ (iter : natlt (div_ceil (rows*cols) nthr)).
    let flat_idx = iter * nthr + tid <: nat in
    let i = flat_idx/cols <: nat in
    let j = flat_idx%cols <: nat in
      // if (flat_idx < rows*cols)
      if (i < rows && j < cols)
      then gpu_matrix_pts_to_cell m i j (EM.macc em i j)
      else emp

inline_for_extraction noextract
fn cp_tile
  (#et : Type0) {| scalar et |}
  (#brows #bcols #mrows #mcols: szp)
  (#slM : mlayout brows bcols) {| clayout slM |}
  // this should just be a submatrix dst
  (sM : gpu_matrix et slM)
  (#esM : ematrix et brows bcols)
  (#lA : mlayout4 mrows mcols brows bcols)
  {| clayout4 lA |}
  // this should just be a submatrix src
  (gM : gpu_matrix4 et lA)
  (#fM : perm)
  (#eM : ematrix4 et mrows mcols brows bcols)
  (mrow : szlt mrows)
  (mcol : szlt mcols)
  (#nthr : sz)
  (tid : szlt nthr)
  preserves
    gpu **
    (gM |-> Frac fM eM)
  requires
    own_tile_stride_cells sM esM nthr tid
{
  unfold own_tile_stride_cells sM esM nthr tid;
  let mlen = mrows *^ brows *^ mcols *^ bcols;
  admit();
  let mut i = tid;
  while (SZ.(!i <^ mlen))
    invariant
      exists* (ite: sz) (vflatIdx: sz{vflatIdx < mlen}).
        (pure(ite * nthr + tid == vflatIdx)) **
        (i |-> vflatIdx) //**
        // (gpu_matrix_pts_to_cell sM
        //   ((ite*nthr+tid)/bcols) ((ite*nthr+tid%bcols))
        //   (macc eM ((ite*nthr+tid)/bcols) ((ite*nthr+tid%bcols))))// **
        // (forall+ (ii: natlt ite).
        //   gpu_matrix_pts_to_cell gM
        //     ((ii*nthr+tid)/bcols) ((ii*nthr+tid)%bcols)
        //     (macc eM ((ii*nthr+tid)/bcols) ((ii*nthr+tid)%bcols)))
  {
    let v = M4.gpu_matrix_read gM mrow mcol (!i /^ bcols) (!i %^ bcols);
    M.gpu_matrix_write_cell sM (!i /^ bcols) (!i %^ bcols) v;

    i := !i + nthr;
  };

  // fold own_1_cell sA (tid/bk) (tid%bk);
}

let barrier_p
  (#et : Type0)
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
      (exists* (em: ematrix _ _ _). own_tile_stride_cells m1 em nthr tid) **
      (exists* (em: ematrix _ _ _). own_tile_stride_cells m2 em nthr tid)

let barrier_q
  (#et : Type0)
  (#bm #bn #bk : szp)
  (#l1 : mlayout bm bk)
  (#l2 : mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  (nthr : pos)
  : B.barrier_side nthr =
  fun it tid -> barrier_p m1 m2 nthr (it+1) tid (* flip flop *)

let barrier_tok
  (#et : Type0)
  (#bm #bn #bk : szp)
  (* This is defined over the base shared gpu_arrays, as
  this spec must make sense before the arrays are viewed as
  a matrix. *)
  (l1 : mlayout bm bk)
  (l2 : mlayout bk bn)
  (sar1 : gpu_array et (bm * bk))
  (sar2 : gpu_array et (bk * bn))
  (it : nat)
  (nthr : pos)
  (tid : natlt nthr)
  : slprop
  =
  B.barrier_tok (barrier_p (M.from_array l1 sar1) (M.from_array l2 sar2) nthr)
                (barrier_q (M.from_array l1 sar1) (M.from_array l2 sar2) nthr)
                it tid

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (slA : mlayout bm bk)
  (slB : mlayout bk bn)
  (#lA : mlayout4 mrows   mshared bm bk)
  (#lB : mlayout4 mshared mcols   bk bn)
  (#lC : mlayout4 mrows   mcols   bm bn)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (eA : ematrix4 et mrows mshared bm bk)
  (eB : ematrix4 et mshared mcols bk bn)
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt (mrows * mcols))
  (tid : natlt (bm/tm * bn/tn))
  : slprop
  =
  kpre1 comb tm tn gA gB gC eA eB fA fB bid tid **
  (exists* (x : seq _). fst sh |-> Frac (1.0R /. (bm/tm * bn/tn)) x) **
  (exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. (bm/tm * bn/tn)) x) **
  barrier_tok slA slB (fst sh) (fst (snd sh)) 0 (bm/tm * bn/tn) tid

unfold
let kpost
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (slA : mlayout bm bk)
  (slB : mlayout bk bn)
  (#lA : mlayout4 mrows   mshared bm bk)
  (#lB : mlayout4 mshared mcols   bk bn)
  (#lC : mlayout4 mrows   mcols   bm bn)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (eA : ematrix4 et mrows mshared bm bk)
  (eB : ematrix4 et mshared mcols bk bn)
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt (mrows * mcols))
  (tid : natlt (bm/tm * bn/tn))
  : slprop
  =
  kpre comb tm tn slA slB gA gB gC eA eB fA fB sh bid tid
  // kpost1 comb tm tn gA gB gC eA eB fA fB bid tid **
  // (exists* (x : seq _). fst sh |-> Frac (1.0R /. (bm/tm * bn/tn)) x) **
  // (exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. (bm/tm * bn/tn)) x) **
  // barrier_tok slA slB (fst sh) (fst (snd sh)) 0 (bm/tm * bn/tn) tid

inline_for_extraction noextract
fn populate_shmem
  (#et : Type0) {| scalar et |}
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : erased nat)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (#slA : mlayout bm bk) {| clayout slA |}
  (#slB : mlayout bk bn) {| clayout slB |}
  (sA : gpu_matrix et slA)
  (sB : gpu_matrix et slB)
  (#lA : mlayout4 mrows   mshared bm bk)
  (#lB : mlayout4 mshared mcols   bk bn)
  {| clayout4 lA, clayout4 lB |}
  // should be gpu_matrix and just pointers to the tiles
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (#fA #fB : perm)
  (#eA : ematrix4 et mrows   mshared bm bk)
  (#eB : ematrix4 et mshared mcols   bk bn)
  (mrow : szlt mrows)
  (mk : szlt mshared)
  (mcol : szlt mcols)
  (tid : szlt (bm/tm * bn/tn))
  preserves
    gpu **
    (gA |-> Frac fA eA) **
    (exists* (eA: _) (eB: _).
      own_tile_stride_cells sA eA (bm/tm * bn/tn) **
      own_tile_stride_cells sB eB (bm/tm * bn/tn))
{
  admit();
  // unfold own_1_cell sA (tid/bk) (tid%bk);
  // let va = M4.gpu_matrix_read gA mrow mk (tid /^ bk) (tid %^ bk);
  // M.gpu_matrix_write_cell sA (tid /^ bk) (tid %^ bk) va;
  // fold own_1_cell sA (tid/bk) (tid%bk);

  // unfold own_1_cell sB (tid/bn) (tid%bn);
  // let vb = M4.gpu_matrix_read gB mk mcol (tid /^ bn) (tid %^ bn);
  // M.gpu_matrix_write_cell sB (tid /^ bn) (tid %^ bn) vb;
  // fold own_1_cell sB (tid/bn) (tid%bn);
}

inline_for_extraction noextract
fn subproducts2d
  (#et : Type0) {| scalar et |}
  (#bm #bn #bk: szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (rAcol rBrow rchProd: array et)
  (#vrAcol #vrBrow #vrchProd : erased (seq et))
  (#l1 : mlayout bm bk) {| clayout l1 |}
  (#l2 : mlayout bk bn) {| clayout l2 |}
  (gA : gpu_matrix et l1)
  (gB : gpu_matrix et l2)
  (#eA : ematrix et bm bk)
  (#eB : ematrix et bk bn)
  (#f : perm)
  (arow: szlt (bm/tm))
  (bcol : szlt (bn/tn))
  preserves
    gpu **
    (gA |-> Frac f eA) **
    (gB |-> Frac f eB)
  requires
    pure (Seq.length vrAcol == tm /\
          Seq.length vrBrow == tn /\
          Seq.length vrchProd == tm * tn /\
          SZ.fits (tm * tn)) **
    (rAcol |-> vrAcol) **
    (rBrow |-> vrBrow) **
    (rchProd |-> vrchProd)
  ensures
    exists* vrAcol' vrBrow' vrchProd'.
      pure (Seq.length vrAcol' == tm /\
            Seq.length vrBrow' == tn /\
            Seq.length vrchProd' == tm * tn) **
      (rAcol |-> vrAcol') **
      (rBrow |-> vrBrow') **
      (rchProd |-> vrchProd')
{
  let mut dotIdx : sz = 0sz;
  while (SZ.(!dotIdx <^ bk))
    invariant
      exists* (vdotIdx : sz{vdotIdx <= bk}) (vrAcol0 : erased (lseq et tm))
        (vrBrow0 : erased (lseq et tn)) (vrchProd0 : erased (lseq et (tm*tn))).
        (dotIdx |-> vdotIdx) **
        (rAcol |-> vrAcol0) **
        (rBrow |-> vrBrow0) **
        (rchProd |-> vrchProd0)
  {
    open Pulse.Lib.Array;

    // for-loop
    // {
    let mut i0 = 0sz;
    while (SZ.(!i0 <^ tm))
      invariant
        exists* (vi : sz{vi <= tm}) (vrAcol : erased (lseq et tm)).
          (i0 |-> vi) **
          (rAcol |-> vrAcol)
    {
      let va = M.gpu_matrix_read gA (arow *^ tm +^ !i0) !dotIdx;
      rAcol.(!i0) <- va;

      i0 := !i0 +^ 1sz;
    };
    // };

    // for-loop
    // {
    let mut i1 = 0sz;
    while (SZ.(!i1 <^ tn))
      invariant
        exists* (vi : sz{vi <= tn}) (vrBrow : erased (lseq et tn)).
          (i1 |-> vi) **
          (rBrow |-> vrBrow)
    {
      let vb = M.gpu_matrix_read gB !dotIdx (bcol *^ tn +^ !i1);
      rBrow.(!i1) <- vb;

      i1 := !i1 +^ 1sz;
    };
    // };

    let mut resIdxM = 0sz;
    while (SZ.(!resIdxM <^ tm))
      invariant
        exists* (vresIdxM : sz{vresIdxM <= tm}) (vrchProd : erased (lseq et (tm*tn))).
          (resIdxM |-> vresIdxM) **
          (rchProd |-> vrchProd)
    {
      let mut resIdxN = 0sz;
      while (SZ.(!resIdxN <^ tn))
        invariant
          exists* (vresIdxN : sz{vresIdxN <= tn}) (vrchProd : erased (lseq et (tm*tn))).
            (resIdxN |-> vresIdxN) **
            (rchProd |-> vrchProd)
      {
        let ra = rAcol.(!resIdxM);
        let rb = rBrow.(!resIdxN);
        let iM = !resIdxM;
        let iN = !resIdxN;
        assert(pure(SZ.fits(iM *^ tn +^ iN)));
        let old = rchProd.(!resIdxM *^ tn +^ !resIdxN);
        let mad = old `add` (ra `mul` rb);
        rchProd.(!resIdxM *^ tn +^ !resIdxN) <- mad;

        resIdxN := !resIdxN +^ 1sz;
      };

      resIdxM := !resIdxM +^ 1sz;
    };

    dotIdx := !dotIdx +^ 1sz;
  }
}


// even 20 isn't evenough for the checking from the terminal
//  (but enough for the vs code extension)
#push-options "--z3rlimit 30"
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : mlayout bm bk)
  (slB : mlayout bk bn)
  {| clayout slA, clayout slB |}
  (#lA : mlayout4 mrows   mshared bm bk)
  (#lB : mlayout4 mshared mcols   bk bn)
  (#lC : mlayout4 mrows   mcols   bm bn)
  {| clayout4 lA, clayout4 lB, clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (#fA : perm)
  (gB : gpu_matrix4 et lB)
  (#fB : perm)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared bm bk)
  (#eB : ematrix4 et mshared mcols   bk bn)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : szlt (mrows * mcols))
  (tid : szlt (bm/tm * bn))
  ()
  requires
    gpu **
    kpre comb tm slA slB gA gB gC eA eB fA fB sh bid tid **
    thread_id (bm/tm * bn) tid **
    block_id (mrows * mcols) bid
  ensures
    gpu **
    kpost comb tm slA slB gA gB gC eA eB fA fB sh bid tid **
    thread_id (bm/tm * bn) tid **
    block_id (mrows * mcols) bid
{
  let sarA : gpu_array et (bm * bk) = fst sh;
  let sarB : gpu_array et (bk * bn) = fst (snd sh);
  rewrite each fst sh as sarA;
  rewrite each fst (snd sh) as sarB;

  gpu_pts_to_ref sarA;
  gpu_pts_to_ref sarB;

  unfold barrier_tok tm slA slB sarA sarB 0 tid;

  M.gpu_matrix_abs' slA sarA;
  let sA = M.from_array slA sarA;
  rewrite each M.from_array slA sarA as sA;

  M.gpu_matrix_abs' slB sarB;
  let sB = M.from_array slB sarB;
  rewrite each M.from_array slB sarB as sB;


  let mrow = bid /^ mcols;
  let mcol = bid %^ mcols;
  let threadRow = tid /^ bn;
  let threadCol = tid %^ bn;

  (* thread-local result cache *)
  let mut cache1d : Pulse.Lib.Array.array et = [| zero #et #_ ; tm |];

  let mut bkIdx  : sz = 0sz;
  while (SZ.(!bkIdx <^ mshared))
    invariant b.
      exists* (vbkIdx : SZ.t{vbkIdx <= mshared}) (cache1dv : lseq et tm).
        pure (b == (SZ.v vbkIdx < mshared)) **
        (bkIdx |-> vbkIdx) **
        (cache1d |-> cache1dv) **
        (gA |-> Frac (fA /. (mrows * mcols * (bm/tm * bn))) eA) **
        (gB |-> Frac (fB /. (mrows * mcols * (bm/tm * bn))) eB) **
        (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. (bm/tm * bn)) x) **
        (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. (bm/tm * bn)) x) **
        B.barrier_tok (barrier_p tm sA sB) (barrier_q tm sA sB) (2 * vbkIdx) tid **
        gpu
  {
    (* This assert should not be needed. I don't know what effect it even has. *)
    let vbkIdx = !bkIdx;
    assert B.barrier_tok (barrier_p tm sA sB) (barrier_q tm sA sB) (2 * vbkIdx) tid;
    even_2x vbkIdx;
    rewrite
        (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. (bm/tm * bn)) x) **
        (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. (bm/tm * bn)) x)
      as barrier_p tm sA sB (2 * vbkIdx) tid;

    B.barrier_wait ();
    rewrite (barrier_q tm sA sB (2 * vbkIdx) tid)
         as own_1_cell sA (tid /^ bk) (tid%bk) ** own_1_cell sB (tid/bn) (tid%bn);

    (* At this point we exclusively own a full column of the SHMEM
       cache. Populate it. *)
    populate_shmem tm sA sB gA gB mrow !bkIdx mcol tid;

    assert (B.barrier_tok (barrier_p tm sA sB) (barrier_q tm sA sB) (2 * vbkIdx + 1) tid);
    odd_2x1 vbkIdx;
    assert (pure (odd (2 * vbkIdx + 1)));
    rewrite own_1_cell sA (tid/bk) (tid%bk) ** own_1_cell sB (tid/bn) (tid%bn)
         as (barrier_p tm sA sB (2 * vbkIdx + 1) tid);
    B.barrier_wait ();
    even_2x (vbkIdx + 1);
    (* sigh *)
    assert (pure (2 * (vbkIdx + 1) == 2 * vbkIdx + 2));
    assert (pure (even (2 * vbkIdx + 2)));
    rewrite (barrier_q tm sA sB (2 * vbkIdx + 1) tid)
    as
      (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. (bm/tm * bn)) x) **
      (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. (bm/tm * bn)) x);

    subproducts1d tm cache1d sA sB threadRow threadCol;

    (* Move to next tile *)
    bkIdx := !bkIdx +^ 1sz;
  };

  ghost
  fn aux (i: natlt tm)
  requires
    (exists* (v: et).
      M4.gpu_matrix_pts_to_cell #et #(SZ.v mrows) #(SZ.v mcols) #(SZ.v bm)
        #(SZ.v bn) #lC gC #1.0R (SZ.v bid / SZ.v mcols)
        (SZ.v bid % SZ.v mcols) (SZ.v tid / SZ.v bn * SZ.v tm + i)
        (SZ.v tid % SZ.v bn) v)
  ensures
    (exists* (v: et).
      M4.gpu_matrix_pts_to_cell #et #(SZ.v mrows) #(SZ.v mcols) #(SZ.v bm)
        #(SZ.v bn) #lC gC #1.0R (SZ.v mrow)
        (SZ.v mcol) (SZ.v threadRow * SZ.v tm + i)
        (SZ.v threadCol) v)
  {
    with v. _;
    rewrite
      (M4.gpu_matrix_pts_to_cell #et #(SZ.v mrows) #(SZ.v mcols) #(SZ.v bm)
          #(SZ.v bn) #lC gC #1.0R (SZ.v bid / SZ.v mcols)
          (SZ.v bid % SZ.v mcols) (SZ.v tid / SZ.v bn * SZ.v tm + i)
          (SZ.v tid % SZ.v bn) v)
    as
      (M4.gpu_matrix_pts_to_cell #et #(SZ.v mrows) #(SZ.v mcols) #(SZ.v bm)
          #(SZ.v bn) #lC gC #1.0R (SZ.v (bid /^ mcols))
          (SZ.v (bid %^ mcols)) (SZ.v (tid /^ bn) * SZ.v tm + i)
          (SZ.v (tid %^ bn)) v);
    rewrite each SZ.v (bid /^ mcols) as SZ.v mrow,
                 SZ.v (bid %^ mcols) as SZ.v mcol,
                 SZ.v (tid /^ bn) as SZ.v threadRow,
                 SZ.v (tid %^ bn) as SZ.v threadCol;
    ()
  };
  forevery_map _ _ aux;

  (* Write all the accumulated dotproducts. *)
  let mut resIdx : sz = 0sz;
  Pulse.Lib.Array.pts_to_len cache1d;
  while (SZ.(!resIdx <^ tm))
    invariant b.
      exists* (vresIdx : SZ.t{vresIdx <= tm}) (dotpv : lseq et tm).
        pure (b == (SZ.v vresIdx < tm)) **
        (resIdx |-> vresIdx) **
        (cache1d |-> dotpv) **
        (forall+ (ii : natlt tm).
          (exists* v.
            m4_pts_to_cell gC #1.0R
              mrow mcol
              (threadRow * tm + ii) threadCol v)) **
        gpu
  {
    let vresIdx = !resIdx;
    forevery_extract #(natlt tm) (SZ.v vresIdx) _;

    let innerTileRow = threadRow *^ tm +^ vresIdx;
    with bi0 bj0 i0 j0 v0.
      rewrite m4_pts_to_cell gC bi0  bj0  i0   j0   v0
          as m4_pts_to_cell gC mrow mcol (threadRow * tm + vresIdx) threadCol v0;

    let v0 = M4.gpu_matrix_read_cell gC mrow mcol (threadRow *^ tm +^ vresIdx) threadCol;
    open Pulse.Lib.Array;
    let v1 = cache1d.(!resIdx);
    let v' = comb v0 v1;
    M4.gpu_matrix_write_cell gC mrow mcol (threadRow *^ tm +^ vresIdx) threadCol v';

    with bi0 bj0 i0 j0 v0.
      rewrite m4_pts_to_cell gC bi0  bj0  i0   j0   v0
          as m4_pts_to_cell gC mrow mcol (threadRow * tm + vresIdx) threadCol v0;

    resIdx := !resIdx +^ 1sz;
    Pulse.Lib.Trade.elim_trade
      (exists* v.
        m4_pts_to_cell gC #1.0R
          mrow mcol
          (threadRow * tm + vresIdx) threadCol v)
      (forall+ (ii : natlt tm).
        (exists* v.
          m4_pts_to_cell gC #1.0R
            mrow mcol
            (threadRow * tm + ii) threadCol v));
  };

  M.gpu_matrix_concr sA; rewrite each M.core sA as sarA;
  M.gpu_matrix_concr sB; rewrite each M.core sB as sarB;

  fold barrier_tok tm slA slB sarA sarB (2 * mshared) tid;

  rewrite each sarA as fst sh;
  rewrite each sarB as fst (snd sh);

  ghost
  fn raux (i: natlt tm)
  requires
    (exists* (v: et).
      M4.gpu_matrix_pts_to_cell #et #(SZ.v mrows) #(SZ.v mcols) #(SZ.v bm)
        #(SZ.v bn) #lC gC #1.0R (SZ.v mrow)
        (SZ.v mcol) (SZ.v threadRow * SZ.v tm + i)
        (SZ.v threadCol) v)
  ensures
    (exists* (v: et).
      M4.gpu_matrix_pts_to_cell #et #(SZ.v mrows) #(SZ.v mcols) #(SZ.v bm)
        #(SZ.v bn) #lC gC #1.0R (SZ.v bid / SZ.v mcols)
        (SZ.v bid % SZ.v mcols) (SZ.v tid / SZ.v bn * SZ.v tm + i)
        (SZ.v tid % SZ.v bn) v)
  {
    with v. _;
    rewrite each SZ.v mrow as SZ.v (bid /^ mcols),
                 SZ.v mcol as SZ.v (bid %^ mcols),
                 SZ.v threadRow as SZ.v (tid /^ bn),
                 SZ.v threadCol as SZ.v (tid %^ bn);
    rewrite
      (M4.gpu_matrix_pts_to_cell #et #(SZ.v mrows) #(SZ.v mcols) #(SZ.v bm)
          #(SZ.v bn) #lC gC #1.0R (SZ.v (bid /^ mcols))
          (SZ.v (bid %^ mcols)) (SZ.v (tid /^ bn) * SZ.v tm + i)
          (SZ.v (tid %^ bn)) v)
    as
      (M4.gpu_matrix_pts_to_cell #et #(SZ.v mrows) #(SZ.v mcols) #(SZ.v bm)
          #(SZ.v bn) #lC gC #1.0R (SZ.v bid / SZ.v mcols)
          (SZ.v bid % SZ.v mcols) (SZ.v tid / SZ.v bn * SZ.v tm + i)
          (SZ.v tid % SZ.v bn) v);
    ()
  };
  forevery_map _ _ raux;
  ()
}
#pop-options


ghost
fn setup
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  // because of how the original code loads into shmem,
  //  the following is required
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#lA : mlayout4 mrows   mshared bm bk)
  (#lB : mlayout4 mshared mcols   bk bn)
  (#lC : mlayout4 mrows   mcols   bm bn)
  {| clayout4 lA, clayout4 lB, clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (#fA : perm)
  (gB : gpu_matrix4 et lB)
  (#fB : perm)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows mshared bm bk)
  (#eB : ematrix4 et mshared mcols bk bn)
  (#eC : ematrix4 et mrows mcols bm bn)
  ()
  requires
    (gA |-> Frac fA eA) **
    (gB |-> Frac fB eB) **
    (gC |-> eC)
  ensures
    (forall+ (bid : natlt (mrows *^ mcols))
             (tid : natlt (bm /^ tm *^ bn)).
      kpre1 comb tm gA gB gC eA eB fA fB bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn block_setup
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (slA : mlayout bm bk)
  (slB : mlayout bk bn)
  (#lA : mlayout4 mrows   mshared bm bk)
  (#lB : mlayout4 mshared mcols   bk bn)
  (#lC : mlayout4 mrows   mcols   bm bn)
  {| clayout4 lA, clayout4 lB, clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (#fA : perm)
  (gB : gpu_matrix4 et lB)
  (#fB : perm)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows mshared bm bk)
  (#eB : ematrix4 et mshared mcols bk bn)
  (#eC : ematrix4 et mrows mcols bm bn)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt (mrows *^ mcols))
  ()
  requires
    block_setup_tok (bm /^ tm *^ bn) **
    live_c_shmems sh **
    (forall+ (tid : natlt (bm/^tm *^ bn)).
      kpre1 comb tm gA gB gC eA eB fA fB bid tid)
  ensures
    block_setup_tok (bm/^tm *^ bn) **
    (forall+ (tid : natlt (bm/^tm *^ bn)).
      kpre comb tm slA slB gA gB gC eA eB fA fB sh bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn block_teardown
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (slA : mlayout bm bk)
  (slB : mlayout bk bn)
  (#lA : mlayout4 mrows   mshared bm bk)
  (#lB : mlayout4 mshared mcols   bk bn)
  (#lC : mlayout4 mrows   mcols   bm bn)
  {| clayout4 lA, clayout4 lB, clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (#fA : perm)
  (gB : gpu_matrix4 et lB)
  (#fB : perm)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows mshared bm bk)
  (#eB : ematrix4 et mshared mcols bk bn)
  (#eC : ematrix4 et mrows mcols bm bn)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt (mrows *^ mcols))
  ()
  requires
    (forall+ (tid : natlt (bm/^tm *^ bn)).
      kpost comb tm slA slB gA gB gC eA eB fA fB sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt (bm/^tm *^ bn)).
      kpost1 comb tm gA gB gC eA eB fA fB bid tid)
{
  admit();
}

ghost
fn teardown
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#lA : mlayout4 mrows   mshared bm bk)
  (#lB : mlayout4 mshared mcols   bk bn)
  (#lC : mlayout4 mrows   mcols   bm bn)
  {| clayout4 lA, clayout4 lB, clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (#fA : perm)
  (gB : gpu_matrix4 et lB)
  (#fB : perm)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows mshared bm bk)
  (#eB : ematrix4 et mshared mcols bk bn)
  (#eC : ematrix4 et mrows mcols bm bn)
  ()
  requires
    (forall+ (bid : natlt (mrows *^ mcols))
             (tid : natlt (bm /^ tm *^ bn)).
      kpost1 comb tm gA gB gC eA eB fA fB bid tid) **
    emp (* frame *)
  ensures
    (gA |-> Frac fA eA) **
    (gB |-> Frac fB eB) **
    (gC |-> MS.mmcomb comb eC eA eB)
{
  // forevery_flatten #(natlt2 mrows mcols) #_ #(natlt tile)
  //   (fun bid tid -> kpost1 comb tile gA gB gC eA eB 1.0R bid tid);
  // forevery_unzip #(natlt2 mrows mcols & natlt tile) _ _;
  // forevery_unzip #(natlt2 mrows mcols & natlt tile) _ _;
  // forevery_tostar #(natlt2 mrows mcols & natlt tile) (fun _tid -> m4_pts_to gA #(1.0R /. mlayout_size lC) eA);

    // (fun (bid, tid) -> kpost1 comb tile gA gB gC eA eB 1.0R bid tid);
  admit();
}

inline_for_extraction noextract
let mk_kernel
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : mlayout bm bk)
  (slB : mlayout bk bn)
  {| clayout slA, clayout slB |}
  (#lA : mlayout4 mrows   mshared bm bk)
  (#lB : mlayout4 mshared mcols   bk bn)
  (#lC : mlayout4 mrows   mcols   bm bn)
  {| clayout4 lA, clayout4 lB, clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (#fA : perm)
  (gB : gpu_matrix4 et lB)
  (#fB : perm)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows mshared bm bk)
  (#eB : ematrix4 et mshared mcols bk bn)
  (#eC : ematrix4 et mrows mcols bm bn)
  (_ : squash (mrows * mcols <= max_blocks
               /\ (bm/tm * bn) <= max_threads))
  : kernel_desc
      ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) ** (gC |-> eC))
      ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) ** (gC |-> MS.mmcomb comb eC eA eB))
= {
  nblk = mrows *^ mcols; //SZ.uint_to_t (SZ.v mrows * SZ.v mcols);
  nthr = (bm /^ tm *^ bn);

  shmems_desc = shmems_desc et bm bn bk;

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt (bm/^tm *^ bn)). kpre1  comb tm gA gB gC eA eB fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt (bm/^tm *^ bn)). kpost1 comb tm gA gB gC eA eB fA fB bid tid);
  setup      = setup    comb tm gA gB gC #eA #eB #eC;
  teardown   = teardown comb tm gA gB gC #eA #eB #eC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    =  block_setup    comb tm slA slB gA gB gC #eA #eB #eC;
  block_teardown = block_teardown comb tm slA slB gA gB gC #eA #eB #eC;

  kpre      = kpre  comb tm slA slB gA gB gC eA eB fA fB;
  kpost     = kpost comb tm slA slB gA gB gC eA eB fA fB;

  f = kf comb tm slA slB gA #fA gB #fB gC #eA #eB;
}

inline_for_extraction noextract
fn mmcomb_gpu
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#lA : mlayout4 mrows   mshared bm bk)
  (#lB : mlayout4 mshared mcols   bk bn)
  (#lC : mlayout4 mrows   mcols   bm bn)
  {| clayout4 lA, clayout4 lB, clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (#fA : perm)
  (gB : gpu_matrix4 et lB)
  (#fB : perm)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows mshared bm bk)
  (#eB : ematrix4 et mshared mcols bk bn)
  (#eC : ematrix4 et mrows mcols bm bn)
  preserves
    cpu **
    (gA |-> Frac fA eA) **
    (gB |-> Frac fB eB)
  requires
    pure (mrows * mcols <= max_blocks) **
    pure (bm/tm * bn <= max_threads) **
    (gC |-> eC)
  ensures
    gC |-> MS.mmcomb comb eC eA eB
{
  (* fixed the inner layouts, or we'd have to propagate this everywhere? *)
  launch_sync (mk_kernel comb tm (R.row_major _ _) (R.row_major _ _) gA gB gC ());
}
