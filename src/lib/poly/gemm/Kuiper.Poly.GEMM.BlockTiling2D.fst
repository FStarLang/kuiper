module Kuiper.Poly.GEMM.BlockTiling2D

#lang-pulse

open Kuiper

#set-options "--z3rlimit 20"

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


open Kuiper.EMatrix { ematrix }
open Kuiper.VArray {
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

let thread_row_offset
  (bm bn : szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tid : natlt (bm/tm * (bn/tn)))
  (i : natlt tm)
  : GTot (natlt bm)
  =
  tid / (bn/tn) * tm + i

let own_thread_tile
  (#et : Type0) {| scalar et |}
  (bm bn : szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (#mrows #mcols : erased nat)
  (#lC : mlayout4 mrows mcols bm bn)
  (gC : gpu_matrix4 et lC)
  (bid : natlt (mrows * mcols))
  (tid : natlt (bm/tm * (bn/tn)))
  : slprop
  = (forall+ (i : natlt tm) (j : natlt tn).
      (exists* v.
        m4_pts_to_cell gC
          (bid / mcols) (bid % mcols)
          (thread_row_offset bm bn tm tn tid i) //(tid / (bn/tn) * tm) + i)
          ((tid % (bn/tn) * tn) + j)
          v))

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
  (tid : natlt (bm/tm * (bn/tn)))
  : slprop
  =
  (gA |-> Frac (fA /. (mrows * mcols * (bm/tm * (bn/tn)))) eA) **
  (gB |-> Frac (fB /. (mrows * mcols * (bm/tm * (bn/tn)))) eB) **
  own_thread_tile bm bn tm tn gC bid tid

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
  (tid : natlt (bm/tm * (bn/tn)))
  : slprop
  =
  (gA |-> Frac (fA /. (mrows * mcols * (bm/tm * (bn/tn)))) eA) **
  (gB |-> Frac (fB /. (mrows * mcols * (bm/tm * (bn/tn)))) eB) **
  own_thread_tile bm bn tm tn gC bid tid

let own_cell
  (#et : Type0)
  (#rows #cols : nat)
  (#lm : mlayout rows cols)
  (m : gpu_matrix et lm)
  (i : natlt rows)
  (j : natlt cols)
  : slprop
  = exists* v. gpu_matrix_pts_to_cell m i j v

let div_ceil (a : nat) (b : pos) = (a + (b-1))/b
let own_tile_stride_cells
  (#et : Type0)
  (#rows #cols : nat)
  (#lm : mlayout rows cols)
  (m : gpu_matrix et lm)
  // (em : ematrix et rows cols)
  (nthr : nat)
  (tid : natlt nthr)
  : slprop
  =
  forall+ (it : natlt (div_ceil (rows*cols) nthr)).
    let flat_idx = tid + (it * nthr) <: nat in
    let i = flat_idx/cols <: nat in
    let j = flat_idx%cols <: nat in
      if (i < rows && j < cols)
      then own_cell m i j
      else emp

inline_for_extraction noextract
fn cp_tile
  (#et : Type0) {| scalar et |}
  (brows bcols : szp)
  (#mrows #mcols: erased nat)
  (#slM : mlayout brows bcols)
  {| clayout slM |}
  // this should just be a submatrix dst
  (sM : gpu_matrix et slM)
  // (#esM : ematrix et brows bcols)
  (#lA : mlayout4 mrows mcols brows bcols)
  {| clayout4 lA |}
  // this should just be a submatrix src
  (gM : gpu_matrix4 et lA)
  (#fM : perm)
  (#eM : ematrix4 et mrows mcols brows bcols)
  (mrow : szlt mrows)
  (mcol : szlt mcols)
  (nthr : sz)
  (tid : szlt nthr)
  preserves
    gpu **
    pure (SZ.fits (brows * bcols + nthr)) **
    (gM |-> Frac fM eM)
  requires
    own_tile_stride_cells sM nthr tid
  ensures
    // exists* eM'.
    own_tile_stride_cells sM (*eM'*) nthr tid
{
  let mlen = brows *^ bcols;

  let mut i : sz = tid;
  while (SZ.(!i <^ mlen))
    invariant
      exists* (vi : sz).
        pure (vi >= tid) **
        pure (vi % nthr == tid) **
        (i |-> vi) **
        own_tile_stride_cells sM nthr tid **
        pure (vi < mlen + nthr)
  {
    let v = M4.gpu_matrix_read gM mrow mcol (!i /^ bcols) (!i %^ bcols);
    unfold own_tile_stride_cells sM nthr tid;

    let ite : erased (natlt (div_ceil (brows*bcols) nthr)) = (!i - tid) / nthr;
    forevery_extract (reveal ite) _;

    rewrite each
      (((tid + ite * nthr) / bcols < brows) &&
       ((tid + ite * nthr) % bcols < bcols))
    as true;
    let vi = !i;
    assert pure (SZ.v vi == (SZ.v tid + ite * SZ.v nthr));
    rewrite each (tid + ite * nthr) as vi;

    unfold own_cell sM (vi / bcols) (vi % bcols);
    M.gpu_matrix_write_cell sM (!i /^ bcols) (!i %^ bcols) v;
    fold own_cell sM (vi / bcols) (vi % bcols);

    rewrite each SZ.v vi as (tid + ite * nthr);

    Pulse.Lib.Trade.elim_trade _ _;

    fold own_tile_stride_cells sM nthr tid;

    Math.Lemmas.modulo_addition_lemma vi nthr 1;
    i := !i +^ nthr;
  };

  ()
}

let barrier_p
  (#et : Type0)
  (#bm #bn #bk : szp)
  (#l1 : full_mlayout bm bk)
  (#l2 : full_mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  (nthr : pos)
  : B.barrier_side nthr =
  fun it tid ->
    if even it then
      (exists* (x : ematrix _ _ _). m1 |-> Frac (1.0R /. nthr) x) **
      (exists* (x : ematrix _ _ _). m2 |-> Frac (1.0R /. nthr) x)
    else
      own_tile_stride_cells m1 nthr tid **
      own_tile_stride_cells m2 nthr tid

let barrier_q
  (#et : Type0)
  (#bm #bn #bk : szp)
  (#l1 : full_mlayout bm bk)
  (#l2 : full_mlayout bk bn)
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
  (l1 : full_mlayout bm bk)
  (l2 : full_mlayout bk bn)
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
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
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
  (tid : natlt (bm/tm * (bn/tn)))
  : slprop
  =
  kpre1 comb tm tn gA gB gC eA eB fA fB bid tid **
  (exists* (x : seq _). fst sh |-> Frac (1.0R /. (bm/tm * (bn/tn))) x) **
  (exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. (bm/tm * (bn/tn))) x) **
  barrier_tok slA slB (fst sh) (fst (snd sh)) 0 (bm/tm * (bn/tn)) tid

unfold
let kpost
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
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
  (tid : natlt (bm/tm * (bn/tn)))
  : slprop
  =
  kpost1 comb tm tn gA gB gC eA eB fA fB bid tid **
  (exists* (x : seq _). fst sh |-> Frac (1.0R /. (bm/tm * (bn/tn))) x) **
  (exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. (bm/tm * (bn/tn))) x) **
  barrier_tok slA slB (fst sh) (fst (snd sh)) (2 * mshared) (bm/tm * (bn/tn)) tid

inline_for_extraction noextract
fn populate_shmem
  (#et : Type0) {| scalar et |}
  (bm bn bk : szp)
  (#mrows #mshared #mcols : erased nat)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (#slA : full_mlayout bm bk) {| clayout slA |}
  (#slB : full_mlayout bk bn) {| clayout slB |}
  (sA : gpu_matrix et slA)
  (sB : gpu_matrix et slB)
  (#lA : mlayout4 mrows   mshared bm bk)
  (#lB : mlayout4 mshared mcols   bk bn)
  {| clA: clayout4 lA, clayout4 lB |}
  // should be gpu_matrix and just pointers to the tiles
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (#fA #fB : perm)
  (#eA : ematrix4 et mrows   mshared bm bk)
  (#eB : ematrix4 et mshared mcols   bk bn)
  (mrow : szlt mrows)
  (mk : szlt mshared)
  (mcol : szlt mcols)
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (tid : szlt (bm/^tm *^ (bn/^tn)))
  preserves
    gpu **
    (gA |-> Frac fA eA) **
    (gB |-> Frac fB eB) **
    own_tile_stride_cells sA (bm/tm * (bn/tn)) tid **
    own_tile_stride_cells sB (bm/tm * (bn/tn)) tid
{
  let nthr = bm /^ tm *^ (bn /^ tn);  assert (rewrites_to nthr (bm /^ tm *^ (bn /^ tn)));
  cp_tile bm bk sA gA mrow mk nthr tid;
  cp_tile bk bn sB gB mk mcol nthr tid;
  ();
}

inline_for_extraction noextract
fn subproducts2d
  (#et : Type0) {| scalar et |}
  (bm bn bk: szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (rAcol rBrow rchProd: array et)
  (#vrAcol #vrBrow #vrchProd : erased (seq et))
  (#l1 : full_mlayout bm bk) {| clayout l1 |}
  (#l2 : full_mlayout bk bn) {| clayout l2 |}
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
      exists* (vdotIdx : sz{vdotIdx <= bk}) (vrAcol : erased (lseq et tm))
        (vrBrow : erased (lseq et tn)) (vrchProd : erased (lseq et (tm*tn))).
        (dotIdx |-> vdotIdx) **
        (rAcol |-> vrAcol) **
        (rBrow |-> vrBrow) **
        (rchProd |-> vrchProd)
  {
    open Pulse.Lib.Array;

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

inline_for_extraction noextract
fn epilogue
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (bm bn bk : szp)
  (#mrows #mcols : szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (rchProd: array et)
  (#lC : mlayout4 mrows   mcols   bm bn)
  {| clayout4 lC |}
  (gC : gpu_matrix4 et lC)
  (bid : szlt (mrows * mcols))
  (tid : szlt (bm/tm * (bn/tn)))
  requires
    gpu **
    own_thread_tile bm bn tm tn gC bid tid **
    (exists* vrchProd.
      pure (Seq.length vrchProd == tm * tn) **
      (rchProd |-> vrchProd))
  ensures
    gpu **
    own_thread_tile bm bn tm tn gC bid tid **
    (exists* vrchProd'.
      pure (Seq.length vrchProd' == tm * tn) **
      (rchProd |-> vrchProd'))
{
  unfold own_thread_tile bm bn tm tn gC bid tid;

  let mrow = bid /^ mcols;
  let mcol = bid %^ mcols;
  let tRowOff = tid /^ (bn /^ tn) *^ tm;
  let tColOff = tid %^ (bn /^ tn) *^ tn;

  let mut resIdxM = 0sz;
  while (SZ.(!resIdxM <^ tm))
    invariant
      exists* (vresIdxM : sz{vresIdxM <= tm}) (vrchProd : lseq et (tm*tn)).
        (resIdxM |-> vresIdxM) **
        (rchProd |-> vrchProd)
  {
    let mut resIdxN = 0sz;
    while (SZ.(!resIdxN <^ tn))
      invariant
        exists* (vresIdxN : sz{vresIdxN <= tn}) (vrchProd : lseq et (tm*tn)).
          (resIdxN |-> vresIdxN) **
          (rchProd |-> vrchProd)
    {
      let vresIdxM = !resIdxM;
      let vresIdxN = !resIdxN;

      (* get separate access to the thread's current cell in gC *)
      forevery_extract_2 #(natlt tm) #_ #(natlt tn) (SZ.v vresIdxM) (SZ.v vresIdxN) _;

      (* tame the SMT solver *)
      assert pure(SZ.v tid / (SZ.v bn / SZ.v tn) * SZ.v tm == tRowOff);
      assert pure(SZ.v tid % (SZ.v bn / SZ.v tn) * SZ.v tn == tColOff);

      assert pure (thread_row_offset bm bn tm tn tid vresIdxM == tid/(bn/tn)*tm + vresIdxM ==> tid/(bn/tn)*tm + vresIdxM < bm);
      (* read the current cell in gC *)
      with bi0 bj0 i0 j0 v0.
      rewrite M4.gpu_matrix_pts_to_cell gC bi0  bj0  i0   j0   v0
          as M4.gpu_matrix_pts_to_cell gC mrow mcol (tRowOff + vresIdxM) (tColOff + vresIdxN) v0;
      let v0 = M4.gpu_matrix_read_cell gC mrow mcol (tRowOff +^ vresIdxM) (tColOff +^ vresIdxN);

      (* add the new result in the register cache to the value from gC and overwrite the the cell in gC *)
      open Pulse.Lib.Array;
      let v1 = rchProd.(!resIdxM *^ tn +^ !resIdxN);
      let v' = comb v0 v1;
      M4.gpu_matrix_write_cell gC mrow mcol (tRowOff +^ vresIdxM) (tColOff +^ vresIdxN) v';

      (* return separate access to the thread's current cell in gC *)
      rewrite each (SZ.v mrow) as (bid/mcols);
      rewrite each (SZ.v mcol) as (bid%mcols);
      rewrite each (SZ.v tRowOff) as (tid/(bn/tn) * tm);
      rewrite each (tid/(bn/tn) * tm + (SZ.v vresIdxM)) as thread_row_offset bm bn tm tn tid (SZ.v vresIdxM);
      rewrite each (SZ.v tColOff) as (tid%(bn/tn) * tn);
      Pulse.Lib.Trade.elim_trade _ _;

      resIdxN := !resIdxN +^ 1sz;
    };

    resIdxM := !resIdxM +^ 1sz;
  };

  fold own_thread_tile bm bn tm tn gC bid tid;
  ()
}

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
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
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : szlt (mrows * mcols))
  (tid : szlt (bm/tm * (bn/tn)))
  ()
  requires
    gpu **
    kpre comb tm tn slA slB gA gB gC eA eB fA fB sh bid tid **
    thread_id (bm/tm * (bn/tn)) tid **
    block_id (mrows * mcols) bid
  ensures
    gpu **
    kpost comb tm tn slA slB gA gB gC eA eB fA fB sh bid tid **
    thread_id (bm/tm * (bn/tn)) tid **
    block_id (mrows * mcols) bid
{
  let sarA : gpu_array et (bm * bk) = fst sh;
  let sarB : gpu_array et (bk * bn) = fst (snd sh);
  rewrite each fst sh as sarA;
  rewrite each fst (snd sh) as sarB;

  gpu_pts_to_ref sarA;
  gpu_pts_to_ref sarB;

  unfold barrier_tok slA slB sarA sarB 0 (bm/tm * (bn/tn)) tid;

  M.gpu_matrix_abs' slA sarA;
  let sA = M.from_array slA sarA;
  rewrite each M.from_array slA sarA as sA;

  M.gpu_matrix_abs' slB sarB;
  let sB = M.from_array slB sarB;
  rewrite each M.from_array slB sarB as sB;

  let mrow = bid /^ mcols;
  let mcol = bid %^ mcols;
  let threadRow = tid /^ (bn/^tn);
  let threadCol = tid %^ (bn/^tn);

  (* register caches *)
  let mut rAcol : Pulse.Lib.Array.array et = [| zero #et #_ ; tm |];
  let mut rBrow : Pulse.Lib.Array.array et = [| zero #et #_ ; tn |];
  let mut rchProd : Pulse.Lib.Array.array et = [| zero #et #_ ; tm*^tn |];

  let mut bkIdx  : sz = 0sz;
  while (SZ.(!bkIdx <^ mshared))
    invariant b.
      exists* (vbkIdx : SZ.t{vbkIdx <= mshared}) (vrAcol : lseq et tm) (vrBrow : lseq et tn) (vrchProd : lseq et (tm*tn)).
        pure (b == (SZ.v vbkIdx < mshared)) **
        (bkIdx |-> vbkIdx) **
        (rAcol |-> vrAcol) **
        (rBrow |-> vrBrow) **
        (rchProd |-> vrchProd) **
        (gA |-> Frac (fA /. (mrows * mcols * (bm/tm * (bn/tn)))) eA) **
        (gB |-> Frac (fB /. (mrows * mcols * (bm/tm * (bn/tn)))) eB) **
        (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. (bm/tm * (bn/tn))) x) **
        (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. (bm/tm * (bn/tn))) x) **
        B.barrier_tok (barrier_p sA sB (bm/tm * (bn/tn))) (barrier_q sA sB (bm/tm * (bn/tn))) (2 * vbkIdx) tid **
        gpu
  {
    (* This assert should not be needed. I don't know what effect it even has. *)
    let vbkIdx = !bkIdx;
    assert B.barrier_tok (barrier_p sA sB (bm/tm * (bn/tn))) (barrier_q sA sB (bm/tm * (bn/tn))) (2 * vbkIdx) tid;
    even_2x vbkIdx;
    rewrite
        (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. (bm/tm * (bn/tn))) x) **
        (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. (bm/tm * (bn/tn))) x)
      as barrier_p sA sB (bm/tm * (bn/tn)) (2 * vbkIdx) tid;

    B.barrier_wait ();
    rewrite (barrier_q sA sB (bm/tm * (bn/tn)) (2 * vbkIdx) tid)
        as own_tile_stride_cells sA (bm/tm * (bn/tn)) tid **
           own_tile_stride_cells sB (bm/tm * (bn/tn)) tid;

    populate_shmem bm bn bk tm tn sA sB gA gB mrow !bkIdx mcol tid;

    assert (B.barrier_tok (barrier_p sA sB (bm/tm * (bn/tn))) (barrier_q sA sB (bm/tm * (bn/tn))) (2 * vbkIdx + 1) tid);
    odd_2x1 vbkIdx;
    assert (pure (odd (2 * vbkIdx + 1)));
    rewrite own_tile_stride_cells sA (bm/tm * (bn/tn)) tid **
            own_tile_stride_cells sB (bm/tm * (bn/tn)) tid
         as (barrier_p sA sB (bm/tm * (bn/tn)) (2 * vbkIdx + 1) tid);

    B.barrier_wait ();

    even_2x (vbkIdx + 1);
    assert (pure (2 * (vbkIdx + 1) == 2 * vbkIdx + 2));
    assert (pure (even (2 * vbkIdx + 2)));
    rewrite (barrier_q sA sB (bm/tm * (bn/tn)) (2 * vbkIdx + 1) tid)
    as
      (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. (bm/tm * (bn/tn))) x) **
      (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. (bm/tm * (bn/tn))) x);

    subproducts2d bm bn bk tm tn rAcol rBrow rchProd sA sB threadRow threadCol;

    bkIdx := !bkIdx +^ 1sz;
  };

  epilogue comb bm bn bk tm tn rchProd gC bid tid;

  M.gpu_matrix_concr sA; rewrite each M.core sA as sarA;
  M.gpu_matrix_concr sB; rewrite each M.core sB as sarB;

  fold barrier_tok slA slB sarA sarB (2 * mshared) (bm/tm * (bn/tn)) tid;

  rewrite each sarA as fst sh;
  rewrite each sarB as fst (snd sh);
  ()
}


ghost
fn setup
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (bm bn bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
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
             (tid : natlt (bm /^ tm *^ (bn /^ tn))).
      kpre1 comb tm tn gA gB gC eA eB fA fB bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn block_setup
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (bm bn bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
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
    block_setup_tok (bm /^ tm *^ (bn /^ tn)) **
    live_c_shmems sh **
    (forall+ (tid : natlt (bm/^tm *^ (bn /^ tn))).
      kpre1 comb tm tn gA gB gC eA eB fA fB bid tid)
  ensures
    block_setup_tok (bm/^tm *^ (bn /^ tn)) **
    (forall+ (tid : natlt (bm/^tm *^ (bn /^ tn))).
      kpre comb tm tn slA slB gA gB gC eA eB fA fB sh bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn block_teardown
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (bm bn bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
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
    (forall+ (tid : natlt (bm/^tm *^ (bn /^ tn))).
      kpost comb tm tn slA slB gA gB gC eA eB fA fB sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt (bm/^tm *^ (bn /^ tn))).
      kpost1 comb tm tn gA gB gC eA eB fA fB bid tid)
{
  admit();
}

ghost
fn teardown
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (bm bn bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
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
             (tid : natlt (bm /^ tm *^ (bn /^ tn))).
      kpost1 comb tm tn gA gB gC eA eB fA fB bid tid) **
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
  (bm bn bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
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
               /\ (bm/tm * (bn/tn)) <= max_threads))
  : kernel_desc
      ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) ** (gC |-> eC))
      ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) ** (gC |-> MS.mmcomb comb eC eA eB))
= {
  nblk = mrows *^ mcols;
  nthr = (bm /^ tm *^ (bn /^ tn));

  shmems_desc = shmems_desc et bm bn bk;

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt (bm/^tm *^ (bn/^tn))). kpre1  comb tm tn gA gB gC eA eB fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt (bm/^tm *^ (bn/^tn))). kpost1 comb tm tn gA gB gC eA eB fA fB bid tid);
  setup      = setup    comb bm bn bk tm tn gA gB gC #eA #eB #eC;
  teardown   = teardown comb bm bn bk tm tn gA gB gC #eA #eB #eC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup comb bm bn bk tm tn slA slB gA gB gC #eA #eB #eC;
  block_teardown = block_teardown comb bm bn bk tm tn slA slB gA gB gC #eA #eB #eC;

  kpre      = kpre  comb tm tn slA slB gA gB gC eA eB fA fB;
  kpost     = kpost comb tm tn slA slB gA gB gC eA eB fA fB;

  f = kf comb tm tn slA slB gA #fA gB #fB gC #eA #eB;
}

inline_for_extraction noextract
fn mmcomb_gpu
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (bm bn bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (lA : mlayout4 mrows   mshared bm bk)
  (lB : mlayout4 mshared mcols   bk bn)
  (lC : mlayout4 mrows   mcols   bm bn)
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
    pure (bm/tm * (bn/tn) <= max_threads) **
    (gC |-> eC)
  ensures
    gC |-> MS.mmcomb comb eC eA eB
{
  (* fixed the inner layouts, or we'd have to propagate this everywhere? *)
  launch_sync (mk_kernel comb bm bn bk tm tn slA slB gA gB gC ());
}
