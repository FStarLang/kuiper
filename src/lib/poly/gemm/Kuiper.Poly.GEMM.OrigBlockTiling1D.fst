module Kuiper.Poly.GEMM.OrigBlockTiling1D

#lang-pulse

#set-options "--z3rlimit 30"

open Kuiper
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
module SZ = Kuiper.SizeT
module B = Kuiper.Barrier

module R = Kuiper.Matrix.Reprs

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

let constraint_test
  (#bm #bn #bk : szp)
  (tm : szp{tm /? bm})
  // because of how the original populates shmem,
  //  the following is required
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  =
  assert (bm == bn);
  assert ((bm/tm * bn) == bm * bk /\ (bm/tm * bm) == bn * bk);
  assert ((bm/tm * bm) == bm * bk /\ (bm/tm * bm) == bn * bk);
  assert ((bm/tm * bm) == bm * bk /\ bm * bk == bn * bk);
  assert ((bm/tm * bm) == bm * bk /\ bm == bn);
  assert (SZ.v bm == tm * bk /\ bm == bn);
  // ^ A simpler constraint
  ()

(* without shmem ownership *)
unfold
let kpre1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  // because of how the original populates shmem,
  //  the following is required
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#lA : mlayout4 mrows   mshared bm bk)
  (#lB : mlayout4 mshared mcols   bk bn)
  (#lC : mlayout4 mrows   mcols   bm bn)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (eA : ematrix4 et mrows mshared bm bk)
  (eB : ematrix4 et mshared mcols bk bn)
  (fA fB : perm)
  // as many blocks as output tiles (i.e. tiles in gC)
  (bid : natlt (mrows * mcols))
  //  so elements in a tile divided by tm
  // each thread in a block computes tm many elements in M dimension,
  (tid : natlt (bm/tm * bn))
  : slprop
  =
  (gA |-> Frac (fA /. (mrows * mcols * (bm/tm * bn))) eA) **
  (gB |-> Frac (fB /. (mrows * mcols * (bm/tm * bn))) eB) **
  (forall+ (i : natlt tm).
    (exists* v.
      m4_pts_to_cell gC #1.0R
        // within each block
        (bid / mcols) (bid % mcols)
        // Each thread computes tm many results in a subcolumn of C.
        // bn threads next to each other compute an innertilerow,
        // sharing the row indices
        ((tid / bn * tm) + i)
        // and not sharing the column indices
        (tid % bn)
        v))

(* NO FUNCTIONAL SPEC RIGHT NOW *)
unfold
let kpost1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
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
  (tid : natlt (bm/tm * bn))
  : slprop
  =
  (gA |-> Frac (fA /. (mrows * mcols * (bm/tm * bn))) eA) **
  (gB |-> Frac (fB /. (mrows * mcols * (bm/tm * bn))) eB) **
  (forall+ (i : natlt tm).
    (exists* v.
      m4_pts_to_cell gC #1.0R
        (bid / mcols) (bid % mcols)
        ((tid/bn * tm) + i)
        (tid % bn)
        v))

(* The barrier flip-flops between an initial state
where every threads shares all of the two arrays, and
a second state where every thread owns a single cell
in each array, related to their tid.

So in even steps, they give their shared ownership,
and receive their cells. (p)
*)
let own_1_cell
  (#et : Type0)
  (#rows #cols : szp)
  (#l : mlayout rows cols)
  (m : gpu_matrix et l)
  (i : natlt rows)
  (j : natlt cols)
  : slprop =
  exists* va. gpu_matrix_pts_to_cell m i j va

let barrier_p
  (#et : Type0)
  (#bm #bn #bk : szp)
  (tm : szp{tm `divides` bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#l1 : full_mlayout bm bk)
  (#l2 : full_mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  : B.barrier_side (bm/tm * bn) =
  fun it tid ->
    if even it then
      (exists* (x : ematrix _ _ _). m1 |-> Frac (1.0R /. (bm/tm * bn)) x) **
      (exists* (x : ematrix _ _ _). m2 |-> Frac (1.0R /. (bm/tm * bn)) x)
    else
      own_1_cell m1 (tid/bk) (tid%bk) ** own_1_cell m2 (tid/bn) (tid%bn)

let barrier_q
  (#et : Type0)
  (#bm #bn #bk : szp)
  (tm : szp{tm `divides` bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#l1 : full_mlayout bm bk)
  (#l2 : full_mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  : B.barrier_side (bm/tm * bn) =
  fun it tid -> barrier_p tm m1 m2 (it+1) tid (* flip flop *)


let barrier_tok
  (#et : Type0)
  (#bm #bn #bk : szp)
  (tm : szp{tm /? bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (* This is defined over the base shared gpu_arrays, as
  this spec must make sense before the arrays are viewed as
  a matrix. *)
  (l1 : full_mlayout bm bk)
  (l2 : full_mlayout bk bn)
  (sar1 : gpu_array et (bm * bk))
  (sar2 : gpu_array et (bk * bn))
  (it : nat)
  (tid : natlt (bm/tm * bn))
  : slprop
  =
  B.barrier_tok (barrier_p tm (M.from_array l1 sar1) (M.from_array l2 sar2))
                (barrier_q tm (M.from_array l1 sar1) (M.from_array l2 sar2))
                it tid

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  // because of how the original code loads into shmem,
  //  the following is required
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn) // shmem layouts
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
  (tid : natlt (bm/tm * bn))
  : slprop
  =
  kpre1 comb tm gA gB gC eA eB fA fB bid tid **
  (exists* (x : seq _). fst sh |-> Frac (1.0R /. (bm/tm * bn)) x) **
  (exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. (bm/tm * bn)) x) **
  barrier_tok tm slA slB (fst sh) (fst (snd sh)) 0 tid

unfold
let kpost
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  // because of how the original code loads into shmem,
  //  the following is required
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn) // shmem layouts
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
  (tid : natlt (bm/tm * bn))
  : slprop
  =
  kpost1 comb tm gA gB gC eA eB fA fB bid tid **
  (exists* (x : seq _). fst sh |-> Frac (1.0R /. (bm/tm * bn)) x) **
  (exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. (bm/tm * bn)) x) **
  barrier_tok tm slA slB (fst sh) (fst (snd sh)) (2 * mshared) tid

inline_for_extraction noextract
fn populate_shmem
  (#et : Type0) {| scalar et |}
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : erased nat)
  (tm : szp{tm /? bm})
  // every thread loads a single element for either matrix,
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#slA : full_mlayout bm bk) {| clayout slA |}
  (#slB : full_mlayout bk bn) {| clayout slB |}
  (sA : gpu_matrix et slA)
  (sB : gpu_matrix et slB)
  (#lA : mlayout4 mrows   mshared bm bk)
  (#lB : mlayout4 mshared mcols   bk bn)
  {| clayout4 lA, clayout4 lB |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (#fA #fB : perm)
  (#eA : ematrix4 et mrows   mshared bm bk)
  (#eB : ematrix4 et mshared mcols   bk bn)
  (mrow : szlt mrows)
  (mk : szlt mshared)
  (mcol : szlt mcols)
  (tid : szlt (bm/tm * bn))
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    own_1_cell sA (tid/bk) (tid%bk) **
    own_1_cell sB (tid/bn) (tid%bn)
{
  unfold own_1_cell sA (tid/bk) (tid%bk);
  let va = M4.gpu_matrix_read gA mrow mk (tid /^ bk) (tid %^ bk);
  M.gpu_matrix_write_cell sA (tid /^ bk) (tid %^ bk) va;
  fold own_1_cell sA (tid/bk) (tid%bk);

  unfold own_1_cell sB (tid/bn) (tid%bn);
  let vb = M4.gpu_matrix_read gB mk mcol (tid /^ bn) (tid %^ bn);
  M.gpu_matrix_write_cell sB (tid /^ bn) (tid %^ bn) vb;
  fold own_1_cell sB (tid/bn) (tid%bn);
}

inline_for_extraction noextract
fn subproducts1d
  (#et : Type0) {| scalar et |}
  (#bm #bn #bk: szp)
  (tm : szp{tm /? bm})
  (rch1d : array et) (* register cache, 1d *)
  (#resvs : erased (seq et))
  (#l1 : full_mlayout bm bk) {| clayout l1 |}
  (#l2 : full_mlayout bk bn) {| clayout l2 |}
  (gA : gpu_matrix et l1)
  (gB : gpu_matrix et l2)
  (#eA : ematrix et bm bk)
  (#eB : ematrix et bk bn)
  (#f : perm)
  (arow: szlt (bm/tm))
  (bcol : szlt bn)
  preserves
    gpu **
    gA |-> Frac f eA **
    gB |-> Frac f eB
  requires
    pure (Seq.length resvs == tm) **
    rch1d |-> resvs
  ensures
    exists* resvs'.
      pure (Seq.length resvs' == tm) **
      (rch1d |-> resvs')
{
  let mut dotIdx : sz = 0sz;
  while (SZ.(!dotIdx <^ bk))
    invariant
      exists* (vIdx : SZ.t{vIdx <= bk}) (resvs : erased (lseq et tm)).
        dotIdx |-> vIdx **
        rch1d |-> resvs
  {
    let tmpB = M.gpu_matrix_read gB !dotIdx bcol;
    let mut resIdx = 0sz;
    while (SZ.(!resIdx <^ tm))
      invariant
        exists* (vi : SZ.t{vi <= tm}) (resvs : erased (lseq et tm)).
          resIdx |-> vi **
          rch1d |-> resvs
    {
      let va = M.gpu_matrix_read gA (arow *^ tm +^ !resIdx) !dotIdx;

      open Pulse.Lib.Array;
      let sum0 = rch1d.(!resIdx);
      let sum1 = sum0 `add` (va `mul` tmpB);
      rch1d.(!resIdx) <- sum1;
      resIdx := !resIdx +^ 1sz;
    };
    dotIdx := !dotIdx +^ 1sz;
  }
}

// even 20 isn't evenough for the checking from the terminal
//  (but enough for the vs code extension)
#push-options "--z3rlimit 50"
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
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
  (#eA : ematrix4 et mrows   mshared bm bk)
  (#eB : ematrix4 et mshared mcols   bk bn)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : szlt (mrows * mcols))
  (tid : szlt (bm/tm * bn))
  ()
  norewrite
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
  let (sarA, (sarB, _)) = sh;

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
    invariant
      exists* (vbkIdx : SZ.t{vbkIdx <= mshared}) (cache1dv : lseq et tm).
        bkIdx |-> vbkIdx **
        cache1d |-> cache1dv **
        (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. (bm/tm * bn)) x) **
        (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. (bm/tm * bn)) x) **
        B.barrier_tok (barrier_p tm sA sB) (barrier_q tm sA sB) (2 * vbkIdx) tid
  {
    (* This assert should not be needed. I don't know what effect it even has. *)
    assert B.barrier_tok (barrier_p tm sA sB) (barrier_q tm sA sB) (2 * !bkIdx) tid;
    even_2x !bkIdx;
    rewrite
        (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. (bm/tm * bn)) x) **
        (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. (bm/tm * bn)) x)
      as barrier_p tm sA sB (2 * !bkIdx) tid;

    B.barrier_wait ();
    rewrite (barrier_q tm sA sB (2 * !bkIdx) tid)
         as own_1_cell sA (tid /^ bk) (tid%bk) ** own_1_cell sB (tid/bn) (tid%bn);

    (* At this point we exclusively own a full column of the SHMEM
       cache. Populate it. *)
    populate_shmem tm sA sB gA gB mrow !bkIdx mcol tid;

    assert (B.barrier_tok (barrier_p tm sA sB) (barrier_q tm sA sB) (2 * !bkIdx + 1) tid);
    odd_2x1 !bkIdx;
    assert (pure (odd (2 * !bkIdx + 1)));
    rewrite own_1_cell sA (tid/bk) (tid%bk) ** own_1_cell sB (tid/bn) (tid%bn)
         as (barrier_p tm sA sB (2 * !bkIdx + 1) tid);
    B.barrier_wait ();
    even_2x (!bkIdx + 1);
    (* sigh *)
    let vbkIdx = !bkIdx;
    assert (pure (2 * (vbkIdx + 1) == 2 * vbkIdx + 1 + 1));
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
  norewrite
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
    with v. assert
      M4.gpu_matrix_pts_to_cell #et #(SZ.v mrows) #(SZ.v mcols) #(SZ.v bm)
        #(SZ.v bn) #lC gC #1.0R (SZ.v bid / SZ.v mcols)
        (SZ.v bid % SZ.v mcols) (SZ.v tid / SZ.v bn * SZ.v tm + i)
        (SZ.v tid % SZ.v bn) v;

    rewrite
      (M4.gpu_matrix_pts_to_cell #et #(SZ.v mrows) #(SZ.v mcols) #(SZ.v bm)
          #(SZ.v bn) #lC gC #1.0R (SZ.v bid / SZ.v mcols)
          (SZ.v bid % SZ.v mcols) (SZ.v tid / SZ.v bn * SZ.v tm + i)
          (SZ.v tid % SZ.v bn) v)
    as
      (M4.gpu_matrix_pts_to_cell #et #(SZ.v mrows) #(SZ.v mcols) #(SZ.v bm)
          #(SZ.v bn) #lC gC #1.0R (SZ.v mrow)
          (SZ.v mcol) (SZ.v threadRow * SZ.v tm + i)
          (SZ.v threadCol) v);
    ()
  };
  forevery_map _ _ aux;

  (* Write all the accumulated dotproducts. *)
  let mut resIdx : sz = 0sz;
  // Pulse.Lib.Array.pts_to_len cache1d;
  while (SZ.(!resIdx <^ tm))
    invariant
      exists* (vresIdx : SZ.t{vresIdx <= tm}) (dotpv : lseq et tm).
        resIdx |-> vresIdx **
        cache1d |-> dotpv
  {
    forevery_extract #(natlt tm) (SZ.v !resIdx) _;

    with v0.
      assert m4_pts_to_cell gC mrow mcol (threadRow * tm + !resIdx) threadCol v0;

    let v0 = M4.gpu_matrix_read_cell gC mrow mcol (threadRow *^ tm +^ !resIdx) threadCol;
    open Pulse.Lib.Array;
    let v1 = cache1d.(!resIdx);
    let v' = comb v0 v1;
    M4.gpu_matrix_write_cell gC mrow mcol (threadRow *^ tm +^ !resIdx) threadCol v';

    with v0.
      assert m4_pts_to_cell gC mrow mcol (threadRow * tm + !resIdx) threadCol v0;

    resIdx := !resIdx +^ 1sz;
    Pulse.Lib.Trade.elim_trade _ _
  };

  M.gpu_matrix_concr sA; rewrite each M.core sA as sarA;
  M.gpu_matrix_concr sB; rewrite each M.core sB as sarB;

  rewrite each sA as M.from_array slA sarA;
  rewrite each sB as M.from_array slB sarB;
  fold barrier_tok tm slA slB sarA sarB (2 * mshared) tid;

  rewrite each sarA as fst sh;
  rewrite each sarB as fst (snd sh);

  ghost
  fn raux (i: natlt tm)
    norewrite
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
    with v. assert
      M4.gpu_matrix_pts_to_cell #et #(SZ.v mrows) #(SZ.v mcols) #(SZ.v bm)
        #(SZ.v bn) #lC gC #1.0R (SZ.v mrow)
        (SZ.v mcol) (SZ.v threadRow * SZ.v tm + i)
        (SZ.v threadCol) v;
    rewrite
      (M4.gpu_matrix_pts_to_cell #et #(SZ.v mrows) #(SZ.v mcols) #(SZ.v bm)
          #(SZ.v bn) #lC gC #1.0R (SZ.v mrow)
          (SZ.v mcol) (SZ.v threadRow * SZ.v tm + i)
          (SZ.v threadCol) v)
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
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
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
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
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
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
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
  norewrite
  requires
    can_create_barrier (bm /^ tm *^ bn) **
    live_c_shmems sh **
    (forall+ (tid : natlt (bm/^tm *^ bn)).
      kpre1 comb tm gA gB gC eA eB fA fB bid tid)
  ensures
    consumed_can_create_barrier **
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
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
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
  norewrite
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
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
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
  ()
  norewrite
  requires
    (forall+ (bid : natlt (mrows *^ mcols))
             (tid : natlt (bm /^ tm *^ bn)).
      kpost1 comb tm gA gB gC eA eB fA fB bid tid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> MS.mmcomb comb eC eA eB
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
               /\ (bm/tm * bn) <= max_threads))
  : kernel_desc
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.mmcomb comb eC eA eB)
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
  (bm bn bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /? bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
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
  norewrite
  preserves
    cpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure (mrows * mcols <= max_blocks) **
    pure (bm/tm * bn <= max_threads) **
    gC |-> eC
  ensures
    gC |-> MS.mmcomb comb eC eA eB
{
  (* fixed the inner layouts, or we'd have to propagate this everywhere? *)
  launch_sync (mk_kernel comb tm (R.row_major _ _) (R.row_major _ _) gA gB gC ());
}
