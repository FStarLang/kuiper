module Kuiper.Kernel.GEMM.BlockTiling2D

#lang-pulse

#set-options "--z3rlimit 60"

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix { chest2 }
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling
open Kuiper.Tensor
open Kuiper.Seq.Common { op_At_Bang }

module B = Kuiper.Barrier
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module FB = Kuiper.Kernel.GEMM.FlipFlopBarrier2
module T = Kuiper.Tensor
module CV2 = Kuiper.Kernel.GEMM.Copy.Vec2
module Trade = Pulse.Lib.Trade

module MU = Kuiper.Kernel.GEMM.Util

(* Shared memory description for tiled matmul kernels. *)
inline_for_extraction noextract
let shmems_desc
  (et:Type0) {| sized et |}
  (bm bn bk: szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  : list shmem_desc = [
  SHArray et (bm *^ bk);
  SHArray et (bk *^ bn);
]

inline_for_extraction noextract
let ttile
  (#et : Type0)
  (#m : erased nat)
  (#n : erased nat)
  (#lC : layout2 m n)
  (gC : array2 et lC)
  (bm : nat{bm > 0 /\ bm /?+ m})
  (bn : nat{bn > 0 /\ bn /?+ n})
  (tm : nat{tm > 0 /\ tm /?+ bm})
  (tn : nat{tn > 0 /\ tn /?+ bn})
  (bid : natlt ((m/bm) * (n/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  : array2 et _
  = array2_subtile (array2_subtile gC bm bn (bid/(n/bn)) (bid%(n/bn))) tm tn (tid/(bn/tn)) (tid%(bn/tn))

let ettile
  (#et : Type0)
  (#m : nat)
  (#n : nat)
  (em : chest2 et m n)
  (bm : nat{bm > 0 /\ bm /?+ m})
  (bn : nat{bn > 0 /\ bn /?+ n})
  (tm : nat{tm > 0 /\ tm /?+ bm})
  (tn : nat{tn > 0 /\ tn /?+ bn})
  (bid : natlt ((m/bm) * (n/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  : chest2 et tm tn
  = ematrix_subtile (ematrix_subtile em bm bn (bid/(n/bn)) (bid%(n/bn))) tm tn (tid/(bn/tn)) (tid%(bn/tn))

unfold
let kpre1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  (gA : array2 et lA)
  (eA : chest2 et m k)
  (gB : array2 et lB)
  (eB : chest2 et k n)
  (gC : array2 et lC)
  (eC : chest2 et m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (fA fB : perm)
  (bid : enatlt (m/bm * (n/bn)))
  (tid : enatlt (bm/tm * (bn/tn)))
  : slprop
  =
  gA |-> Frac (fA /. (m/tm * (n/tn))) eA **
  gB |-> Frac (fB /. (m/tm * (n/tn))) eB **
  ttile gC bm bn tm tn bid tid |-> ettile eC bm bn tm tn bid tid **
  pure (SZ.fits (m * n)) **
  pure (aligned 16 (core gA)) **
  pure (aligned 16 (core gB))

unfold
let kpost1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  (gA : array2 et lA)
  (eA : chest2 et m k)
  (gB : array2 et lB)
  (eB : chest2 et k n)
  (gC : array2 et lC)
  (eC : chest2 et m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (fA fB : perm)
  (bid : enatlt (m/bm * (n/bn)))
  (tid : enatlt (bm/tm * (bn/tn)))
  : slprop
  =
  gA |-> Frac (fA /. (m/tm * (n/tn))) eA **
  gB |-> Frac (fB /. (m/tm * (n/tn))) eB **
  ttile gC bm bn tm tn bid tid |-> ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid

unfold
let kpre
  (#et : Type0) {| scalar et, v : has_vec_cpy et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  (gA : array2 et lA)
  (eA : chest2 et m k)
  (gB : array2 et lB)
  (eB : chest2 et k n)
  (gC : array2 et lC)
  (eC : chest2 et m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (fA fB : perm)
  (nthr : nat {nthr == bm/tm * (bn/tn)})
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt (m/bm * (n/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  : slprop
  =
  kpre1 comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid **
  live_c_shmems sh #(1.0R /. (bm/tm * (bn/tn)))

instance kpre_block_sendable
  (#et : Type0) (_:scalar et) (v : has_vec_cpy et)
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  (gA : array2 et lA { is_global gA })
  (eA : chest2 et m k)
  (gB : array2 et lB { is_global gB })
  (eB : chest2 et k n)
  (gC : array2 et lC { is_global gC })
  (eC : chest2 et m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (fA fB : perm)
  (nblk: SZ.t { SZ.v nblk == m/bm * (n/bn) })
  (nthr: SZ.t { SZ.v nthr == bm/tm * (bn/tn) })
  (sh : c_shmems (shmems_desc et bm bn bk))
  (pf : c_shmems_inv sh)
  (i : natlt nblk)
  (j : natlt nthr)
: is_send_across block_of
  (kpre comb gA eA gB eB gC eC bm bn bk slA slB tm tn fA fB nthr sh i j)
= solve

unfold
let kpost
  (#et : Type0) {| scalar et, v : has_vec_cpy et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  (gA : array2 et lA)
  (eA : chest2 et m k)
  (gB : array2 et lB)
  (eB : chest2 et k n)
  (gC : array2 et lC)
  (eC : chest2 et m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (fA fB : perm)
  (nthr : nat {nthr == bm/tm * (bn/tn)})
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt (m/bm * (n/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  : slprop
  =
  kpost1 comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid **
  live_c_shmems sh #(1.0R /. (bm/tm * (bn/tn)))

instance kpost_block_sendable
  (#et : Type0) (_:scalar et) (v : has_vec_cpy et)
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  (gA : array2 et lA { is_global gA })
  (eA : chest2 et m k)
  (gB : array2 et lB { is_global gB })
  (eB : chest2 et k n)
  (gC : array2 et lC { is_global gC })
  (eC : chest2 et m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (fA fB : perm)
  (nblk: SZ.t { SZ.v nblk == m/bm * (n/bn) })
  (nthr: SZ.t { SZ.v nthr == bm/tm * (bn/tn) })
  (sh : c_shmems (shmems_desc et bm bn bk))
  (pf : c_shmems_inv sh)
  (i : natlt nblk)
  (j : natlt nthr)
: is_send_across block_of
  (kpost comb gA eA gB eB gC eC bm bn bk slA slB tm tn fA fB nthr sh i j)
= solve

(* Wrapper for __gmatmul_single that doesn't require refined row/col/to.
   Returns the initial value when arguments are out of bounds. *)
let __gms
  (#et : Type0) {| scalar et |}
  (#m #k #columns : nat)
  (z : et)
  (m1 : chest2 et m k)
  (m2 : chest2 et k columns)
  (row : nat) (col : nat) (to : nat)
  : GTot et
  = if row < m && col < columns && to <= k
    then MS.__gmatmul_single z mul add m1 m2 row col to
    else z

(* Step lemma in the "forward" direction: given a partial sum and one more
   product term, the result is __gms at (d+1). Uses an SMTPat so the SMT
   can apply this automatically inside universal quantifiers. *)
let __gms_fwd
  (#et : Type0) {| scalar et |}
  (#m #k #columns : nat)
  (z : et)
  (m1 : chest2 et m k)
  (m2 : chest2 et k columns)
  (row : nat) (col : nat) (d : nat)
  : Lemma
    (requires row < m /\ col < columns /\ d < k)
    (ensures add (__gms z m1 m2 row col d)
                 (mul (acc2 m1 row d) (acc2 m2 d col))
             == __gms z m1 m2 row col (d + 1))
    [SMTPat (__gms z m1 m2 row col (d + 1))]
  = MS.__gmatmul_single_lemma z mul add m1 m2 row col (d + 1)

(* Tiled accumulation step: computing __gms on a subtile with the previous
   accumulation equals advancing the full accumulation by d more elements. *)
let rec __gms_tiled_step
  (#et : Type0) {| scalar et |}
  (#m #n #k : nat)
  (eA : chest2 et m k)
  (eB : chest2 et k n)
  (bm : pos{bm /?+ m})
  (bn : pos{bn /?+ n})
  (bk : pos{bk /?+ k})
  (mrow : natlt (m/bm))
  (mcol : natlt (n/bn))
  (local_r : natlt bm)
  (local_c : natlt bn)
  (bkIdx : natlt (k/bk))
  (d : nat{d <= bk})
  : Lemma
    (ensures (
      let glob_r = mrow * bm + local_r in
      let glob_c = mcol * bn + local_c in
      __gms (__gms (zero #et) eA eB glob_r glob_c (bkIdx * bk))
            (ematrix_subtile eA bm bk mrow bkIdx)
            (ematrix_subtile eB bk bn bkIdx mcol)
            local_r local_c d
      == __gms (zero #et) eA eB glob_r glob_c (bkIdx * bk + d)))
    (decreases d)
  = if d = 0 then ()
    else (
      __gms_tiled_step eA eB bm bn bk mrow mcol local_r local_c bkIdx (d - 1);
      MS.__gmatmul_single_lemma
        (__gms (zero #et) eA eB (mrow * bm + local_r) (mcol * bn + local_c) (bkIdx * bk))
        mul add
        (ematrix_subtile eA bm bk mrow bkIdx)
        (ematrix_subtile eB bk bn bkIdx mcol)
        local_r local_c d;
      MS.__gmatmul_single_lemma
        (zero #et) mul add eA eB
        (mrow * bm + local_r) (mcol * bn + local_c)
        (bkIdx * bk + d)
    )

(* Non-recursive wrapper for d=bk with SMTPat: the full tile step. *)
let __gms_tile_full
  (#et : Type0) {| scalar et |}
  (#m #n #k : nat)
  (eA : chest2 et m k)
  (eB : chest2 et k n)
  (bm : pos{bm /?+ m})
  (bn : pos{bn /?+ n})
  (bk : pos{bk /?+ k})
  (mrow : natlt (m/bm))
  (mcol : natlt (n/bn))
  (local_r : natlt bm)
  (local_c : natlt bn)
  (bkIdx : natlt (k/bk))
  : Lemma
    (ensures (
      let glob_r = mrow * bm + local_r in
      let glob_c = mcol * bn + local_c in
      __gms (__gms (zero #et) eA eB glob_r glob_c (bkIdx * bk))
            (ematrix_subtile eA bm bk mrow bkIdx)
            (ematrix_subtile eB bk bn bkIdx mcol)
            local_r local_c bk
      == __gms (zero #et) eA eB glob_r glob_c ((bkIdx + 1) * bk)))
    [SMTPat (__gms (__gms (zero #et) eA eB (mrow * bm + local_r) (mcol * bn + local_c) (bkIdx * bk))
                   (ematrix_subtile eA bm bk mrow bkIdx)
                   (ematrix_subtile eB bk bn bkIdx mcol)
                   local_r local_c bk)]
  = __gms_tiled_step eA eB bm bn bk mrow mcol local_r local_c bkIdx bk;
    assert (bkIdx * bk + bk == (bkIdx + 1) * bk)

(* Full accumulation: __gms with zero initial value over the entire shared
   dimension equals matmul_single. *)
let __gms_full
  (#et : Type0) {| scalar et |}
  (#m #n #k : nat)
  (eA : chest2 et m k)
  (eB : chest2 et k n)
  (row : natlt m) (col : natlt n)
  : Lemma (__gms (zero #et) eA eB row col k == MS.matmul_single eA eB row col)
          [SMTPat (__gms (zero #et) eA eB row col k)]
  = ()

(* Helper for the bkIdx loop body: given the old invariant (rchProd tracks
   partial accumulations up to bkIdx*bk) and the subproducts2d postcondition
   (one more tile of bk columns accumulated), derive the new invariant
   (accumulation up to (bkIdx+1)*bk). *)
let __bkIdx_loop_step
  (#et : Type0) {| scalar et |}
  (#m #n #k : nat)
  (eA : chest2 et m k)
  (eB : chest2 et k n)
  (bm : pos{bm /?+ m})
  (bn : pos{bn /?+ n})
  (bk : pos{bk /?+ k})
  (tm : pos{tm /?+ bm})
  (tn : pos{tn /?+ bn})
  (mrow : natlt (m/bm))
  (mcol : natlt (n/bn))
  (threadRow : natlt (bm/tm))
  (threadCol : natlt (bn/tn))
  (bkIdx : natlt (k/bk))
  (old_v : seq et)
  : Lemma
    (requires
      Seq.length old_v == tm * tn /\
      (forall (idx : natlt (tm * tn)).
        old_v @! idx == __gms (zero #et) eA eB
          (mrow * bm + tm * threadRow + idx / tn)
          (mcol * bn + tn * threadCol + idx % tn)
          (bkIdx * bk)))
    (ensures (
      let new_v = Seq.init_ghost (tm * tn) (fun idx ->
        __gms (old_v @! idx)
          (ematrix_subtile eA bm bk mrow bkIdx)
          (ematrix_subtile eB bk bn bkIdx mcol)
          (tm * threadRow + idx / tn)
          (tn * threadCol + idx % tn)
          bk) in
      Seq.length new_v == tm * tn /\
      (forall (idx : natlt (tm * tn)).
        new_v @! idx == __gms (zero #et) eA eB
          (mrow * bm + tm * threadRow + idx / tn)
          (mcol * bn + tn * threadCol + idx % tn)
          ((bkIdx + 1) * bk))))
  = let new_v = Seq.init_ghost (tm * tn) (fun idx ->
      __gms (old_v @! idx)
        (ematrix_subtile eA bm bk mrow bkIdx)
        (ematrix_subtile eB bk bn bkIdx mcol)
        (tm * threadRow + idx / tn)
        (tn * threadCol + idx % tn)
        bk) in
    let aux (idx : natlt (tm * tn))
      : Lemma (new_v @! idx == __gms (zero #et) eA eB
                (mrow * bm + tm * threadRow + idx / tn)
                (mcol * bn + tn * threadCol + idx % tn)
                ((bkIdx + 1) * bk))
      = let local_r : natlt bm = tm * threadRow + idx / tn in
        let local_c : natlt bn = tn * threadCol + idx % tn in
        __gms_tiled_step eA eB bm bn bk mrow mcol local_r local_c bkIdx bk;
        assert (bkIdx * bk + bk == (bkIdx + 1) * bk)
    in
    FStar.Classical.forall_intro aux

inline_for_extraction noextract
fn subproducts2d
  (#et : Type0) {| scalar et |}
  (bm bn bk: szp)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (rchProd: larray et (tm * tn))
  (#vrchProd : erased (seq et))
  (#l1 : layout2 bm bk) {| T.ctlayout l1 |}
  (#l2 : layout2 bk bn) {| T.ctlayout l2 |}
  (gA : array2 et l1)
  (gB : array2 et l2)
  (#eA : chest2 et bm bk)
  (#eB : chest2 et bk bn)
  (#f : perm)
  (arow: szlt (bm/tm))
  (bcol : szlt (bn/tn))
  (#_ : squash (len vrchProd == tm * tn))
  preserves
    gA |-> Frac f eA **
    gB |-> Frac f eB
  requires
    rchProd |-> vrchProd
  ensures
    rchProd |-> Seq.init_ghost (tm * tn)
      (fun idx ->
        let i = idx / tn in
        let j = idx % tn in
        __gms (vrchProd @! idx) eA eB (tm * arow + i) (tn * bcol + j) bk)
{
  open Pulse.Lib.Array;

  let mut dotIdx : sz = 0sz;
  while (!dotIdx <^ bk)
    invariant live dotIdx ** pure (!dotIdx <= bk)
    invariant exists* (v_d : seq et).
      rchProd |-> v_d **
      pure (len v_d == tm * tn /\
            (forall (idx : natlt (tm * tn)).
              v_d @! idx ==
                __gms (vrchProd @! idx) eA eB
                  (tm * arow + idx / tn) (tn * bcol + idx % tn) !dotIdx))
    decreases (bk - !dotIdx)
  {
    (* register caches *)
    let mut rAcol : Pulse.Lib.Array.array et = [| zero #et #_ ; tm |];
    let mut rBrow : Pulse.Lib.Array.array et = [| zero #et #_ ; tn |];

    let mut j0 = 0sz;
    while (!j0 <^ tm)
      invariant exists* (vj0 : sz{SZ.v vj0 <= tm}). j0 |-> vj0
      invariant exists* (vrAcol : lseq et tm).
        rAcol |-> vrAcol **
        pure (forall (k : natlt (!j0)).
                vrAcol @! k == acc2 eA (tm * arow + k) !dotIdx)
      decreases (tm - !j0)
    {
      pts_to_len rAcol;
      let vj0r = !j0;
      let vdir = !dotIdx;
      let va = tensor_read gA ((tm *^ arow +^ vj0r <: szlt _), ((vdir <: szlt _), ()));
      rAcol.(!j0) <- va;
      j0 := !j0 +^ 1sz;
    };
    with vrAcol. assert rAcol |-> vrAcol;
    assert
        pure (forall (k : natlt tm).
                vrAcol @! k == acc2 eA (tm * arow + k) !dotIdx);

    let mut j1 = 0sz;
    while (!j1 <^ tn)
      invariant exists* (vj1 : sz{SZ.v vj1 <= tn}). j1 |-> vj1
      invariant exists* (vrBrow : lseq et tn).
        rBrow |-> vrBrow **
        pure (forall (k : natlt (!j1)).
                vrBrow @! k == acc2 eB !dotIdx (tn * bcol + k))
      decreases (tn - !j1)
    {
      pts_to_len rBrow;
      let vdir2 = !dotIdx;
      let vj1r = !j1;
      let vb = tensor_read gB ((vdir2 <: szlt _), ((tn *^ bcol +^ vj1r <: szlt _), ()));
      rBrow.(!j1) <- vb;
      j1 := !j1 +^ 1sz;
    };
    with vrBrow. assert rBrow |-> vrBrow;
    assert
        pure (forall (k : natlt tn).
                vrBrow @! k == acc2 eB !dotIdx (tn * bcol + k));

    with v_cur. assert rchProd |-> v_cur;

    let mut resIdxM = 0sz;
    while (!resIdxM <^ tm)
      invariant live resIdxM ** pure (!resIdxM <= tm)
      invariant exists* (v_m : seq et).
        rchProd |-> v_m **
        pure (len v_m == tm * tn /\
              (forall (idx : natlt (tm * tn)).
                v_m @! idx ==
                  (if idx < !resIdxM * tn
                   then add (v_cur @! idx) (mul (vrAcol @! (idx / tn)) (vrBrow @! (idx % tn)))
                   else v_cur @! idx)))
      decreases (tm - !resIdxM)
    {
      let mut resIdxN = 0sz;
      while (!resIdxN <^ tn)
        invariant live resIdxN ** pure (!resIdxN <= tn)
        invariant exists* (v_n : seq et).
          rchProd |-> v_n **
          pure (len v_n == tm * tn /\
                (forall (idx : natlt (tm * tn)).
                  v_n @! idx ==
                    (if idx < !resIdxM * tn + !resIdxN
                     then add (v_cur @! idx) (mul (vrAcol @! (idx / tn)) (vrBrow @! (idx % tn)))
                     else v_cur @! idx)))
        decreases (tn - !resIdxN)
      {
        pts_to_len rAcol;
        pts_to_len rBrow;
        pts_to_len rchProd;

        (* works on arrays and therefore does not have the nice matrix abstraction *)
        let ra = rAcol.(!resIdxM);
        let rb = rBrow.(!resIdxN);
        let idx = !resIdxM *^ tn +^ !resIdxN;
        let old = rchProd.(idx);
        let mad = old `add` (ra `mul` rb);
        rchProd.(idx) <- mad;

        resIdxN := !resIdxN +^ 1sz;
      };

      resIdxM := !resIdxM +^ 1sz;
    };

    (* After the double loop: all elements are updated.
       Re-establish the outer loop invariant for dotIdx + 1.
       Help the SMT with the index bounds for __gms_fwd. *)
    assert pure (forall (idx : natlt (tm * tn)).
      tm * arow + idx / tn < bm /\ tn * bcol + idx % tn < bn /\ !dotIdx < bk);

    dotIdx := !dotIdx +^ 1sz;
  };

  with v. assert rchProd |-> v;
  (* Help SMT derive Seq.init_ghost equality from pointwise invariant *)
  assert pure (Seq.equal v (Seq.init_ghost (tm * tn)
    (fun idx ->
      let i = idx / tn in
      let j = idx % tn in
      __gms (vrchProd @! idx) eA eB (tm * arow + i) (tn * bcol + j) bk)));
}

(* Injectivity of a * n + b when 0 <= b < n: if a*n+b == c*n+d with b,d < n then a=c and b=d. *)
let mul_add_inj (#n : pos) (a : nat) (b : natlt n) (c : nat) (d : natlt n)
  : Lemma (requires a * n + b == c * n + d)
          (ensures a == c /\ b == d)
          [SMTPat (a * n + b); SMTPat (c * n + d)]
  = FStar.Math.Lemmas.division_addition_lemma b n a;
    FStar.Math.Lemmas.small_div b n;
    FStar.Math.Lemmas.division_addition_lemma d n c;
    FStar.Math.Lemmas.small_div d n

(* ettile commutes with chest_comb (and thus mmcomb) pointwise.
   This needs normalization through ematrix_subtile → mk2 → acc2 chains. *)
let ettile_matmul_pointwise
  (#et : Type0) {| scalar et |}
  (#m #n : nat)
  (#k : nat)
  (eA : chest2 et m k)
  (eB : chest2 et k n)
  (bm : nat{bm > 0 /\ bm /?+ m})
  (bn : nat{bn > 0 /\ bn /?+ n})
  (tm : nat{tm > 0 /\ tm /?+ bm})
  (tn : nat{tn > 0 /\ tn /?+ bn})
  (bid : natlt (m/bm * (n/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  (i : natlt tm) (j : natlt tn)
  : Lemma (acc2 (ettile (MS.matmul eA eB) bm bn tm tn bid tid) i j ==
           acc2 (MS.matmul eA eB)
             (bid/(n/bn) * bm + tid/(bn/tn) * tm + i)
             (bid%(n/bn) * bn + tid%(bn/tn) * tn + j))
  = assert_norm (acc2 (ettile (MS.matmul eA eB) bm bn tm tn bid tid) i j ==
                 acc2 (MS.matmul eA eB)
                   (bid/(n/bn) * bm + tid/(bn/tn) * tm + i)
                   (bid%(n/bn) * bn + tid%(bn/tn) * tn + j))

(* Connects the post-bkIdx-loop state to the epilogue precondition.
   The loop invariant tracks __gms zero eA eB glob_r glob_c k.
   The epilogue needs acc2 (ettile (matmul eA eB) ...) (idx/tn) (idx%tn).
   This lemma bridges them via __gms_full + lemma_matmul_index + ettile normalization. *)
let __post_loop_to_epilogue
  (#et : Type0) {| scalar et |}
  (#m #n #k : nat)
  (eA : chest2 et m k)
  (eB : chest2 et k n)
  (bm : pos{bm /?+ m})
  (bn : pos{bn /?+ n})
  (tm : pos{tm /?+ bm})
  (tn : pos{tn /?+ bn})
  (bid : natlt (m/bm * (n/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  (vrch_val : seq et)
  : Lemma
    (requires
      Seq.length vrch_val == tm * tn /\
      (forall (idx : natlt (tm * tn)).
        vrch_val @! idx == __gms (zero #et) eA eB
          (bid/(n/bn) * bm + tm * (tid/(bn/tn)) + idx / tn)
          (bid%(n/bn) * bn + tn * (tid%(bn/tn)) + idx % tn)
          k))
    (ensures
      forall (idx : natlt (tm * tn)).
        vrch_val @! idx == acc2 (ettile (MS.matmul eA eB) bm bn tm tn bid tid) (idx / tn) (idx % tn))
  = let aux (idx : natlt (tm * tn))
      : Lemma (vrch_val @! idx == acc2 (ettile (MS.matmul eA eB) bm bn tm tn bid tid) (idx / tn) (idx % tn))
      = let i : natlt tm = idx / tn in
        let j : natlt tn = idx % tn in
        let br : natlt (m/bm) = bid / (n/bn) in
        let bc : natlt (n/bn) = bid % (n/bn) in
        let tr : natlt (bm/tm) = tid / (bn/tn) in
        let tc : natlt (bn/tn) = tid % (bn/tn) in
        assert (tm * tr + i < bm);
        assert (tn * tc + j < bn);
        assert (br * bm + (tm * tr + i) < m);
        assert (bc * bn + (tn * tc + j) < n);
        let glob_r : natlt m = br * bm + tm * tr + i in
        let glob_c : natlt n = bc * bn + tn * tc + j in
        __gms_full eA eB glob_r glob_c;
        MS.lemma_matmul_index eA eB glob_r glob_c;
        ettile_matmul_pointwise eA eB bm bn tm tn bid tid i j
    in
    Classical.forall_intro aux

let ettile_mmcomb_pointwise
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #n : nat)
  (#k : nat)
  (eA : chest2 et m k)
  (eB : chest2 et k n)
  (eC : chest2 et m n)
  (bm : nat{bm > 0 /\ bm /?+ m})
  (bn : nat{bn > 0 /\ bn /?+ n})
  (tm : nat{tm > 0 /\ tm /?+ bm})
  (tn : nat{tn > 0 /\ tn /?+ bn})
  (bid : natlt (m/bm * (n/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  (i : natlt tm) (j : natlt tn)
  : Lemma (acc2 (ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid) i j ==
           comb (acc2 (ettile eC bm bn tm tn bid tid) i j)
                (acc2 (ettile (MS.matmul eA eB) bm bn tm tn bid tid) i j))
          [SMTPat (acc2 (ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid) i j)]
  = assert_norm (acc2 (ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid) i j ==
                 comb (acc2 (ettile eC bm bn tm tn bid tid) i j)
                      (acc2 (ettile (MS.matmul eA eB) bm bn tm tn bid tid) i j))

(* Pure-arithmetic tile div/mod facts, extracted to top level so they
   typecheck in a minimal context. Inside [epilogue]'s large proof state the
   ambient sizeof SMTPats (the size_layout lemmas in Kuiper.Tensor.Layout)
   pollute the context and make these [forall] asserts ill-typed when stated
   inline. *)
let epilogue_tile_div_mod (tm tn : pos)
  : Lemma (forall (i:natlt tm) (j:natlt tn).
            (i * tn + j) / tn == i /\ (i * tn + j) % tn == j)
  = introduce forall (i:natlt tm) (j:natlt tn).
      (i * tn + j) / tn == i /\ (i * tn + j) % tn == j
    with (FStar.Math.Lemmas.lemma_div_plus j i tn;
          FStar.Math.Lemmas.small_div j tn;
          FStar.Math.Lemmas.lemma_mod_plus j i tn;
          FStar.Math.Lemmas.small_mod j tn)

let epilogue_tile_lt_succ (tm : pos) (tn : pos) (rM : nat) (rN : nat{rN < tn})
  : Lemma (forall (i:natlt tm) (j:natlt tn).
            (i * tn + j < rM * tn + rN + 1 <==>
             (i * tn + j < rM * tn + rN \/ (i == rM /\ j == rN))))
  = introduce forall (i:natlt tm) (j:natlt tn).
      (i * tn + j < rM * tn + rN + 1 <==>
       (i * tn + j < rM * tn + rN \/ (i == rM /\ j == rN)))
    with (lemma_eucl_lt_succ tn i j rM rN)

#push-options "--z3rlimit 30"
inline_for_extraction noextract
fn epilogue
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #n : sz)
  (#k : sz)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (rchProd: larray et (tm * tn))
  (#vrch : erased (seq et))
  (#_ : squash (Seq.length vrch == tm * tn))
  (#lC : layout2 m n)
  {| T.ctlayout lC |}
  (gC : array2 et lC)
  (eA : chest2 et m k)
  (eB : chest2 et k n)
  (eC : chest2 et m n)
  (bid : szlt (m/bm * (n/bn)))
  (tid : szlt (bm/tm * (bn/tn)))
  preserves
    rchProd |-> vrch
  requires
    pure (forall (idx : natlt (tm * tn)).
      vrch @! idx == acc2 (ettile (MS.matmul eA eB) bm bn tm tn bid tid) (idx / tn) (idx % tn)) **
    ttile gC bm bn tm tn bid tid |-> ettile eC bm bn tm tn bid tid
  ensures
    ttile gC bm bn tm tn bid tid |-> ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid
{
  (* Help the SMT connect vrch to the matmul subtile via div/mod *)
  epilogue_tile_div_mod tm tn;
  assert pure (forall (i:natlt tm) (j:natlt tn).
    vrch @! (i * tn + j) == acc2 (ettile (MS.matmul eA eB) bm bn tm tn bid tid) i j);

  let t_tile = ttile gC (SZ.v bm) (SZ.v bn) (SZ.v tm) (SZ.v tn) (SZ.v bid) (SZ.v tid);
  assert (rewrites_to t_tile (ttile gC (SZ.v bm) (SZ.v bn) (SZ.v tm) (SZ.v tn) (SZ.v bid) (SZ.v tid)));

  let eC_tile = Ghost.hide (ettile eC (SZ.v bm) (SZ.v bn) (SZ.v tm) (SZ.v tn) (SZ.v bid) (SZ.v tid));

  let mut resIdxM = 0sz;
  while (!resIdxM <^ tm)
    invariant live resIdxM ** pure (!resIdxM <= tm)
    invariant exists* (m_cur : chest2 et tm tn).
      t_tile |-> m_cur **
      pure (forall (i:natlt tm) (j:natlt tn).
        acc2 m_cur i j ==
          (if i < !resIdxM
           then comb (acc2 eC_tile i j) (vrch @! (i * tn + j))
           else acc2 eC_tile i j))
    decreases (tm - !resIdxM)
  {
    let mut resIdxN = 0sz;
    while (!resIdxN <^ tn)
      invariant live resIdxN ** pure (!resIdxN <= tn)
      invariant exists* (m_cur : chest2 et tm tn).
        t_tile |-> m_cur **
        pure (forall (i:natlt tm) (j:natlt tn).
          acc2 m_cur i j ==
            (if i * tn + j < !resIdxM * tn + !resIdxN
             then comb (acc2 eC_tile i j) (vrch @! (i * tn + j))
             else acc2 eC_tile i j))
      decreases (tn - !resIdxN)
    {
      open Pulse.Lib.Array;
      pts_to_len rchProd;

      (* Combine the new result in the register cache to the value from gC and
      overwrite the the cell in gC *)
      let vrm = !resIdxM;
      let vrn = !resIdxN;
      let v0 = tensor_read t_tile ((vrm <: szlt _), ((vrn <: szlt _), ()));
      let v1 = rchProd.(!resIdxM *^ tn +^ !resIdxN);
      let v' = comb v0 v1;
      tensor_write t_tile ((vrm <: szlt _), ((vrn <: szlt _), ())) v';

      (* Key arithmetic fact for the invariant step: for (i,j) in bounds,
         i*tn+j == resIdxM*tn+resIdxN iff i==resIdxM /\ j==resIdxN.
         This is needed so the SMT can connect upd2 to the linearized
         index comparison in the invariant. *)
      assert pure (forall (i:natlt tm) (j:natlt tn).
        i * tn + j == !resIdxM * tn + !resIdxN <==> (i == !resIdxM /\ j == !resIdxN));

      // Bridge for invariant step: decompose `< bound+1` into
      // `< bound` (handled by old invariant) or `== bound` (freshly written).
      // Use lemma_eucl_lt_succ for each (i,j) to avoid flaky Z3 non-linear reasoning.
      let rM = !resIdxM;
      let rN = !resIdxN;
      epilogue_tile_lt_succ tm tn (SZ.v rM) (SZ.v rN);

      resIdxN := !resIdxN +^ 1sz;
    };

    (* Bridge inner→outer: when resIdxN==tn, the linearized condition
       i*tn+j < resIdxM*tn+tn is equivalent to i <= resIdxM, and
       since j < tn, also to i < resIdxM+1. *)
    assert pure (forall (i:natlt tm) (j:natlt tn).
      i * tn + j < !resIdxM * tn + tn <==> i <= !resIdxM);

    resIdxM := !resIdxM +^ 1sz;
  };

  with m. assert tensor_pts_to t_tile m;

  assert pure (Kuiper.Chest.equal m (ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid));
  ()
}
#pop-options

#push-options "--fuel 1 --ifuel 1"
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et) str_B))
  (gA : array2 et lA)
  (#eA : chest2 et m k)
  (gB : array2 et lB)
  (#eB : chest2 et k n)
  (gC : array2 et lC)
  (#eC : chest2 et m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (#fA #fB : perm)
  (nthr : nat {nthr == bm/tm * (bn/tn)})
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : szlt (m/bm * (n/bn)))
  (tid : szlt (bm/tm * (bn/tn)))
  ()
  norewrite
  requires
    gpu **
    kpre comb gA eA gB eB gC eC bm bn bk slA slB tm tn fA fB nthr sh bid tid **
    thread_id (bm/tm * (bn/tn)) tid **
    block_id (m/bm * (n/bn)) bid **
    B.barrier_tok (FB.contract eA eB slA slB (fst sh) (fst (snd sh)) nthr bid) **
    B.barrier_state 0
  ensures
    gpu **
    kpost comb gA eA gB eB gC eC bm bn bk slA slB tm tn fA fB nthr sh bid tid **
    thread_id (bm/tm * (bn/tn)) tid **
    block_id (m/bm * (n/bn)) bid **
    B.barrier_tok (FB.contract eA eB slA slB (fst sh) (fst (snd sh)) nthr bid) **
    B.barrier_state (2 * (k / bk))
{
  unfold_c_shmems sh (`%shmems_desc);
  let (sarA, (sarB, _)) = sh;

  gpu_pts_to_ref sarA;
  gpu_pts_to_ref sarB;
  tensor_pts_to_ref gA;
  tensor_pts_to_ref gB;

  tensor_abs' slA sarA;
  let sA = from_array slA sarA;
  rewrite each from_array slA sarA as sA;

  tensor_abs' slB sarB;
  let sB = from_array slB sarB;
  rewrite each from_array slB sarB as sB;

  let num_k_tiles = k /^ bk;
  let num_n_tiles = n /^ bn;
  let mrow = bid /^ num_n_tiles;
  let mcol = bid %^ num_n_tiles;

  let threadRow = tid /^ (bn/^tn);
  let threadCol = tid %^ (bn/^tn);

  (* register caches *)
  assert pure (tm <= m);
  assert pure (tn <= n);
  assert pure (tm * tn <= m * n);
  assert pure (SZ.fits (tm * tn)); // should be obvious
  let mut rchProd : Pulse.Lib.Array.array et = [| zero #et #_ ; tm*^tn |];

  with x. fold FB.bp_sharing sA x nthr;
  with x. fold FB.bp_sharing sB x nthr;

  let mut bkIdx  : sz = 0sz;
  while (!bkIdx <^ num_k_tiles)
    invariant live bkIdx ** pure (!bkIdx <= num_k_tiles)
    invariant exists* (v : seq et).
      rchProd |-> v **
      pure (len v == tm * tn /\
        forall (idx : natlt (tm * tn)).
          v @! idx == __gms (zero #et) eA eB
            (mrow * bm + tm * threadRow + idx / tn)
            (mcol * bn + tn * threadCol + idx % tn)
            (!bkIdx * bk))
    invariant B.barrier_state (2 * !bkIdx) **
        (exists* (x : chest2 _ _ _). FB.bp_sharing sA x nthr) **
        (exists* (x : chest2 _ _ _). FB.bp_sharing sB x nthr)
    decreases (num_k_tiles - !bkIdx)
  {
    even_2x !bkIdx;
    FB.fold_barrier_p_even eA eB sA sB nthr bid !bkIdx tid;
    rewrite FB.barrier_p eA eB sA sB nthr bid (2 * !bkIdx) tid
         as (FB.contract eA eB slA slB sarA sarB nthr bid).rin (2 * !bkIdx) tid;

    B.barrier_wait ();

    rewrite (FB.contract eA eB slA slB sarA sarB nthr bid).rout (2 * !bkIdx) tid
         as (FB.barrier_q eA eB sA sB nthr bid (2 * !bkIdx) tid);
    FB.unfold_barrier_q_even eA eB sA sB nthr bid !bkIdx tid;

    {
      unfold FB.live_strided_chunks sA nthr tid;
      with edstA. assert (FB.own_strided_chunks sA edstA nthr tid);
      rewrite FB.own_strided_chunks sA edstA nthr tid
           as CV2.own_strided_chunks sA edstA nthr tid;

      let tileA = array2_extract_tile_ro' gA
        (SZ.v bm) (SZ.v bk) (SZ.v mrow) (SZ.v !bkIdx);

      Kuiper.Divides.lemma_divides_product_l (chunk et) str_A.stride (mrow * bm);
      Kuiper.Divides.lemma_divides_product_r (chunk et) !bkIdx bk;
      Kuiper.Divides.lemma_divides_sum (chunk et) str_A.offset (str_A.stride * (mrow * bm));
      Kuiper.Divides.lemma_divides_sum (chunk et) (str_A.offset + str_A.stride * (mrow * bm)) (!bkIdx * bk);
      assert pure (chunk et /?+ (str_A.offset + str_A.stride * (mrow * bm) + (!bkIdx * bk)));

      CV2.cp_array2_vec bm bk tileA sA (bm/^tm *^ (bn/^tn)) tid;

      Trade.elim_trade _ _;
    };

    {
      unfold FB.live_strided_chunks sB nthr tid;
      with edstB. assert (FB.own_strided_chunks sB edstB nthr tid);
      rewrite FB.own_strided_chunks sB edstB nthr tid
           as CV2.own_strided_chunks sB edstB nthr tid;

      let tileB = array2_extract_tile_ro' gB
        (SZ.v bk) (SZ.v bn) (SZ.v !bkIdx) (SZ.v mcol);

      Kuiper.Divides.lemma_divides_product_l (chunk et) str_B.stride (!bkIdx * bk);
      Kuiper.Divides.lemma_divides_product_r (chunk et) mcol bn;
      Kuiper.Divides.lemma_divides_sum (chunk et) str_B.offset (str_B.stride * (!bkIdx * bk));
      Kuiper.Divides.lemma_divides_sum (chunk et) (str_B.offset + str_B.stride * (!bkIdx * bk)) (mcol * bn);
      assert pure (chunk et /?+ (str_B.offset + str_B.stride * (!bkIdx * bk) + (mcol * bn)));

      CV2.cp_array2_vec bk bn tileB sB (bm/^tm *^ (bn/^tn)) tid;

      Trade.elim_trade _ _;
    };

    // Convert back from CV2 to FB own_strided_chunks for the barrier
    rewrite CV2.own_strided_chunks sA (ematrix_subtile eA bm bk mrow !bkIdx) nthr tid
         as FB.own_strided_chunks sA (ematrix_subtile eA bm bk mrow !bkIdx) nthr tid;
    rewrite CV2.own_strided_chunks sB (ematrix_subtile eB bk bn !bkIdx mcol) nthr tid
         as FB.own_strided_chunks sB (ematrix_subtile eB bk bn !bkIdx mcol) nthr tid;

    odd_2x1 !bkIdx;
    assert (pure (odd (2 * !bkIdx + 1)));
    FB.fold_barrier_p_odd eA eB sA sB nthr bid mrow mcol !bkIdx tid;
    rewrite FB.barrier_p eA eB sA sB nthr bid (2 * !bkIdx + 1) tid
         as (FB.contract eA eB slA slB sarA sarB nthr bid).rin (2 * !bkIdx + 1) tid;

    B.barrier_wait ();

    even_2x (SZ.v !bkIdx + 1);
    assert (pure (2 * (SZ.v !bkIdx + 1) == 2 * !bkIdx + 2));
    assert (pure (even (2 * !bkIdx + 2)));
    assert (pure (odd (2 * !bkIdx + 1)));
    assert pure ((2 * !bkIdx + 1) < (2 * (k /^ bk)));
    assert pure ((2 * !bkIdx + 1) / 2 == !bkIdx);
    rewrite (FB.contract eA eB slA slB sarA sarB nthr bid).rout (2 * !bkIdx + 1) tid
        as FB.barrier_q eA eB sA sB nthr bid (2 * !bkIdx + 1) tid;
    FB.unfold_barrier_q_odd eA eB sA sB nthr bid mrow mcol !bkIdx tid;

    unfold FB.bp_sharing sA (ematrix_subtile eA bm bk mrow !bkIdx) nthr;
    unfold FB.bp_sharing sB (ematrix_subtile eB bk bn !bkIdx mcol) nthr;

    with old_v. assert (rchProd |-> old_v);
    pts_to_len rchProd;
    subproducts2d bm bn bk tm tn rchProd sA sB threadRow threadCol;
    __bkIdx_loop_step eA eB bm bn bk tm tn mrow mcol threadRow threadCol !bkIdx old_v;

    fold FB.bp_sharing sA (ematrix_subtile eA bm bk mrow !bkIdx) nthr;
    fold FB.bp_sharing sB (ematrix_subtile eB bk bn !bkIdx mcol) nthr;

    // What the hell.
    Math.Lemmas.distributivity_add_right 2 (!bkIdx) 1;
    assert (pure (2 * (!bkIdx + 1) == 2 * !bkIdx + 2 * 1));
    assert (pure (2 * (!bkIdx + 1) == 2 * !bkIdx + 1 + 1));

    bkIdx := !bkIdx +^ 1sz;
  };

  with emA. unfold FB.bp_sharing sA emA nthr;
  with emB. unfold FB.bp_sharing sB emB nthr;

  let vbkIdx = !bkIdx;
  assert pure (vbkIdx <= num_k_tiles);
  assert pure (not (vbkIdx < num_k_tiles));
  assert pure (vbkIdx == num_k_tiles); // Somehow this is flaky.

  (* After the loop: rchProd[idx] == __gms zero eA eB glob_r glob_c (num_k_tiles * bk)
     == __gms zero eA eB glob_r glob_c k == matmul_single eA eB glob_r glob_c
     == acc2 (matmul eA eB) glob_r glob_c == acc2 (ettile (matmul eA eB) ...) (idx/tn) (idx%tn) *)
  with vrch_val. assert (rchProd |-> vrch_val);
  pts_to_len rchProd;
  assert pure (num_k_tiles * bk == k);
  __post_loop_to_epilogue eA eB bm bn tm tn bid tid vrch_val;
  epilogue comb bm bn tm tn rchProd gC eA eB eC bid tid;

  tensor_concr sA; rewrite each core sA as sarA;
  tensor_concr sB; rewrite each core sB as sarB;

  rewrite each sarA as fst sh;
  rewrite each sarB as fst (snd sh);

  fold_c_shmems sh #(1.0R /. (bm/tm * (bn/tn))) (`%shmems_desc);
  ()
}
#pop-options

ghost
fn setup
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA)
  (eA : chest2 et m k)
  (gB : array2 et lB)
  (eB : chest2 et k n)
  (gC : array2 et lC)
  (eC : chest2 et m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  // (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_: squash (SZ.fits (m * n)))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (nblk : szp{SZ.v nblk == m/bm * (n/bn)})
  (nthr : szp{SZ.v nthr == bm/tm * (bn/tn)})
  (#_ : squash (aligned 16 (core gA) /\ aligned 16 (core gB)))
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
      kpre1 comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid) **
    emp (* frame *)
{
  let n_total = m/tm * (n/tn);

  (* Step 1: Share gA/gB *)
  tensor_share_n gA n_total;
  tensor_share_n gB n_total;

  (* Step 2: Tile gC at block level *)
  array2_tile gC (SZ.v bm) (SZ.v bn);
  forevery_rw_size2 (m / bm) (SZ.v (m /^ bm)) (n / bn) (SZ.v (n /^ bn));

  (* Step 3: For each block tile, tile at thread level and collapse to tid *)
  forevery_map_2
    (fun (br : natlt (SZ.v (m /^ bm))) (bc : natlt (SZ.v (n /^ bn))) ->
      array2_subtile gC (SZ.v bm) (SZ.v bn) br bc |->
        Frac 1.0R (ematrix_subtile eC (SZ.v bm) (SZ.v bn) br bc))
    (fun (br : natlt (SZ.v (m /^ bm))) (bc : natlt (SZ.v (n /^ bn))) ->
      forall+ (tid : natlt (bm/tm * (bn/tn))).
        let tr = tid / (bn/tn) in
        let tc = tid % (bn/tn) in
        array2_subtile (array2_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc |->
          Frac 1.0R (ematrix_subtile (ematrix_subtile eC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc))
    fn br bc {
      array2_tile (array2_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn);
      forevery_unfactor' (bm/tm * (bn/tn)) (bm/tm) (bn/tn)
        (fun (tr : natlt (bm/tm)) (tc : natlt (bn/tn)) ->
          array2_subtile (array2_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc |->
            Frac 1.0R (ematrix_subtile (ematrix_subtile eC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc));
    };

  (* Step 4: Collapse (br, bc) → bid *)
  forevery_rw_size2 (SZ.v (m /^ bm)) (m / bm) (SZ.v (n /^ bn)) (n / bn);
  forevery_unfactor' (m/bm * (n/bn)) (m/bm) (n/bn)
    (fun (br : natlt (m/bm)) (bc : natlt (n/bn)) ->
      forall+ (tid : natlt (bm/tm * (bn/tn))).
        let tr = tid / (bn/tn) in
        let tc = tid % (bn/tn) in
        array2_subtile (array2_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc |->
          Frac 1.0R (ematrix_subtile (ematrix_subtile eC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc));

  (* Step 5: Factor gA/gB to 2D *)
  (* Divisibility chain: m/tm == (m/bm) * (bm/tm), n/tn == (n/bn) * (bn/tn) *)
  lemma_div_product tm bm m;
  assert pure (m/tm == (bm/tm) * (m/bm));
  lemma_div_product tn bn n;
  assert pure (n/tn == (bn/tn) * (n/bn));
  assert pure (n_total == (m/bm * (n/bn)) * (bm/tm * (bn/tn)));
  forevery_factor n_total (m/bm * (n/bn)) (bm/tm * (bn/tn))
    (fun _ -> gA |-> Frac (fA /. n_total) eA);
  forevery_factor n_total (m/bm * (n/bn)) (bm/tm * (bn/tn))
    (fun _ -> gB |-> Frac (fB /. n_total) eB);

  (* Step 6: Zip and fold kpre1 *)
  forevery_zip3_2
    (fun (_ : natlt (m/bm * (n/bn))) (_ : natlt (bm/tm * (bn/tn))) ->
      gA |-> Frac (fA /. n_total) eA)
    (fun (_ : natlt (m/bm * (n/bn))) (_ : natlt (bm/tm * (bn/tn))) ->
      gB |-> Frac (fB /. n_total) eB)
    (fun (bid : natlt (m/bm * (n/bn))) (tid : natlt (bm/tm * (bn/tn))) ->
      let br = bid / (n/bn) in
      let bc = bid % (n/bn) in
      let tr = tid / (bn/tn) in
      let tc = tid % (bn/tn) in
      array2_subtile (array2_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc |->
        Frac 1.0R (ematrix_subtile (ematrix_subtile eC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc));

  forevery_rw_size2 (m/bm * (n/bn)) (SZ.v nblk) (bm/tm * (bn/tn)) (SZ.v nthr);

  (* Step 7: Fold into kpre1 — introduce pure facts *)
  forevery_map_2
    (fun (bid : natlt nblk) (tid : natlt nthr) ->
      gA |-> Frac (fA /. n_total) eA **
      gB |-> Frac (fB /. n_total) eB **
      (let br = bid / (n/bn) in
       let bc = bid % (n/bn) in
       let tr = tid / (bn/tn) in
       let tc = tid % (bn/tn) in
       array2_subtile (array2_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc |->
         Frac 1.0R (ematrix_subtile (ematrix_subtile eC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc)))
    (fun (bid : natlt nblk) (tid : natlt nthr) ->
      kpre1 comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid)
    fn bid tid {
      assert pure (SZ.fits (m * n));
      assert pure (aligned 16 (core gA));
      assert pure (aligned 16 (core gB));
    };
}

ghost
fn block_setup
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  (gA : array2 et lA)
  (eA : chest2 et m k)
  (gB : array2 et lB)
  (eB : chest2 et k n)
  (gC : array2 et lC)
  (eC : chest2 et m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  // (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_: squash (SZ.fits (m * n)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (nblk : szp{SZ.v nblk == m/bm * (n/bn)})
  (nthr : szp{SZ.v nthr == bm/tm * (bn/tn)})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpre1 comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid)
  ensures
    (forall+ (tid : natlt nthr).
      kpre comb gA eA gB eB gC eC bm bn bk slA slB tm tn fA fB nthr sh bid tid) **
    emp (* frame *)
{
  gpu_live_c_shmems_share_underspec sh #1.0R #(bm/tm * (bn/tn));
  forevery_rw_size (bm/tm * (bn/tn)) (SZ.v nthr);
  forevery_zip
    (fun (tid : natlt nthr) -> kpre1 comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid)
    _;
}

ghost
fn block_teardown
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  (gA : array2 et lA)
  (eA : chest2 et m k)
  (gB : array2 et lB)
  (eB : chest2 et k n)
  (gC : array2 et lC)
  (eC : chest2 et m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_: squash (SZ.fits (m * n)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (nblk : szp{SZ.v nblk == m/bm * (n/bn)})
  (nthr : szp{SZ.v nthr == bm/tm * (bn/tn)})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
      kpost comb gA eA gB eB gC eC bm bn bk slA slB tm tn fA fB nthr sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpost1 comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid)
{
  forevery_unzip
    (fun (tid : natlt nthr) -> kpost1 comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid)
    (fun (_ : natlt nthr) -> live_c_shmems sh #(1.0R /. (bm/tm * (bn/tn))));

  (* Need to convert natlt nthr → natlt (bm/tm * (bn/tn)) for gather *)
  forevery_rw_size (SZ.v nthr) (bm/tm * (bn/tn))
    #(fun (_ : natlt (SZ.v nthr)) -> live_c_shmems sh #(1.0R /. (bm/tm * (bn/tn))));
  gpu_live_c_shmems_gather_underspec sh #1.0R #(bm/tm * (bn/tn));
}

ghost
fn teardown
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| T.ctlayout lC |}
  (gA : array2 et lA)
  (eA : chest2 et m k)
  (gB : array2 et lB)
  (eB : chest2 et k n)
  (gC : array2 et lC)
  (eC : chest2 et m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_: squash (SZ.fits (m * n)))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (nblk : szp{SZ.v nblk == m/bm * (n/bn)})
  (nthr : szp{SZ.v nthr == bm/tm * (bn/tn)})
  (fA fB : perm)
  ()
  norewrite
  requires
    (forall+ (bid : natlt nblk)
             (tid : natlt nthr).
      kpost1 comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> MS.mmcomb comb eC eA eB
{
  let n_total = m/tm * (n/tn);
  let nblk_val = m/bm * (n/bn);
  let nthr_val = bm/tm * (bn/tn);

  (* Step 1: Collapse 2D → 1D (single forall+ in context, no ambiguity) *)
  forevery_rw_size2 (SZ.v nblk) nblk_val (SZ.v nthr) nthr_val;
  (* Divisibility chain: m/tm == (m/bm) * (bm/tm), n/tn == (n/bn) * (bn/tn) *)
  lemma_div_product tm bm m;
  assert pure (m/tm == (bm/tm) * (m/bm));
  lemma_div_product tn bn n;
  assert pure (n/tn == (bn/tn) * (n/bn));
  assert pure (n_total == nblk_val * nthr_val);
  forevery_unfactor' (m/tm * (n/tn)) nblk_val nthr_val
    (fun (bid : natlt nblk_val) (tid : natlt nthr_val) ->
      gA |-> Frac (fA /. (m/tm * (n/tn))) eA **
      gB |-> Frac (fB /. (m/tm * (n/tn))) eB **
      ttile gC bm bn tm tn bid tid |-> ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid);

  (* Step 2: Separate and gather gA/gB *)
  forevery_unzip3
    (fun (_ : natlt (m/tm * (n/tn))) -> gA |-> Frac (fA /. (m/tm * (n/tn))) eA)
    (fun (_ : natlt (m/tm * (n/tn))) -> gB |-> Frac (fB /. (m/tm * (n/tn))) eB)
    (fun (k : natlt (m/tm * (n/tn))) ->
      ttile gC bm bn tm tn (k / nthr_val) (k % nthr_val) |->
        ettile (MS.mmcomb comb eC eA eB) bm bn tm tn (k / nthr_val) (k % nthr_val));
  tensor_gather_n gA (m/tm * (n/tn));
  tensor_gather_n gB (m/tm * (n/tn));

  (* Step 3: Factor' gC: 1D → 2D (bid, tid) *)
  forevery_factor' (m/tm * (n/tn)) nblk_val nthr_val
    (fun (bid : natlt nblk_val) (tid : natlt nthr_val) ->
      ttile gC bm bn tm tn bid tid |-> ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid);

  (* Step 4: Convert ttile/ettile to explicit subtile form — Pulse tactic can unfold these *)
  forevery_ext_2
    (fun (bid : natlt nblk_val) (tid : natlt nthr_val) ->
      ttile gC bm bn tm tn bid tid |-> ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid)
    (fun (bid : natlt nblk_val) (tid : natlt nthr_val) ->
      let br = bid / (n/bn) in
      let bc = bid % (n/bn) in
      let tr = tid / (bn/tn) in
      let tc = tid % (bn/tn) in
      array2_subtile (array2_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc |->
        Frac 1.0R (ematrix_subtile (ematrix_subtile (MS.mmcomb comb eC eA eB) (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc));

  (* Step 5: Factor' bid → (br, bc) — now the body uses explicit div/mod *)
  forevery_factor' nblk_val (m/bm) (n/bn)
    (fun (br : natlt (m/bm)) (bc : natlt (n/bn)) ->
      forall+ (tid : natlt nthr_val).
        let tr = tid / (bn/tn) in
        let tc = tid % (bn/tn) in
        array2_subtile (array2_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc |->
          Frac 1.0R (ematrix_subtile (ematrix_subtile (MS.mmcomb comb eC eA eB) (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc));

  (* Step 6: Per block, factor' tid → (tr, tc) and untile *)
  forevery_map_2
    (fun (br : natlt (m/bm)) (bc : natlt (n/bn)) ->
      forall+ (tid : natlt nthr_val).
        let tr = tid / (bn/tn) in
        let tc = tid % (bn/tn) in
        array2_subtile (array2_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc |->
          Frac 1.0R (ematrix_subtile (ematrix_subtile (MS.mmcomb comb eC eA eB) (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc))
    (fun (br : natlt (m/bm)) (bc : natlt (n/bn)) ->
      array2_subtile gC (SZ.v bm) (SZ.v bn) br bc |->
        Frac 1.0R (ematrix_subtile (MS.mmcomb comb eC eA eB) (SZ.v bm) (SZ.v bn) br bc))
    fn br bc {
      forevery_factor' nthr_val (bm/tm) (bn/tn)
        (fun (tr : natlt (bm/tm)) (tc : natlt (bn/tn)) ->
          array2_subtile (array2_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc |->
            Frac 1.0R (ematrix_subtile (ematrix_subtile (MS.mmcomb comb eC eA eB) (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc));
      assert pure (SZ.fits ((subtile_layout lC bm bn br bc).ulen));
      array2_untile (array2_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn);
    };

  (* Step 7: Untile block tiles *)
  assert pure (SZ.fits (lC.ulen));
  array2_untile gC (SZ.v bm) (SZ.v bn);
}

#push-options "--z3rlimit_factor 4 --split_queries no --fuel 1 --ifuel 1"
inline_for_extraction noextract
let mk_kernel
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et) str_B))
  (gA : array2 et lA { is_global gA })
  (#fA : perm)
  (#eA : chest2 et m k)
  (gB : array2 et lB { is_global gB })
  (#fB : perm)
  (#eB : chest2 et k n)
  (gC : array2 et lC { is_global gC })
  (#eC : chest2 et m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (nblk : szp{SZ.v nblk == m/bm * (n/bn)})
  (nthr : szp{SZ.v nthr == bm/tm * (bn/tn)})
  (#_ : squash (chunk et * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (m/bm * (n/bn) <= max_blocks
               /\ (bm/tm * (bn/tn)) <= max_threads))
  (#_ : squash (aligned 16 (core gA) /\ aligned 16 (core gB)))
  ()
  : kernel_desc
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.mmcomb comb eC eA eB)
= {
  nblk;// = m/^bm *^ (n/^bn);
  nthr;// = (bm /^ tm *^ (bn /^ tn));

  shmems_desc = shmems_desc et bm bn bk;

  barrier_contract = (fun bid ptrs -> FB.contract eA eB slA slB (fst ptrs) (fst (snd ptrs)) nthr bid);
  barrier_count    = (fun _bid -> 2 * (SZ.v k / SZ.v bk));
  barrier_ok = (fun bid ptrs -> FB.barrier_p_to_q_transform eA eB slA slB (fst ptrs) (fst (snd ptrs)) nthr bid);

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt nthr). kpre1  comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt nthr). kpost1 comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid);

  setup      = setup    comb gA eA gB eB gC eC bm bn bk tm tn nblk nthr fA fB;
  teardown   = teardown comb gA eA gB eB gC eC bm bn bk tm tn nblk nthr fA fB;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup comb gA eA gB eB gC eC bm bn bk slA slB tm tn nblk nthr fA fB;
  block_teardown = block_teardown comb gA eA gB eB gC eC bm bn bk slA slB tm tn nblk nthr fA fB ;

  kpre      = kpre  comb gA eA gB eB gC eC bm bn bk slA slB tm tn fA fB (SZ.v nthr);
  kpost     = kpost comb gA eA gB eB gC eC bm bn bk slA slB tm tn fA fB (SZ.v nthr);

  f = kf comb gA #eA gB #eB gC bm bn bk slA slB tm tn #() #() #() #() #fA #fB (SZ.v nthr);

  block_pre_sendable=solve;
  block_post_sendable=solve;
  kpre_sendable=solve;
  kpost_sendable=solve;
}
#pop-options

inline_for_extraction noextract
fn mmcomb_gpu_exact
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et) str_B))
  (gA : array2 et lA { is_global gA })
  (#eA : chest2 et m k)
  (gB : array2 et lB { is_global gB })
  (#eB : chest2 et k n)
  (gC : array2 et lC { is_global gC })
  (#eC : chest2 et m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (m/bm * (n/bn) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.mmcomb comb eC eA eB)
{
  (* fixed the inner layouts, or we'd have to propagate this everywhere? *)
  launch_sync (mk_kernel comb gA gB gC bm bn bk slA slB tm tn (m/^bm *^ (n/^bn)) (bm/^tm *^ (bn/^tn)) ());
}

inline_for_extraction noextract
fn mmcomb_gpu_approx
  (#et : Type0) {| scalar et, has_vec_cpy et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et) str_B))
  (gA : array2 et lA { is_global gA })
  (#eA : chest2 et m k)
  (gB : array2 et lB { is_global gB })
  (#eB : chest2 et k n)
  (gC : array2 et lC { is_global gC })
  (#eC : chest2 et m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (#fA #fB : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (m/bm * (n/bn) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    exists* (eC' : chest2 et m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB)
{
  mmcomb_gpu_exact #et #_ #_ comb #m #n #k #lA #lB #lC gA #eA gB #eB gC #eC bm bn bk tm tn slA slB #_ #_;
  MU.mmcomb_approx_real comb comb_r eC eA eB rA rB rC;
  ()
}
