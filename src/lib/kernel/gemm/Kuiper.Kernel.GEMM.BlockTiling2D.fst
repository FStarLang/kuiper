module Kuiper.Kernel.GEMM.BlockTiling2D

#lang-pulse

#set-options "--z3rlimit 60"

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix { ematrix }
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Matrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Tiling
open Kuiper.Kernel.GEMM.Copy.Vec
open Kuiper.Kernel.GEMM.Tiled.Common.Vec
open Kuiper.VArray { varray, varray_pts_to, varray_pts_to_cell }

module B = Kuiper.Barrier
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module FB = Kuiper.Kernel.GEMM.FlipFlopBarrier

module MU = Kuiper.Kernel.GEMM.Util

inline_for_extraction noextract
let ttile
  (#et : Type0)
  (#rows : erased nat)
  (#cols : erased nat)
  (#lC : mlayout rows cols)
  (gC : gpu_matrix et lC)
  (bm : nat{bm > 0 /\ bm /?+ rows})
  (bn : nat{bn > 0 /\ bn /?+ cols})
  (tm : nat{tm > 0 /\ tm /?+ bm})
  (tn : nat{tn > 0 /\ tn /?+ bn})
  (bid : natlt ((rows/bm) * (cols/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  : gpu_matrix et _
  = thread_tile (block_tile gC bm bn bid) tm tn tid

let ettile
  (#et : Type0)
  (#rows : nat)
  (#cols : nat)
  (em : ematrix et rows cols)
  (bm : nat{bm > 0 /\ bm /?+ rows})
  (bn : nat{bn > 0 /\ bn /?+ cols})
  (tm : nat{tm > 0 /\ tm /?+ bm})
  (tn : nat{tn > 0 /\ tn /?+ bn})
  (bid : natlt ((rows/bm) * (cols/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  : ematrix et tm tn
  = ematrix_subtile (ematrix_subtile em bm bn (bid/(cols/bn)) (bid%(cols/bn))) tm tn (tid/(bn/tn)) (tid%(bn/tn))

unfold
let kpre1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et lA)
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
  (eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (fA fB : perm)
  (bid : enatlt (rows/bm * (cols/bn)))
  (tid : enatlt (bm/tm * (bn/tn)))
  : slprop
  =
  gA |-> Frac (fA /. (rows/tm * (cols/tn))) eA **
  gB |-> Frac (fB /. (rows/tm * (cols/tn))) eB **
  ttile gC bm bn tm tn bid tid |-> ettile eC bm bn tm tn bid tid **
  pure (SZ.fits (rows * cols)) **
  pure (aligned 16 (core gA)) **
  pure (aligned 16 (core gB))

unfold
let kpost1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et lA)
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
  (eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (fA fB : perm)
  (bid : enatlt (rows/bm * (cols/bn)))
  (tid : enatlt (bm/tm * (bn/tn)))
  : slprop
  =
  gA |-> Frac (fA /. (rows/tm * (cols/tn))) eA **
  gB |-> Frac (fB /. (rows/tm * (cols/tn))) eB **
  ttile gC bm bn tm tn bid tid |-> ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid

unfold
let kpre
  (#et : Type0) {| scalar et, v : has_vec_cpy et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et lA)
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
  (eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (fA fB : perm)
  (nthr : nat {nthr == bm/tm * (bn/tn)})
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  : slprop
  =
  kpre1 comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid **
  live_c_shmems sh #(1.0R /. (bm/tm * (bn/tn)))

instance kpre_block_sendable
  (#et : Type0) (_:scalar et) (v : has_vec_cpy et)
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (fA fB : perm)
  (nblk: SZ.t { SZ.v nblk == rows/bm * (cols/bn) })
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
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et lA)
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
  (eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (fA fB : perm)
  (nthr : nat {nthr == bm/tm * (bn/tn)})
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  : slprop
  =
  kpost1 comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid **
  live_c_shmems sh #(1.0R /. (bm/tm * (bn/tn)))

instance kpost_block_sendable
  (#et : Type0) (_:scalar et) (v : has_vec_cpy et)
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (fA fB : perm)
  (nblk: SZ.t { SZ.v nblk == rows/bm * (cols/bn) })
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
  (#rows #shared #columns : nat)
  (z : et)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (row : nat) (col : nat) (to : nat)
  : GTot et
  = if row < rows && col < columns && to <= shared
    then MS.__gmatmul_single z mul add m1 m2 row col to
    else z

(* Step lemma in the "forward" direction: given a partial sum and one more
   product term, the result is __gms at (d+1). Uses an SMTPat so the SMT
   can apply this automatically inside universal quantifiers. *)
let __gms_fwd
  (#et : Type0) {| scalar et |}
  (#rows #shared #columns : nat)
  (z : et)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (row : nat) (col : nat) (d : nat)
  : Lemma
    (requires row < rows /\ col < columns /\ d < shared)
    (ensures add (__gms z m1 m2 row col d)
                 (mul (macc m1 row d) (macc m2 d col))
             == __gms z m1 m2 row col (d + 1))
    [SMTPat (__gms z m1 m2 row col (d + 1))]
  = MS.__gmatmul_single_lemma z mul add m1 m2 row col (d + 1)

(* Tiled accumulation step: computing __gms on a subtile with the previous
   accumulation equals advancing the full accumulation by d more elements. *)
let rec __gms_tiled_step
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (bm : pos{bm /?+ rows})
  (bn : pos{bn /?+ cols})
  (bk : pos{bk /?+ shared})
  (mrow : natlt (rows/bm))
  (mcol : natlt (cols/bn))
  (local_r : natlt bm)
  (local_c : natlt bn)
  (bkIdx : natlt (shared/bk))
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
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (bm : pos{bm /?+ rows})
  (bn : pos{bn /?+ cols})
  (bk : pos{bk /?+ shared})
  (mrow : natlt (rows/bm))
  (mcol : natlt (cols/bn))
  (local_r : natlt bm)
  (local_c : natlt bn)
  (bkIdx : natlt (shared/bk))
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
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (row : natlt rows) (col : natlt cols)
  : Lemma (__gms (zero #et) eA eB row col shared == MS.matmul_single eA eB row col)
          [SMTPat (__gms (zero #et) eA eB row col shared)]
  = ()

(* Helper for the bkIdx loop body: given the old invariant (rchProd tracks
   partial accumulations up to bkIdx*bk) and the subproducts2d postcondition
   (one more tile of bk columns accumulated), derive the new invariant
   (accumulation up to (bkIdx+1)*bk). *)
let __bkIdx_loop_step
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (bm : pos{bm /?+ rows})
  (bn : pos{bn /?+ cols})
  (bk : pos{bk /?+ shared})
  (tm : pos{tm /?+ bm})
  (tn : pos{tn /?+ bn})
  (mrow : natlt (rows/bm))
  (mcol : natlt (cols/bn))
  (threadRow : natlt (bm/tm))
  (threadCol : natlt (bn/tn))
  (bkIdx : natlt (shared/bk))
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
  (#l1 : mlayout bm bk) {| clayout l1 |}
  (#l2 : mlayout bk bn) {| clayout l2 |}
  (gA : gpu_matrix et l1)
  (gB : gpu_matrix et l2)
  (#eA : ematrix et bm bk)
  (#eB : ematrix et bk bn)
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
                vrAcol @! k == macc eA (tm * arow + k) !dotIdx)
      decreases (tm - !j0)
    {
      pts_to_len rAcol;
      let va = gpu_matrix_read gA (tm *^ arow +^ !j0) !dotIdx;
      rAcol.(!j0) <- va;
      j0 := !j0 +^ 1sz;
    };
    with vrAcol. assert rAcol |-> vrAcol;
    assert
        pure (forall (k : natlt tm).
                vrAcol @! k == macc eA (tm * arow + k) !dotIdx);

    let mut j1 = 0sz;
    while (!j1 <^ tn)
      invariant exists* (vj1 : sz{SZ.v vj1 <= tn}). j1 |-> vj1
      invariant exists* (vrBrow : lseq et tn).
        rBrow |-> vrBrow **
        pure (forall (k : natlt (!j1)).
                vrBrow @! k == macc eB !dotIdx (tn * bcol + k))
      decreases (tn - !j1)
    {
      pts_to_len rBrow;
      let vb = gpu_matrix_read gB !dotIdx (tn *^ bcol +^ !j1);
      rBrow.(!j1) <- vb;
      j1 := !j1 +^ 1sz;
    };
    with vrBrow. assert rBrow |-> vrBrow;
    assert
        pure (forall (k : natlt tn).
                vrBrow @! k == macc eB !dotIdx (tn * bcol + k));

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

(* macc distributes over mupd: gives the new value at the updated position,
   or the old value elsewhere. *)
let macc_mupd
  (#et : Type0) (#rows #cols : nat)
  (m : ematrix et rows cols)
  (i0 : natlt rows) (j0 : natlt cols) (v : et)
  (i : natlt rows) (j : natlt cols)
  : Lemma (macc (mupd m i0 j0 v) i j == (if i = i0 && j = j0 then v else macc m i j))
          [SMTPat (macc (mupd m i0 j0 v) i j)]
  = ()

(* ettile commutes with matrix_comb (and thus mmcomb) pointwise.
   This needs normalization through ematrix_subtile → mkM → macc chains. *)
let ettile_matmul_pointwise
  (#et : Type0) {| scalar et |}
  (#rows #cols : nat)
  (#shared : nat)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (bm : nat{bm > 0 /\ bm /?+ rows})
  (bn : nat{bn > 0 /\ bn /?+ cols})
  (tm : nat{tm > 0 /\ tm /?+ bm})
  (tn : nat{tn > 0 /\ tn /?+ bn})
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  (i : natlt tm) (j : natlt tn)
  : Lemma (macc (ettile (MS.matmul eA eB) bm bn tm tn bid tid) i j ==
           macc (MS.matmul eA eB)
             (bid/(cols/bn) * bm + tid/(bn/tn) * tm + i)
             (bid%(cols/bn) * bn + tid%(bn/tn) * tn + j))
  = assert_norm (macc (ettile (MS.matmul eA eB) bm bn tm tn bid tid) i j ==
                 macc (MS.matmul eA eB)
                   (bid/(cols/bn) * bm + tid/(bn/tn) * tm + i)
                   (bid%(cols/bn) * bn + tid%(bn/tn) * tn + j))

(* Connects the post-bkIdx-loop state to the epilogue precondition.
   The loop invariant tracks __gms zero eA eB glob_r glob_c shared.
   The epilogue needs macc (ettile (matmul eA eB) ...) (idx/tn) (idx%tn).
   This lemma bridges them via __gms_full + lemma_matmul_index + ettile normalization. *)
let __post_loop_to_epilogue
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (bm : pos{bm /?+ rows})
  (bn : pos{bn /?+ cols})
  (tm : pos{tm /?+ bm})
  (tn : pos{tn /?+ bn})
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  (vrch_val : seq et)
  : Lemma
    (requires
      Seq.length vrch_val == tm * tn /\
      (forall (idx : natlt (tm * tn)).
        vrch_val @! idx == __gms (zero #et) eA eB
          (bid/(cols/bn) * bm + tm * (tid/(bn/tn)) + idx / tn)
          (bid%(cols/bn) * bn + tn * (tid%(bn/tn)) + idx % tn)
          shared))
    (ensures
      forall (idx : natlt (tm * tn)).
        vrch_val @! idx == macc (ettile (MS.matmul eA eB) bm bn tm tn bid tid) (idx / tn) (idx % tn))
  = let aux (idx : natlt (tm * tn))
      : Lemma (vrch_val @! idx == macc (ettile (MS.matmul eA eB) bm bn tm tn bid tid) (idx / tn) (idx % tn))
      = let i : natlt tm = idx / tn in
        let j : natlt tn = idx % tn in
        let br : natlt (rows/bm) = bid / (cols/bn) in
        let bc : natlt (cols/bn) = bid % (cols/bn) in
        let tr : natlt (bm/tm) = tid / (bn/tn) in
        let tc : natlt (bn/tn) = tid % (bn/tn) in
        assert (tm * tr + i < bm);
        assert (tn * tc + j < bn);
        assert (br * bm + (tm * tr + i) < rows);
        assert (bc * bn + (tn * tc + j) < cols);
        let glob_r : natlt rows = br * bm + tm * tr + i in
        let glob_c : natlt cols = bc * bn + tn * tc + j in
        __gms_full eA eB glob_r glob_c;
        MS.lemma_matmul_index eA eB glob_r glob_c;
        ettile_matmul_pointwise eA eB bm bn tm tn bid tid i j
    in
    Classical.forall_intro aux

let ettile_mmcomb_pointwise
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #cols : nat)
  (#shared : nat)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (eC : ematrix et rows cols)
  (bm : nat{bm > 0 /\ bm /?+ rows})
  (bn : nat{bn > 0 /\ bn /?+ cols})
  (tm : nat{tm > 0 /\ tm /?+ bm})
  (tn : nat{tn > 0 /\ tn /?+ bn})
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  (i : natlt tm) (j : natlt tn)
  : Lemma (macc (ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid) i j ==
           comb (macc (ettile eC bm bn tm tn bid tid) i j)
                (macc (ettile (MS.matmul eA eB) bm bn tm tn bid tid) i j))
          [SMTPat (macc (ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid) i j)]
  = assert_norm (macc (ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid) i j ==
                 comb (macc (ettile eC bm bn tm tn bid tid) i j)
                      (macc (ettile (MS.matmul eA eB) bm bn tm tn bid tid) i j))

#push-options "--z3rlimit 250" // Huge
inline_for_extraction noextract
fn epilogue
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #cols : sz)
  (#shared : sz)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (rchProd: larray et (tm * tn))
  (#vrch : erased (seq et))
  (#_ : squash (Seq.length vrch == tm * tn))
  (#lC : mlayout rows cols)
  {| clayout lC |}
  (gC : gpu_matrix et lC)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (eC : ematrix et rows cols)
  (bid : szlt (rows/bm * (cols/bn)))
  (tid : szlt (bm/tm * (bn/tn)))
  preserves
    rchProd |-> vrch
  requires
    pure (forall (idx : natlt (tm * tn)).
      vrch @! idx == macc (ettile (MS.matmul eA eB) bm bn tm tn bid tid) (idx / tn) (idx % tn)) **
    ttile gC bm bn tm tn bid tid |-> ettile eC bm bn tm tn bid tid
  ensures
    ttile gC bm bn tm tn bid tid |-> ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid
{
  (* Help the SMT connect vrch to the matmul subtile via div/mod *)
  assert pure (forall (i:natlt tm) (j:natlt tn).
    (i * tn + j) / tn == i);
  assert pure (forall (i:natlt tm) (j:natlt tn).
    (i * tn + j) % tn == j);
  assert pure (forall (i:natlt tm) (j:natlt tn).
    vrch @! (i * tn + j) == macc (ettile (MS.matmul eA eB) bm bn tm tn bid tid) i j);

  let t_tile = ttile gC (SZ.v bm) (SZ.v bn) (SZ.v tm) (SZ.v tn) (SZ.v bid) (SZ.v tid);
  assert (rewrites_to t_tile (ttile gC (SZ.v bm) (SZ.v bn) (SZ.v tm) (SZ.v tn) (SZ.v bid) (SZ.v tid)));

  let eC_tile = Ghost.hide (ettile eC (SZ.v bm) (SZ.v bn) (SZ.v tm) (SZ.v tn) (SZ.v bid) (SZ.v tid));

  let mut resIdxM = 0sz;
  while (!resIdxM <^ tm)
    invariant live resIdxM ** pure (!resIdxM <= tm)
    invariant exists* (m_cur : ematrix et tm tn).
      t_tile |-> m_cur **
      pure (forall (i:natlt tm) (j:natlt tn).
        macc m_cur i j ==
          (if i < !resIdxM
           then comb (macc eC_tile i j) (vrch @! (i * tn + j))
           else macc eC_tile i j))
    decreases (tm - !resIdxM)
  {
    let mut resIdxN = 0sz;
    while (!resIdxN <^ tn)
      invariant live resIdxN ** pure (!resIdxN <= tn)
      invariant exists* (m_cur : ematrix et tm tn).
        t_tile |-> m_cur **
        pure (forall (i:natlt tm) (j:natlt tn).
          macc m_cur i j ==
            (if i * tn + j < !resIdxM * tn + !resIdxN
             then comb (macc eC_tile i j) (vrch @! (i * tn + j))
             else macc eC_tile i j))
      decreases (tn - !resIdxN)
    {
      open Pulse.Lib.Array;
      pts_to_len rchProd;

      (* Combine the new result in the register cache to the value from gC and
      overwrite the the cell in gC *)
      let v0 = gpu_matrix_read t_tile !resIdxM !resIdxN;
      let v1 = rchProd.(!resIdxM *^ tn +^ !resIdxN);
      let v' = comb v0 v1;
      gpu_matrix_write t_tile !resIdxM !resIdxN v';

      (* Key arithmetic fact for the invariant step: for (i,j) in bounds,
         i*tn+j == resIdxM*tn+resIdxN iff i==resIdxM /\ j==resIdxN.
         This is needed so the SMT can connect mupd to the linearized
         index comparison in the invariant. *)
      assert pure (forall (i:natlt tm) (j:natlt tn).
        i * tn + j == !resIdxM * tn + !resIdxN <==> (i == !resIdxM /\ j == !resIdxN));

      resIdxN := !resIdxN +^ 1sz;
    };

    (* Bridge inner→outer: when resIdxN==tn, the linearized condition
       i*tn+j < resIdxM*tn+tn is equivalent to i <= resIdxM, and
       since j < tn, also to i < resIdxM+1. *)
    assert pure (forall (i:natlt tm) (j:natlt tn).
      i * tn + j < !resIdxM * tn + tn <==> i <= !resIdxM);

    resIdxM := !resIdxM +^ 1sz;
  };

  with m. assert gpu_matrix_pts_to t_tile m;

  assert pure (Kuiper.EMatrix.equal m (ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid));
  ()
}
#pop-options

#push-options "--fuel 1 --ifuel 1"
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et) str_B))
  (gA : gpu_matrix et lA)
  (#eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (#eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
  (#eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (#fA #fB : perm)
  (nthr : nat {nthr == bm/tm * (bn/tn)})
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : szlt (rows/bm * (cols/bn)))
  (tid : szlt (bm/tm * (bn/tn)))
  ()
  norewrite
  requires
    gpu **
    kpre comb gA eA gB eB gC eC bm bn bk slA slB tm tn fA fB nthr sh bid tid **
    thread_id (bm/tm * (bn/tn)) tid **
    block_id (rows/bm * (cols/bn)) bid **
    B.barrier_tok (FB.contract eA eB slA slB (fst sh) (fst (snd sh)) nthr bid) **
    B.barrier_state 0
  ensures
    gpu **
    kpost comb gA eA gB eB gC eC bm bn bk slA slB tm tn fA fB nthr sh bid tid **
    thread_id (bm/tm * (bn/tn)) tid **
    block_id (rows/bm * (cols/bn)) bid **
    B.barrier_tok (FB.contract eA eB slA slB (fst sh) (fst (snd sh)) nthr bid) **
    B.barrier_state (2 * (shared / bk))
{
  unfold_c_shmems sh (`%shmems_desc);
  let (sarA, (sarB, _)) = sh;

  gpu_pts_to_ref sarA;
  gpu_pts_to_ref sarB;
  gpu_matrix_pts_to_ref gA;
  gpu_matrix_pts_to_ref gB;

  gpu_matrix_abs' slA sarA;
  let sA = from_array slA sarA;
  rewrite each from_array slA sarA as sA;

  gpu_matrix_abs' slB sarB;
  let sB = from_array slB sarB;
  rewrite each from_array slB sarB as sB;

  let num_k_tiles = shared /^ bk;
  let num_n_tiles = cols /^ bn;
  let mrow = bid /^ num_n_tiles;
  let mcol = bid %^ num_n_tiles;

  let threadRow = tid /^ (bn/^tn);
  let threadCol = tid %^ (bn/^tn);

  (* register caches *)
  assert pure (tm <= rows);
  assert pure (tn <= cols);
  assert pure (tm * tn <= rows * cols);
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
        (exists* (x : ematrix _ _ _). FB.bp_sharing sA x nthr) **
        (exists* (x : ematrix _ _ _). FB.bp_sharing sB x nthr)
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

    copy_tiles_out_of_matrices_vec bm bn bk sA sB gA gB mrow !bkIdx mcol (bm/^tm*^(bn/^tn)) tid;

    assert own_strided_chunks sA (ematrix_subtile eA bm bk mrow !bkIdx) nthr tid;
    assert own_strided_chunks sB (ematrix_subtile eB bk bn !bkIdx mcol) nthr tid;

    odd_2x1 !bkIdx;
    assert (pure (odd (2 * !bkIdx + 1)));
    FB.fold_barrier_p_odd eA eB sA sB nthr bid mrow mcol !bkIdx tid;
    rewrite FB.barrier_p eA eB sA sB nthr bid (2 * !bkIdx + 1) tid
         as (FB.contract eA eB slA slB sarA sarB nthr bid).rin (2 * !bkIdx + 1) tid;

    B.barrier_wait ();

    even_2x (!bkIdx + 1);
    assert (pure (2 * (!bkIdx + 1) == 2 * !bkIdx + 2));
    assert (pure (even (2 * !bkIdx + 2)));
    assert (pure (odd (2 * !bkIdx + 1)));
    assert pure ((2 * !bkIdx + 1) < (2 * (shared /^ bk)));
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
     == __gms zero eA eB glob_r glob_c shared == matmul_single eA eB glob_r glob_c
     == macc (matmul eA eB) glob_r glob_c == macc (ettile (matmul eA eB) ...) (idx/tn) (idx%tn) *)
  with vrch_val. assert (rchProd |-> vrch_val);
  pts_to_len rchProd;
  assert pure (num_k_tiles * bk == shared);
  __post_loop_to_epilogue eA eB bm bn tm tn bid tid vrch_val;
  epilogue comb bm bn tm tn rchProd gC eA eB eC bid tid;

  gpu_matrix_concr sA; rewrite each core sA as sarA;
  gpu_matrix_concr sB; rewrite each core sB as sarB;

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
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
  (eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  // (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_: squash (SZ.fits (rows * cols)))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
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
  let n_total = rows/tm * (cols/tn);

  (* Step 1: Share gA/gB *)
  gpu_matrix_share_n gA n_total;
  gpu_matrix_share_n gB n_total;

  (* Step 2: Tile gC at block level *)
  gpu_matrix_tile gC (SZ.v bm) (SZ.v bn);
  forevery_rw_size2 (rows / bm) (SZ.v (rows /^ bm)) (cols / bn) (SZ.v (cols /^ bn));

  (* Step 3: For each block tile, tile at thread level and collapse to tid *)
  forevery_map_2
    (fun (br : natlt (SZ.v (rows /^ bm))) (bc : natlt (SZ.v (cols /^ bn))) ->
      gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) br bc |->
        Frac 1.0R (ematrix_subtile eC (SZ.v bm) (SZ.v bn) br bc))
    (fun (br : natlt (SZ.v (rows /^ bm))) (bc : natlt (SZ.v (cols /^ bn))) ->
      forall+ (tid : natlt (bm/tm * (bn/tn))).
        let tr = tid / (bn/tn) in
        let tc = tid % (bn/tn) in
        gpu_matrix_subtile (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc |->
          Frac 1.0R (ematrix_subtile (ematrix_subtile eC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc))
    fn br bc {
      gpu_matrix_tile (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn);
      forevery_unfactor' (bm/tm * (bn/tn)) (bm/tm) (bn/tn)
        (fun (tr : natlt (bm/tm)) (tc : natlt (bn/tn)) ->
          gpu_matrix_subtile (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc |->
            Frac 1.0R (ematrix_subtile (ematrix_subtile eC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc));
    };

  (* Step 4: Collapse (br, bc) → bid *)
  forevery_rw_size2 (SZ.v (rows /^ bm)) (rows / bm) (SZ.v (cols /^ bn)) (cols / bn);
  forevery_unfactor' (rows/bm * (cols/bn)) (rows/bm) (cols/bn)
    (fun (br : natlt (rows/bm)) (bc : natlt (cols/bn)) ->
      forall+ (tid : natlt (bm/tm * (bn/tn))).
        let tr = tid / (bn/tn) in
        let tc = tid % (bn/tn) in
        gpu_matrix_subtile (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc |->
          Frac 1.0R (ematrix_subtile (ematrix_subtile eC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc));

  (* Step 5: Factor gA/gB to 2D *)
  (* Divisibility chain: rows/tm == (rows/bm) * (bm/tm), cols/tn == (cols/bn) * (bn/tn) *)
  assert pure (tm * (bm/tm) * (rows/bm) == bm * (rows/bm));
  assert pure (tm * ((bm/tm) * (rows/bm)) == tm * (rows/tm));
  assert pure (rows/tm == (bm/tm) * (rows/bm));
  assert pure (tn * (bn/tn) * (cols/bn) == bn * (cols/bn));
  assert pure (tn * ((bn/tn) * (cols/bn)) == tn * (cols/tn));
  assert pure (cols/tn == (bn/tn) * (cols/bn));
  assert pure (n_total == (rows/bm * (cols/bn)) * (bm/tm * (bn/tn)));
  forevery_factor n_total (rows/bm * (cols/bn)) (bm/tm * (bn/tn))
    (fun _ -> gA |-> Frac (fA /. n_total) eA);
  forevery_factor n_total (rows/bm * (cols/bn)) (bm/tm * (bn/tn))
    (fun _ -> gB |-> Frac (fB /. n_total) eB);

  (* Step 6: Zip and fold kpre1 *)
  forevery_zip3_2
    (fun (_ : natlt (rows/bm * (cols/bn))) (_ : natlt (bm/tm * (bn/tn))) ->
      gA |-> Frac (fA /. n_total) eA)
    (fun (_ : natlt (rows/bm * (cols/bn))) (_ : natlt (bm/tm * (bn/tn))) ->
      gB |-> Frac (fB /. n_total) eB)
    (fun (bid : natlt (rows/bm * (cols/bn))) (tid : natlt (bm/tm * (bn/tn))) ->
      let br = bid / (cols/bn) in
      let bc = bid % (cols/bn) in
      let tr = tid / (bn/tn) in
      let tc = tid % (bn/tn) in
      gpu_matrix_subtile (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc |->
        Frac 1.0R (ematrix_subtile (ematrix_subtile eC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc));

  forevery_rw_size2 (rows/bm * (cols/bn)) (SZ.v nblk) (bm/tm * (bn/tn)) (SZ.v nthr);

  (* Step 7: Fold into kpre1 — introduce pure facts *)
  forevery_map_2
    (fun (bid : natlt nblk) (tid : natlt nthr) ->
      gA |-> Frac (fA /. n_total) eA **
      gB |-> Frac (fB /. n_total) eB **
      (let br = bid / (cols/bn) in
       let bc = bid % (cols/bn) in
       let tr = tid / (bn/tn) in
       let tc = tid % (bn/tn) in
       gpu_matrix_subtile (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc |->
         Frac 1.0R (ematrix_subtile (ematrix_subtile eC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc)))
    (fun (bid : natlt nblk) (tid : natlt nthr) ->
      kpre1 comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid)
    fn bid tid {
      assert pure (SZ.fits (rows * cols));
      assert pure (aligned 16 (core gA));
      assert pure (aligned 16 (core gB));
    };
}

ghost
fn block_setup
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et lA)
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
  (eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  // (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_: squash (SZ.fits (rows * cols)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
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
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et lA)
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
  (eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_: squash (SZ.fits (rows * cols)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
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
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lC |}
  (gA : gpu_matrix et lA)
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
  (eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_: squash (SZ.fits (rows * cols)))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
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
  let n_total = rows/tm * (cols/tn);
  let nblk_val = rows/bm * (cols/bn);
  let nthr_val = bm/tm * (bn/tn);

  (* Step 1: Collapse 2D → 1D (single forall+ in context, no ambiguity) *)
  forevery_rw_size2 (SZ.v nblk) nblk_val (SZ.v nthr) nthr_val;
  (* Divisibility chain: rows/tm == (rows/bm) * (bm/tm), cols/tn == (cols/bn) * (bn/tn) *)
  assert pure (tm * (bm/tm) * (rows/bm) == bm * (rows/bm));
  assert pure (tm * ((bm/tm) * (rows/bm)) == tm * (rows/tm));
  assert pure (rows/tm == (bm/tm) * (rows/bm));
  assert pure (tn * (bn/tn) * (cols/bn) == bn * (cols/bn));
  assert pure (tn * ((bn/tn) * (cols/bn)) == tn * (cols/tn));
  assert pure (cols/tn == (bn/tn) * (cols/bn));
  assert pure (n_total == nblk_val * nthr_val);
  forevery_unfactor' (rows/tm * (cols/tn)) nblk_val nthr_val
    (fun (bid : natlt nblk_val) (tid : natlt nthr_val) ->
      gA |-> Frac (fA /. (rows/tm * (cols/tn))) eA **
      gB |-> Frac (fB /. (rows/tm * (cols/tn))) eB **
      ttile gC bm bn tm tn bid tid |-> ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid);

  (* Step 2: Separate and gather gA/gB *)
  forevery_unzip3
    (fun (_ : natlt (rows/tm * (cols/tn))) -> gA |-> Frac (fA /. (rows/tm * (cols/tn))) eA)
    (fun (_ : natlt (rows/tm * (cols/tn))) -> gB |-> Frac (fB /. (rows/tm * (cols/tn))) eB)
    (fun (k : natlt (rows/tm * (cols/tn))) ->
      ttile gC bm bn tm tn (k / nthr_val) (k % nthr_val) |->
        ettile (MS.mmcomb comb eC eA eB) bm bn tm tn (k / nthr_val) (k % nthr_val));
  gpu_matrix_gather_n gA (rows/tm * (cols/tn));
  gpu_matrix_gather_n gB (rows/tm * (cols/tn));

  (* Step 3: Factor' gC: 1D → 2D (bid, tid) *)
  forevery_factor' (rows/tm * (cols/tn)) nblk_val nthr_val
    (fun (bid : natlt nblk_val) (tid : natlt nthr_val) ->
      ttile gC bm bn tm tn bid tid |-> ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid);

  (* Step 4: Convert ttile/ettile to explicit subtile form — Pulse tactic can unfold these *)
  forevery_ext_2
    (fun (bid : natlt nblk_val) (tid : natlt nthr_val) ->
      ttile gC bm bn tm tn bid tid |-> ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid)
    (fun (bid : natlt nblk_val) (tid : natlt nthr_val) ->
      let br = bid / (cols/bn) in
      let bc = bid % (cols/bn) in
      let tr = tid / (bn/tn) in
      let tc = tid % (bn/tn) in
      gpu_matrix_subtile (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc |->
        Frac 1.0R (ematrix_subtile (ematrix_subtile (MS.mmcomb comb eC eA eB) (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc));

  (* Step 5: Factor' bid → (br, bc) — now the body uses explicit div/mod *)
  forevery_factor' nblk_val (rows/bm) (cols/bn)
    (fun (br : natlt (rows/bm)) (bc : natlt (cols/bn)) ->
      forall+ (tid : natlt nthr_val).
        let tr = tid / (bn/tn) in
        let tc = tid % (bn/tn) in
        gpu_matrix_subtile (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc |->
          Frac 1.0R (ematrix_subtile (ematrix_subtile (MS.mmcomb comb eC eA eB) (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc));

  (* Step 6: Per block, factor' tid → (tr, tc) and untile *)
  forevery_map_2
    (fun (br : natlt (rows/bm)) (bc : natlt (cols/bn)) ->
      forall+ (tid : natlt nthr_val).
        let tr = tid / (bn/tn) in
        let tc = tid % (bn/tn) in
        gpu_matrix_subtile (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc |->
          Frac 1.0R (ematrix_subtile (ematrix_subtile (MS.mmcomb comb eC eA eB) (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc))
    (fun (br : natlt (rows/bm)) (bc : natlt (cols/bn)) ->
      gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) br bc |->
        Frac 1.0R (ematrix_subtile (MS.mmcomb comb eC eA eB) (SZ.v bm) (SZ.v bn) br bc))
    fn br bc {
      forevery_factor' nthr_val (bm/tm) (bn/tn)
        (fun (tr : natlt (bm/tm)) (tc : natlt (bn/tn)) ->
          gpu_matrix_subtile (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc |->
            Frac 1.0R (ematrix_subtile (ematrix_subtile (MS.mmcomb comb eC eA eB) (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn) tr tc));
      assert pure (SZ.fits (mlayout_size (subtile_layout lC (SZ.v bm) (SZ.v bn) br bc)));
      gpu_matrix_untile (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) br bc) (SZ.v tm) (SZ.v tn);
    };

  (* Step 7: Untile block tiles *)
  assert pure (SZ.fits (mlayout_size lC));
  gpu_matrix_untile gC (SZ.v bm) (SZ.v bn);
}

#push-options "--z3rlimit_factor 4 --split_queries no --fuel 1 --ifuel 1"
#restart-solver
inline_for_extraction noextract
let mk_kernel
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et) str_B))
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (#fA : perm)
  (#eA : ematrix et rows shared)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (#fB : perm)
  (#eB : ematrix et shared cols)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (#eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/tm * (bn/tn)})
  (#_ : squash (chunk et * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (rows/bm * (cols/bn) <= max_blocks
               /\ (bm/tm * (bn/tn)) <= max_threads))
  (#_ : squash (aligned 16 (core gA) /\ aligned 16 (core gB)))
  ()
  : kernel_desc
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.mmcomb comb eC eA eB)
= {
  nblk;// = rows/^bm *^ (cols/^bn);
  nthr;// = (bm /^ tm *^ (bn /^ tn));

  shmems_desc = shmems_desc et bm bn bk;

  barrier_contract = (fun bid ptrs -> FB.contract eA eB slA slB (fst ptrs) (fst (snd ptrs)) nthr bid);
  barrier_count    = (fun _bid -> 2 * (SZ.v shared / SZ.v bk));
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
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et) str_B))
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (#eA : ematrix et rows shared)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (#eB : ematrix et shared cols)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (#eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (rows/bm * (cols/bn) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.mmcomb comb eC eA eB)
{
  (* fixed the inner layouts, or we'd have to propagate this everywhere? *)
  launch_sync (mk_kernel comb gA gB gC bm bn bk slA slB tm tn (rows/^bm *^ (cols/^bn)) (bm/^tm *^ (bn/^tn)) ());
}

inline_for_extraction noextract
fn mmcomb_gpu_approx
  (#et : Type0) {| scalar et, has_vec_cpy et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et) str_B))
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (#eA : ematrix et rows shared)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (#eB : ematrix et shared cols)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (#eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (#fA #fB : perm)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (rows/bm * (cols/bn) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    exists* (eC' : ematrix et rows cols).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB)
{
  mmcomb_gpu_exact #et #_ #_ comb #rows #shared #cols #lA #lB #lC gA #eA gB #eB gC #eC bm bn bk tm tn slA slB #_ #_;
  MU.mmcomb_approx_real comb comb_r eC eA eB rA rB rC;
  ()
}
