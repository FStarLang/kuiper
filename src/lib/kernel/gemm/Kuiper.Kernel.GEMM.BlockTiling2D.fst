module Kuiper.Kernel.GEMM.BlockTiling2D

#lang-pulse

#set-options "--z3rlimit 60"

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix { chest2 }
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Array2.Strided
open Kuiper.Array2.Strided.Slice
open Kuiper.Tensor.Tiling
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Slice
open Kuiper.Chest
open Kuiper.Bijection
open Kuiper.Seq.Common { op_At_Bang }

module B = Kuiper.Barrier
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module FB = Kuiper.Kernel.GEMM.FlipFlopBarrier2
module T = Kuiper.Tensor
module CV2 = Kuiper.Kernel.GEMM.Copy.Vec2
module Trade = Pulse.Lib.Trade
module MU = Kuiper.Kernel.GEMM.Util
module Chest = Kuiper.Chest
module C = Kuiper.Matrix.Casts

(* Shared memory description for tiled matmul kernels.  Tile A keeps the input
   element type [ta]; tile B keeps [tb] (the vectorized global->shared copy is
   type preserving, so the fused maps are applied at compute time instead). *)
inline_for_extraction noextract
let shmems_desc
  (ta tb : Type0) {| sized ta, sized tb |}
  (bm bn bk: szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  : list shmem_desc = [
  SHArray ta (bm *^ bk);
  SHArray tb (bk *^ bn);
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


(* Fused pre-map applied at compute time: multiply the mapped inputs.  This is
   the element-level product that [Kuiper.Spec.GEMM.gmmcomb] uses (via
   [chest_map mapA]/[chest_map mapB]). *)
inline_for_extraction noextract
let __mulm
  (#ta #tb #tacc : Type0) {| scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (a : ta) (b : tb)
  : tacc
  = mul (mapA a) (mapB b)

(* Wrapper for __gmatmul_single that doesn't require refined row/col/to.
   Returns the initial value when arguments are out of bounds.  The two input
   tiles keep their raw types [ta]/[tb]; the pre-maps [mapA]/[mapB] are applied
   to each element inside the fused multiply [__mulm], accumulating in [tacc]. *)
let __gms
  (#ta #tb #tacc : Type0) {| scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (#m #k #columns : nat)
  (z : tacc)
  (m1 : chest2 ta m k)
  (m2 : chest2 tb k columns)
  (row : nat) (col : nat) (to : nat)
  : GTot tacc
  = if row < m && col < columns && to <= k
    then MS.__gmatmul_single z (__mulm mapA mapB) add m1 m2 row col to
    else z

(* Step lemma in the "forward" direction: given a partial sum and one more
   product term, the result is __gms at (d+1). Uses an SMTPat so the SMT
   can apply this automatically inside universal quantifiers. *)
let __gms_fwd
  (#ta #tb #tacc : Type0) {| scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (#m #k #columns : nat)
  (z : tacc)
  (m1 : chest2 ta m k)
  (m2 : chest2 tb k columns)
  (row : nat) (col : nat) (d : nat)
  : Lemma
    (requires row < m /\ col < columns /\ d < k)
    (ensures add (__gms mapA mapB z m1 m2 row col d)
                 (__mulm mapA mapB (acc2 m1 row d) (acc2 m2 d col))
             == __gms mapA mapB z m1 m2 row col (d + 1))
    [SMTPat (__gms mapA mapB z m1 m2 row col (d + 1))]
  = MS.__gmatmul_single_lemma z (__mulm mapA mapB) add m1 m2 row col (d + 1)

(* Tiled accumulation step: computing __gms on a subtile with the previous
   accumulation equals advancing the full accumulation by d more elements. *)
let rec __gms_tiled_step
  (#ta #tb #tacc : Type0) {| scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (#m #n #k : nat)
  (eA : chest2 ta m k)
  (eB : chest2 tb k n)
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
      __gms mapA mapB (__gms mapA mapB (zero #tacc) eA eB glob_r glob_c (bkIdx * bk))
            (ematrix_subtile eA bm bk mrow bkIdx)
            (ematrix_subtile eB bk bn bkIdx mcol)
            local_r local_c d
      == __gms mapA mapB (zero #tacc) eA eB glob_r glob_c (bkIdx * bk + d)))
    (decreases d)
  = if d = 0 then ()
    else (
      __gms_tiled_step mapA mapB eA eB bm bn bk mrow mcol local_r local_c bkIdx (d - 1);
      MS.__gmatmul_single_lemma
        (__gms mapA mapB (zero #tacc) eA eB (mrow * bm + local_r) (mcol * bn + local_c) (bkIdx * bk))
        (__mulm mapA mapB) add
        (ematrix_subtile eA bm bk mrow bkIdx)
        (ematrix_subtile eB bk bn bkIdx mcol)
        local_r local_c d;
      MS.__gmatmul_single_lemma
        (zero #tacc) (__mulm mapA mapB) add eA eB
        (mrow * bm + local_r) (mcol * bn + local_c)
        (bkIdx * bk + d)
    )

(* Non-recursive wrapper for d=bk with SMTPat: the full tile step. *)
let __gms_tile_full
  (#ta #tb #tacc : Type0) {| scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (#m #n #k : nat)
  (eA : chest2 ta m k)
  (eB : chest2 tb k n)
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
      __gms mapA mapB (__gms mapA mapB (zero #tacc) eA eB glob_r glob_c (bkIdx * bk))
            (ematrix_subtile eA bm bk mrow bkIdx)
            (ematrix_subtile eB bk bn bkIdx mcol)
            local_r local_c bk
      == __gms mapA mapB (zero #tacc) eA eB glob_r glob_c ((bkIdx + 1) * bk)))
    [SMTPat (__gms mapA mapB (__gms mapA mapB (zero #tacc) eA eB (mrow * bm + local_r) (mcol * bn + local_c) (bkIdx * bk))
                   (ematrix_subtile eA bm bk mrow bkIdx)
                   (ematrix_subtile eB bk bn bkIdx mcol)
                   local_r local_c bk)]
  = __gms_tiled_step mapA mapB eA eB bm bn bk mrow mcol local_r local_c bkIdx bk;
    assert (bkIdx * bk + bk == (bkIdx + 1) * bk)

(* Congruence: the fused pre-map accumulation over raw tiles equals the
   ordinary accumulation over the [chest_map]-ped tiles.  Proven by induction
   on [to], reducing [__mulm mapA mapB (acc2 eA ..) (acc2 eB ..)] to
   [mul (acc2 (chest_map mapA eA) ..) (acc2 (chest_map mapB eB) ..)] via the
   [acc_pat] SMTPat on [chest_map]. *)
let rec __gms_full_aux
  (#ta #tb #tacc : Type0) {| scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (#m #k #columns : nat)
  (eA : chest2 ta m k)
  (eB : chest2 tb k columns)
  (row : natlt m) (col : natlt columns) (to : nat{to <= k})
  : Lemma
    (ensures
      MS.__gmatmul_single (zero #tacc) (__mulm mapA mapB) add eA eB row col to
      == MS.__gmatmul_single (zero #tacc) mul add
           (Kuiper.Chest.chest_map mapA eA) (Kuiper.Chest.chest_map mapB eB) row col to)
    (decreases to)
  = if to = 0 then ()
    else (
      __gms_full_aux mapA mapB eA eB row col (to - 1);
      MS.__gmatmul_single_lemma (zero #tacc) (__mulm mapA mapB) add eA eB row col to;
      MS.__gmatmul_single_lemma (zero #tacc) mul add
        (Kuiper.Chest.chest_map mapA eA) (Kuiper.Chest.chest_map mapB eB) row col to
    )

(* Full accumulation: __gms with zero initial value over the entire shared
   dimension equals matmul_single of the mapped tiles. *)
let __gms_full
  (#ta #tb #tacc : Type0) {| scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (#m #n #k : nat)
  (eA : chest2 ta m k)
  (eB : chest2 tb k n)
  (row : natlt m) (col : natlt n)
  : Lemma (__gms mapA mapB (zero #tacc) eA eB row col k
           == MS.matmul_single (Kuiper.Chest.chest_map mapA eA) (Kuiper.Chest.chest_map mapB eB) row col)
          [SMTPat (__gms mapA mapB (zero #tacc) eA eB row col k)]
  = __gms_full_aux mapA mapB eA eB row col k

(* Helper for the bkIdx loop body: given the old invariant (rchProd tracks
   partial accumulations up to bkIdx*bk) and the subproducts2d postcondition
   (one more tile of bk columns accumulated), derive the new invariant
   (accumulation up to (bkIdx+1)*bk). *)
let __bkIdx_loop_step
  (#ta #tb #tacc : Type0) {| scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (#m #n #k : nat)
  (eA : chest2 ta m k)
  (eB : chest2 tb k n)
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
  (old_v : seq tacc)
  : Lemma
    (requires
      Seq.length old_v == tm * tn /\
      (forall (idx : natlt (tm * tn)).
        old_v @! idx == __gms mapA mapB (zero #tacc) eA eB
          (mrow * bm + tm * threadRow + idx / tn)
          (mcol * bn + tn * threadCol + idx % tn)
          (bkIdx * bk)))
    (ensures (
      let new_v = Seq.init_ghost (tm * tn) (fun idx ->
        __gms mapA mapB (old_v @! idx)
          (ematrix_subtile eA bm bk mrow bkIdx)
          (ematrix_subtile eB bk bn bkIdx mcol)
          (tm * threadRow + idx / tn)
          (tn * threadCol + idx % tn)
          bk) in
      Seq.length new_v == tm * tn /\
      (forall (idx : natlt (tm * tn)).
        new_v @! idx == __gms mapA mapB (zero #tacc) eA eB
          (mrow * bm + tm * threadRow + idx / tn)
          (mcol * bn + tn * threadCol + idx % tn)
          ((bkIdx + 1) * bk))))
  = let new_v = Seq.init_ghost (tm * tn) (fun idx ->
      __gms mapA mapB (old_v @! idx)
        (ematrix_subtile eA bm bk mrow bkIdx)
        (ematrix_subtile eB bk bn bkIdx mcol)
        (tm * threadRow + idx / tn)
        (tn * threadCol + idx % tn)
        bk) in
    let aux (idx : natlt (tm * tn))
      : Lemma (new_v @! idx == __gms mapA mapB (zero #tacc) eA eB
                (mrow * bm + tm * threadRow + idx / tn)
                (mcol * bn + tn * threadCol + idx % tn)
                ((bkIdx + 1) * bk))
      = let local_r : natlt bm = tm * threadRow + idx / tn in
        let local_c : natlt bn = tn * threadCol + idx % tn in
        __gms_tiled_step mapA mapB eA eB bm bn bk mrow mcol local_r local_c bkIdx bk;
        assert (bkIdx * bk + bk == (bkIdx + 1) * bk)
    in
    FStar.Classical.forall_intro aux

inline_for_extraction noextract
fn subproducts2d
  (#ta #tb #tacc : Type0) {| scalar ta, scalar tb, scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (bm bn bk: szp)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (rchProd: larray tacc (tm * tn))
  (#vrchProd : erased (seq tacc))
  (#l1 : layout2 bm bk) {| T.ctlayout l1 |}
  (#l2 : layout2 bk bn) {| T.ctlayout l2 |}
  (gA : array2 ta l1)
  (gB : array2 tb l2)
  (#eA : chest2 ta bm bk)
  (#eB : chest2 tb bk bn)
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
        __gms mapA mapB (vrchProd @! idx) eA eB (tm * arow + i) (tn * bcol + j) bk)
{
  open Pulse.Lib.Array;

  let mut dotIdx : sz = 0sz;
  while (!dotIdx <^ bk)
    invariant live dotIdx ** pure (!dotIdx <= bk)
    invariant exists* (v_d : seq tacc).
      rchProd |-> v_d **
      pure (len v_d == tm * tn /\
            (forall (idx : natlt (tm * tn)).
              v_d @! idx ==
                __gms mapA mapB (vrchProd @! idx) eA eB
                  (tm * arow + idx / tn) (tn * bcol + idx % tn) !dotIdx))
    decreases (bk - !dotIdx)
  {
    (* register caches *)
    let mut rAcol : Pulse.Lib.Array.array ta = [| zero #ta #_ ; tm |];
    let mut rBrow : Pulse.Lib.Array.array tb = [| zero #tb #_ ; tn |];

    let mut j0 = 0sz;
    while (!j0 <^ tm)
      invariant exists* (vj0 : sz{SZ.v vj0 <= tm}). j0 |-> vj0
      invariant exists* (vrAcol : lseq ta tm).
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
      invariant exists* (vrBrow : lseq tb tn).
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
      invariant exists* (v_m : seq tacc).
        rchProd |-> v_m **
        pure (len v_m == tm * tn /\
              (forall (idx : natlt (tm * tn)).
                v_m @! idx ==
                  (if idx < !resIdxM * tn
                   then add (v_cur @! idx) (__mulm mapA mapB (vrAcol @! (idx / tn)) (vrBrow @! (idx % tn)))
                   else v_cur @! idx)))
      decreases (tm - !resIdxM)
    {
      let mut resIdxN = 0sz;
      while (!resIdxN <^ tn)
        invariant live resIdxN ** pure (!resIdxN <= tn)
        invariant exists* (v_n : seq tacc).
          rchProd |-> v_n **
          pure (len v_n == tm * tn /\
                (forall (idx : natlt (tm * tn)).
                  v_n @! idx ==
                    (if idx < !resIdxM * tn + !resIdxN
                     then add (v_cur @! idx) (__mulm mapA mapB (vrAcol @! (idx / tn)) (vrBrow @! (idx % tn)))
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
        let mad = old `add` ((mapA ra) `mul` (mapB rb));
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
      __gms mapA mapB (vrchProd @! idx) eA eB (tm * arow + i) (tn * bcol + j) bk)));
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
  (#ta #tb #tacc : Type0) {| scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (#m #n #k : nat)
  (eA : chest2 ta m k)
  (eB : chest2 tb k n)
  (bm : pos{bm /?+ m})
  (bn : pos{bn /?+ n})
  (tm : pos{tm /?+ bm})
  (tn : pos{tn /?+ bn})
  (bid : natlt (m/bm * (n/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  (vrch_val : seq tacc)
  : Lemma
    (requires
      Seq.length vrch_val == tm * tn /\
      (forall (idx : natlt (tm * tn)).
        vrch_val @! idx == __gms mapA mapB (zero #tacc) eA eB
          (bid/(n/bn) * bm + tm * (tid/(bn/tn)) + idx / tn)
          (bid%(n/bn) * bn + tn * (tid%(bn/tn)) + idx % tn)
          k))
    (ensures
      forall (idx : natlt (tm * tn)).
        vrch_val @! idx == acc2 (ettile (MS.matmul (Kuiper.Chest.chest_map mapA eA) (Kuiper.Chest.chest_map mapB eB)) bm bn tm tn bid tid) (idx / tn) (idx % tn))
  = let mA = Kuiper.Chest.chest_map mapA eA in
    let mB = Kuiper.Chest.chest_map mapB eB in
    let aux (idx : natlt (tm * tn))
      : Lemma (vrch_val @! idx == acc2 (ettile (MS.matmul mA mB) bm bn tm tn bid tid) (idx / tn) (idx % tn))
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
        __gms_full mapA mapB eA eB glob_r glob_c;
        MS.lemma_matmul_index mA mB glob_r glob_c;
        ettile_matmul_pointwise mA mB bm bn tm tn bid tid i j
    in
    Classical.forall_intro aux

let ettile_mmcomb_pointwise
  (#ta #tb #tc #tacc : Type0) {| scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#m #n : nat)
  (#k : nat)
  (eA : chest2 ta m k)
  (eB : chest2 tb k n)
  (eC : chest2 tc m n)
  (bm : nat{bm > 0 /\ bm /?+ m})
  (bn : nat{bn > 0 /\ bn /?+ n})
  (tm : nat{tm > 0 /\ tm /?+ bm})
  (tn : nat{tn > 0 /\ tn /?+ bn})
  (bid : natlt (m/bm * (n/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  (i : natlt tm) (j : natlt tn)
  : Lemma (acc2 (ettile (MS.gmmcomb mapA mapB comb eC eA eB) bm bn tm tn bid tid) i j ==
           comb (acc2 (ettile eC bm bn tm tn bid tid) i j)
                (acc2 (ettile (MS.matmul (Kuiper.Chest.chest_map mapA eA) (Kuiper.Chest.chest_map mapB eB)) bm bn tm tn bid tid) i j))
          [SMTPat (acc2 (ettile (MS.gmmcomb mapA mapB comb eC eA eB) bm bn tm tn bid tid) i j)]
  = let ro : nat = (bid/(n/bn))*bm + ((tid/(bn/tn))*tm + i) in
    let co : nat = (bid%(n/bn))*bn + ((tid%(bn/tn))*tn + j) in
    MS.lemma_matmul_index (Kuiper.Chest.chest_map mapA eA) (Kuiper.Chest.chest_map mapB eB) ro co;
    assert_norm (acc2 (ettile (MS.gmmcomb mapA mapB comb eC eA eB) bm bn tm tn bid tid) i j ==
                 comb (acc2 (ettile eC bm bn tm tn bid tid) i j)
                      (MS.matmul_single (Kuiper.Chest.chest_map mapA eA) (Kuiper.Chest.chest_map mapB eB) ro co));
    assert_norm (acc2 (ettile (MS.matmul (Kuiper.Chest.chest_map mapA eA) (Kuiper.Chest.chest_map mapB eB)) bm bn tm tn bid tid) i j ==
                 acc2 (MS.matmul (Kuiper.Chest.chest_map mapA eA) (Kuiper.Chest.chest_map mapB eB)) ro co)

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
  (#ta #tb #tc #tacc : Type0) {| scalar tc, scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#m #n : sz)
  (#k : sz)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (rchProd: larray tacc (tm * tn))
  (#vrch : erased (seq tacc))
  (#_ : squash (Seq.length vrch == tm * tn))
  (#lC : layout2 m n)
  {| T.ctlayout lC |}
  (gC : array2 tc lC)
  (eA : chest2 ta m k)
  (eB : chest2 tb k n)
  (eC : chest2 tc m n)
  (bid : szlt (m/bm * (n/bn)))
  (tid : szlt (bm/tm * (bn/tn)))
  preserves
    rchProd |-> vrch
  requires
    pure (forall (idx : natlt (tm * tn)).
      vrch @! idx == acc2 (ettile (MS.matmul (Kuiper.Chest.chest_map mapA eA) (Kuiper.Chest.chest_map mapB eB)) bm bn tm tn bid tid) (idx / tn) (idx % tn)) **
    ttile gC bm bn tm tn bid tid |-> ettile eC bm bn tm tn bid tid
  ensures
    ttile gC bm bn tm tn bid tid |-> ettile (MS.gmmcomb mapA mapB comb eC eA eB) bm bn tm tn bid tid
{
  (* Help the SMT connect vrch to the matmul subtile via div/mod *)
  epilogue_tile_div_mod tm tn;
  assert pure (forall (i:natlt tm) (j:natlt tn).
    vrch @! (i * tn + j) == acc2 (ettile (MS.matmul (Kuiper.Chest.chest_map mapA eA) (Kuiper.Chest.chest_map mapB eB)) bm bn tm tn bid tid) i j);

  let t_tile = ttile gC (SZ.v bm) (SZ.v bn) (SZ.v tm) (SZ.v tn) (SZ.v bid) (SZ.v tid);
  assert (rewrites_to t_tile (ttile gC (SZ.v bm) (SZ.v bn) (SZ.v tm) (SZ.v tn) (SZ.v bid) (SZ.v tid)));

  let eC_tile = Ghost.hide (ettile eC (SZ.v bm) (SZ.v bn) (SZ.v tm) (SZ.v tn) (SZ.v bid) (SZ.v tid));

  let mut resIdxM = 0sz;
  while (!resIdxM <^ tm)
    invariant live resIdxM ** pure (!resIdxM <= tm)
    invariant exists* (m_cur : chest2 tc tm tn).
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
      invariant exists* (m_cur : chest2 tc tm tn).
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

  assert pure (Kuiper.Chest.equal m (ettile (MS.gmmcomb mapA mapB comb eC eA eB) bm bn tm tn bid tid));
  ()
}
#pop-options
(* C-independent compute core shared by the rank-2 [kf] and the batched [bkf].
   Runs the flip-flop barrier / vectorized shared-copy / [subproducts2d] loop,
   accumulating the per-thread tm*tn products into the caller-provided local
   buffer [rchProd].  Touches neither the output C nor [comb]; the caller writes
   C (subtile for [kf], cells for [bkf]).  The post-loop [rchProd] equals the
   exact matmul subtile of the [chest_map]-ped inputs. *)
#push-options "--fuel 1 --ifuel 1"
inline_for_extraction noextract
fn kf_compute
  (#ta #tb #tacc : Type0)
  {| scalar ta, scalar tb, scalar tacc, has_vec_cpy ta, has_vec_cpy tb |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  {| T.ctlayout lA, T.ctlayout lB |}
  {| str_A : strided_row_major lA, str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk ta) str_A))
  (#_ : squash (aligned_strided_row_major (chunk tb) str_B))
  (gA : array2 ta lA)
  (#eA : chest2 ta m k)
  (gB : array2 tb lB)
  (#eB : chest2 tb k n)
  (#fAe #fBe : perm)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk tb /?+ bn))
  (#_ : squash (chunk ta /?+ bk))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (chunk ta * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk tb * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (nthr : nat {nthr == bm/tm * (bn/tn)})
  (sh : c_shmems (shmems_desc ta tb bm bn bk))
  (bid : szlt (m/bm * (n/bn)))
  (tid : szlt (bm/tm * (bn/tn)))
  (rchProd : larray tacc (tm * tn))
  (#vrch0 : erased (seq tacc))
  ()
  norewrite
  requires
    gpu **
    gA |-> Frac fAe eA **
    gB |-> Frac fBe eB **
    live_c_shmems sh #(1.0R /. (bm/tm * (bn/tn))) **
    (rchProd |-> vrch0) **
    pure (Seq.length vrch0 == tm * tn /\ (forall (idx : natlt (tm * tn)). vrch0 @! idx == zero #tacc)) **
    pure (SZ.fits (m * n)) **
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    thread_id (bm/tm * (bn/tn)) tid **
    B.barrier_tok (FB.contract eA eB slA slB (fst sh) (fst (snd sh)) nthr bid) **
    B.barrier_state 0
  ensures
    gpu **
    gA |-> Frac fAe eA **
    gB |-> Frac fBe eB **
    live_c_shmems sh #(1.0R /. (bm/tm * (bn/tn))) **
    (exists* (vrch : seq tacc).
      (rchProd |-> vrch) **
      pure (Seq.length vrch == tm * tn /\
        (forall (idx : natlt (tm * tn)).
          vrch @! idx ==
            acc2 (ettile (MS.matmul (Kuiper.Chest.chest_map mapA eA) (Kuiper.Chest.chest_map mapB eB)) bm bn tm tn bid tid) (idx / tn) (idx % tn)))) **
    thread_id (bm/tm * (bn/tn)) tid **
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

  with x. fold FB.bp_sharing sA x nthr;
  with x. fold FB.bp_sharing sB x nthr;

  let mut bkIdx  : sz = 0sz;
  while (!bkIdx <^ num_k_tiles)
    invariant live bkIdx ** pure (!bkIdx <= num_k_tiles)
    invariant exists* (v : seq tacc).
      rchProd |-> v **
      pure (len v == tm * tn /\
        forall (idx : natlt (tm * tn)).
          v @! idx == __gms mapA mapB (zero #tacc) eA eB
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

      Kuiper.Divides.lemma_divides_product_l (chunk ta) str_A.stride (mrow * bm);
      Kuiper.Divides.lemma_divides_product_r (chunk ta) !bkIdx bk;
      Kuiper.Divides.lemma_divides_sum (chunk ta) str_A.offset (str_A.stride * (mrow * bm));
      Kuiper.Divides.lemma_divides_sum (chunk ta) (str_A.offset + str_A.stride * (mrow * bm)) (!bkIdx * bk);
      assert pure (chunk ta /?+ (str_A.offset + str_A.stride * (mrow * bm) + (!bkIdx * bk)));

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

      Kuiper.Divides.lemma_divides_product_l (chunk tb) str_B.stride (!bkIdx * bk);
      Kuiper.Divides.lemma_divides_product_r (chunk tb) mcol bn;
      Kuiper.Divides.lemma_divides_sum (chunk tb) str_B.offset (str_B.stride * (!bkIdx * bk));
      Kuiper.Divides.lemma_divides_sum (chunk tb) (str_B.offset + str_B.stride * (!bkIdx * bk)) (mcol * bn);
      assert pure (chunk tb /?+ (str_B.offset + str_B.stride * (!bkIdx * bk) + (mcol * bn)));

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
    subproducts2d mapA mapB bm bn bk tm tn rchProd sA sB threadRow threadCol;
    __bkIdx_loop_step mapA mapB eA eB bm bn bk tm tn mrow mcol threadRow threadCol !bkIdx old_v;

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

  __post_loop_to_epilogue mapA mapB eA eB bm bn tm tn bid tid vrch_val;

  tensor_concr sA; rewrite each core sA as sarA;
  tensor_concr sB; rewrite each core sB as sarB;

  rewrite each sarA as fst sh;
  rewrite each sarB as fst (snd sh);

  fold_c_shmems sh #(1.0R /. (bm/tm * (bn/tn))) (`%shmems_desc);
  ()
}
#pop-options



(* ─── batched index bijection machinery (ported from BT2DBijScratch) ─── *)
(* size-equality bijection on natlt *)
let bij_seq (a b : nat) (_ : squash (a == b)) : (natlt a =~ natlt b) =
  {
    ff = (fun (x : natlt a) -> (x <: natlt b));
    gg = (fun (x : natlt b) -> (x <: natlt a));
    ff_gg = (fun _ -> ());
    gg_ff = (fun _ -> ());
  }

(* factor a dimension d == d1 * (d2 * d3) into three nested factors *)
let dim_split3 (d d1 d2 d3 : nat) (_ : squash (d == d1 * (d2 * d3)))
  : (natlt d =~ (natlt d1 & (natlt d2 & natlt d3)))
  = bij_comp (bij_seq d (d1 * (d2 * d3)) ())
      (bij_comp (bij_sym (bij_nat_prod #d1 #(d2 * d3)))
                (bij_prod (bij_self (natlt d1)) (bij_sym (bij_nat_prod #d2 #d3))))

(* shuffle the abstract tensor index to (rows, cols), batch *)
let bt2d_abs_shuffle (batch m n : nat)
  : (abs (batch @| m @| n @| INil) =~ ((natlt m & natlt n) & natlt batch))
  = {
      ff = (fun (pg, (r, (c, ()))) -> ((r, c), pg));
      gg = (fun ((r, c), pg) -> (pg, (r, (c, ()))));
      ff_gg = (fun ((r, c), pg) -> ());
      gg_ff = (fun (pg, (r, (c, ()))) -> ());
    }

(* regroup a 7-way nested tuple
     ((nbr,(ntr,tm)) & (nbc,(ntc,tn))) & batch
   into
     ((nbr,nbc), batch) & ((ntr,ntc) & (tm,tn))                    *)
let bt2d_regroup
  (#nbr #ntr #tm #nbc #ntc #tn #batch : Type)
  : ((((nbr & (ntr & tm)) & (nbc & (ntc & tn))) & batch)
     =~ (((nbr & nbc) & batch) & ((ntr & ntc) & (tm & tn))))
  = {
      ff = (fun (((br, (tr, i)), (bc, (tc, j))), pg) ->
              (((br, bc), pg), ((tr, tc), (i, j))));
      gg = (fun (((br, bc), pg), ((tr, tc), (i, j))) ->
              (((br, (tr, i)), (bc, (tc, j))), pg));
      ff_gg = (fun (((br, bc), pg), ((tr, tc), (i, j))) -> ());
      gg_ff = (fun (((br, (tr, i)), (bc, (tc, j))), pg) -> ());
    }

(* flatten ((nbr & nbc) & batch) to a page-minor block index natlt(batch*(nbr*nbc)) *)
let bt2d_block_flat (nbr nbc batch : nat)
  : (((natlt nbr & natlt nbc) & natlt batch) =~ natlt (batch * (nbr * nbc)))
  = bij_comp
      (bij_prod (bij_nat_prod #nbr #nbc) (bij_self (natlt batch)))
      (bij_comp (bij_nat_prod #(nbr * nbc) #batch)
                (bij_seq ((nbr * nbc) * batch) (batch * (nbr * nbc)) ()))

(* the full batched index bijection *)
let bt2d_idx_bij (batch m n : nat) (bm bn tm tn : pos)
  (_ : squash (m == (m/bm) * ((bm/tm) * tm)))
  (_ : squash (n == (n/bn) * ((bn/tn) * tn)))
  : (abs (batch @| m @| n @| INil)
     =~ natlt (batch * ((m/bm) * (n/bn)))
        & (natlt ((bm/tm) * (bn/tn)) & natlt (tm * tn)))
  = let nbr = m/bm in let ntr = bm/tm in
    let nbc = n/bn in let ntc = bn/tn in
    bij_comp (bt2d_abs_shuffle batch m n)
      (bij_comp
         (bij_prod
            (bij_prod (dim_split3 m nbr ntr tm ())
                      (dim_split3 n nbc ntc tn ()))
            (bij_self (natlt batch)))
         (bij_comp (bt2d_regroup #(natlt nbr) #(natlt ntr) #(natlt tm)
                                 #(natlt nbc) #(natlt ntc) #(natlt tn) #(natlt batch))
            (bij_prod (bt2d_block_flat nbr nbc batch)
                      (bij_prod (bij_nat_prod #ntr #ntc)
                                (bij_nat_prod #tm #tn)))))

(* direct arithmetic page-minor cell index *)
unfold
let bt2d_cell_idx (batch m n : nat) (bm bn tm tn : pos)
  (_ : squash (m == (m/bm) * ((bm/tm) * tm)))
  (_ : squash (n == (n/bn) * ((bn/tn) * tn)))
  (bid : natlt (batch * ((m/bm) * (n/bn))))
  (tid : natlt ((bm/tm) * (bn/tn)))
  (idx : natlt (tm * tn))
  : abs (batch @| m @| n @| INil)
  = let nbr = m/bm in let ntr = bm/tm in
    let nbc = n/bn in let ntc = bn/tn in
    let page = bid % batch in
    let rest = bid / batch in
    let br = rest / nbc in
    let bc = rest % nbc in
    let tr = tid / ntc in
    let tc = tid % ntc in
    let i = idx / tn in
    let j = idx % tn in
    ((page <: natlt batch),
      ((prod_ff nbr (ntr * tm) ((br <: natlt nbr), prod_ff ntr tm ((tr <: natlt ntr), (i <: natlt tm)))),
        ((prod_ff nbc (ntc * tn) ((bc <: natlt nbc), prod_ff ntc tn ((tc <: natlt ntc), (j <: natlt tn)))), ())))

(* SMT-opaque re-export of [bt2d_cell_idx], used ONLY inside [bkf]'s per-cell
   write loop.  [bt2d_cell_idx] is [unfold], so its nested Euclidean-division
   term is eagerly expanded everywhere it appears; carrying that expansion
   across the quantified [forevery_extract'] / [elim_forall] re-insertion VC
   triggers a div/mod matching loop in Z3.  Wrapping the index in this opaque
   symbol keeps the quantified frame syntactically inert (the pointwise
   predicate equality reduces to arithmetic on the loop counter alone).  The
   definition is exposed only where genuinely needed, per concrete index, via
   [reveal_opaque] / [bt2d_cptr_eq] — a ground fact that does not pollute the
   quantified goal. *)
[@@ "opaque_to_smt"]
let bt2d_cptr (batch m n : nat) (bm bn tm tn : pos)
  (sq1 : squash (m == (m/bm) * ((bm/tm) * tm)))
  (sq2 : squash (n == (n/bn) * ((bn/tn) * tn)))
  (bid : natlt (batch * ((m/bm) * (n/bn))))
  (tid : natlt ((bm/tm) * (bn/tn)))
  (idx : natlt (tm * tn))
  : abs (batch @| m @| n @| INil)
  = bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid idx

let bt2d_cptr_eq (batch m n : nat) (bm bn tm tn : pos)
  (sq1 : squash (m == (m/bm) * ((bm/tm) * tm)))
  (sq2 : squash (n == (n/bn) * ((bn/tn) * tn)))
  (bid : natlt (batch * ((m/bm) * (n/bn))))
  (tid : natlt ((bm/tm) * (bn/tn)))
  (idx : natlt (tm * tn))
  : Lemma (bt2d_cptr batch m n bm bn tm tn sq1 sq2 bid tid idx
             == bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid idx)
  = reveal_opaque (`%bt2d_cptr) (bt2d_cptr batch m n bm bn tm tn sq1 sq2 bid tid idx)

(* Generic ite-threshold congruence lemma, proved in a clean context.  Used to
   discharge [elim_trade]'s pure side-condition inside [bkf] (whose ambient
   context is too heavy for Z3 to re-derive even this trivial arithmetic fact).
   For [x =!= ci0] the two threshold [if]s pick the same branch, so the two
   [tensor_pts_to_cell] slprops coincide. *)
let bt2d_ite_thresh_slprop
  (#et : Type0) (#r : nat) (#d : shape r) (#l : tlayout d) (a : T.tensor et l)
  (nn : pos) (ci0 : nat)
  (ptr : (natlt nn) -> GTot (abs d))
  (fA fB : (natlt nn) -> GTot et)
  : Lemma (forall (x : natlt nn). ~(x == ci0) ==>
      T.tensor_pts_to_cell a (ptr x) (if x < ci0 + 1 then fA x else fB x)
      == T.tensor_pts_to_cell a (ptr x) (if x < ci0 then fA x else fB x))
  = ()

#push-options "--fuel 4 --ifuel 4 --z3rlimit 40"
let bt2d_gg_full (batch m n : nat) (bm bn tm tn : pos)
  (sq1 : squash (m == (m/bm) * ((bm/tm) * tm)))
  (sq2 : squash (n == (n/bn) * ((bn/tn) * tn)))
  (bid : natlt (batch * ((m/bm) * (n/bn))))
  (tid : natlt ((bm/tm) * (bn/tn)))
  (idx : natlt (tm * tn))
  : Lemma ((bt2d_idx_bij batch m n bm bn tm tn sq1 sq2).gg (bid, (tid, idx))
             == bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid idx)
  = assert_norm ((bt2d_idx_bij batch m n bm bn tm tn sq1 sq2).gg (bid, (tid, idx))
                   == bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid idx)
#pop-options

let bt2d_gg_all (batch m n : nat) (bm bn tm tn : pos)
  (sq1 : squash (m == (m/bm) * ((bm/tm) * tm)))
  (sq2 : squash (n == (n/bn) * ((bn/tn) * tn)))
  : Lemma (forall (bid : natlt (batch * ((m/bm) * (n/bn))))
                  (tid : natlt ((bm/tm) * (bn/tn)))
                  (idx : natlt (tm * tn)).
             (bt2d_idx_bij batch m n bm bn tm tn sq1 sq2).gg (bid, (tid, idx))
               == bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid idx)
  = introduce
      forall (bid : natlt (batch * ((m/bm) * (n/bn))))
             (tid : natlt ((bm/tm) * (bn/tn)))
             (idx : natlt (tm * tn)).
        (bt2d_idx_bij batch m n bm bn tm tn sq1 sq2).gg (bid, (tid, idx))
          == bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid idx
      with bt2d_gg_full batch m n bm bn tm tn sq1 sq2 bid tid idx

(* the value at the arithmetic cell equals the acc2 of the page slice at (grow, gcol) *)
#push-options "--split_queries always --fuel 4 --ifuel 4"
let bt2d_acc_bridge
  (#tc : Type0)
  (batch m n : nat) (bm bn tm tn : pos)
  (sq1 : squash (m == (m/bm) * ((bm/tm) * tm)))
  (sq2 : squash (n == (n/bn) * ((bn/tn) * tn)))
  (e : chest3 tc batch m n)
  (bid : natlt (batch * ((m/bm) * (n/bn))))
  (tid : natlt ((bm/tm) * (bn/tn)))
  (idx : natlt (tm * tn))
  : Lemma
    (requires bm == (bm/tm) * tm /\ bn == (bn/tn) * tn)
    (ensures
      Chest.acc e (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid idx)
        == acc2 (slice_page e (bid % batch))
             (((bid / batch) / (n/bn)) * bm + (tid / (bn/tn)) * tm + idx / tn)
             (((bid / batch) % (n/bn)) * bn + (tid % (bn/tn)) * tn + idx % tn))
  = ()
#pop-options

#push-options "--fuel 4 --ifuel 4"
let bt2d_up3_lemma (#b #r #cc : nat) (p : szlt b) (g : szlt r) (co : szlt cc)
  : Lemma (up ((p, (g, (co, ()))) <: conc (b @| r @| cc @| INil))
             == ((SZ.v p <: natlt b), ((SZ.v g <: natlt r), ((SZ.v co <: natlt cc), ()))))
  = ()
#pop-options

(* Coordinate equality for [bkf]'s per-cell write: the global (page, row, col)
   assembled from the block/thread/cell decode equals [bt2d_cell_idx].  Proved
   in a clean context (a modest nonlinear substitution [br*bm == br*((bm/tm)*tm)]
   given [bm == (bm/tm)*tm]) so [bkf] discharges the coordinate assert from this
   lemma's postcondition rather than by re-deriving it in its heavy VC. *)
#push-options "--fuel 4 --ifuel 4 --z3rlimit 80"
let bt2d_coord_eq (batch m n : nat) (bm bn tm tn : pos)
  (sq1 : squash (m == (m/bm) * ((bm/tm) * tm)))
  (sq2 : squash (n == (n/bn) * ((bn/tn) * tn)))
  (bidn : natlt (batch * ((m/bm) * (n/bn))))
  (tidn : natlt ((bm/tm) * (bn/tn)))
  (idxn : natlt (tm * tn))
  (page : szlt batch) (grow_sz : szlt m) (gcol_sz : szlt n)
  : Lemma
      (requires
        bm == (bm/tm) * tm /\ bn == (bn/tn) * tn /\
        SZ.v page == bidn % batch /\
        SZ.v grow_sz == (bidn / batch / (n/bn)) * bm + ((tidn / (bn/tn)) * tm + idxn / tn) /\
        SZ.v gcol_sz == (bidn / batch % (n/bn)) * bn + ((tidn % (bn/tn)) * tn + idxn % tn))
      (ensures
        (up ((page, (grow_sz, (gcol_sz, ()))) <: conc (batch @| m @| n @| INil)))
          == bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bidn tidn idxn)
  = bt2d_up3_lemma #batch #m #n page grow_sz gcol_sz
#pop-options

(* collapsed (tid, idx) -> single threadcell index, for teardown reuse *)
let bt2d_idx_bij2 (batch m n : nat) (bm bn tm tn : pos)
  (sq1 : squash (m == (m/bm) * ((bm/tm) * tm)))
  (sq2 : squash (n == (n/bn) * ((bn/tn) * tn)))
  : (abs (batch @| m @| n @| INil)
     =~ natlt (batch * ((m/bm) * (n/bn)))
        & natlt (((bm/tm) * (bn/tn)) * (tm * tn)))
  = bij_comp (bt2d_idx_bij batch m n bm bn tm tn sq1 sq2)
      (bij_prod (bij_self (natlt (batch * ((m/bm) * (n/bn)))))
                (bij_nat_prod #((bm/tm) * (bn/tn)) #(tm * tn)))

let bt2d_gg2_full (batch m n : nat) (bm bn tm tn : pos)
  (sq1 : squash (m == (m/bm) * ((bm/tm) * tm)))
  (sq2 : squash (n == (n/bn) * ((bn/tn) * tn)))
  (bid : natlt (batch * ((m/bm) * (n/bn))))
  (tcell : natlt (((bm/tm) * (bn/tn)) * (tm * tn)))
  : Lemma ((bt2d_idx_bij2 batch m n bm bn tm tn sq1 sq2).gg (bid, tcell)
             == bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid
                  (tcell / (tm * tn)) (tcell % (tm * tn)))
  = assert_norm ((bt2d_idx_bij2 batch m n bm bn tm tn sq1 sq2).gg (bid, tcell)
                   == (bt2d_idx_bij batch m n bm bn tm tn sq1 sq2).gg
                        (bid, (tcell / (tm * tn), tcell % (tm * tn))));
    bt2d_gg_full batch m n bm bn tm tn sq1 sq2 bid (tcell / (tm * tn)) (tcell % (tm * tn))

let bt2d_gg2_all (batch m n : nat) (bm bn tm tn : pos)
  (sq1 : squash (m == (m/bm) * ((bm/tm) * tm)))
  (sq2 : squash (n == (n/bn) * ((bn/tn) * tn)))
  : Lemma (forall (bid : natlt (batch * ((m/bm) * (n/bn))))
                  (tcell : natlt (((bm/tm) * (bn/tn)) * (tm * tn))).
             (bt2d_idx_bij2 batch m n bm bn tm tn sq1 sq2).gg (bid, tcell)
               == bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid
                    (tcell / (tm * tn)) (tcell % (tm * tn)))
  = introduce
      forall (bid : natlt (batch * ((m/bm) * (n/bn))))
             (tcell : natlt (((bm/tm) * (bn/tn)) * (tm * tn))).
        (bt2d_idx_bij2 batch m n bm bn tm tn sq1 sq2).gg (bid, tcell)
          == bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid
               (tcell / (tm * tn)) (tcell % (tm * tn))
      with bt2d_gg2_full batch m n bm bn tm tn sq1 sq2 bid tcell

(* ─── batched cell predicates + sendables (EXACT, element-level) ─── *)

(* Dimension-factorization squash: m == (m/bm)*((bm/tm)*tm) from the tiling
   divisibility, needed by [bt2d_cell_idx]. *)
let bt2d_dim_sq (m bm tm : szp)
  : Lemma (requires tm /?+ bm /\ bm /?+ m)
          (ensures SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm))
  = Kuiper.Divides.lemma_nat_divides_pos_divides bm m;
    Kuiper.Divides.lemma_nat_divides_pos_divides tm bm

(* Batched block/thread size requirement (rank-2 requirement scaled by batch). *)
unfold
let bt2d_bsize_req (batch m n : nat) (bm bn tm tn : pos) : prop =
  batch * ((m/bm) * (n/bn)) <= max_blocks /\ ((bm/tm) * (bn/tn)) <= max_threads

(* At batch one, the batched size requirement follows from the rank-2 one. *)
let bt2d_size_req_bsize1 (m n : nat) (bm bn tm tn : pos)
  : Lemma (requires (m/bm) * (n/bn) <= max_blocks /\ ((bm/tm) * (bn/tn)) <= max_threads)
          (ensures bt2d_bsize_req 1 m n bm bn tm tn)
  = ()

(* Per-page total thread count (batch * blocks * threads), packaged as an
   [erased pos]-returning [Tot] helper.  Returning [erased pos] (rather than a
   plain [pos]) keeps the definition in [Tot] despite the ghost [SZ.v] coercions
   in the arithmetic (the ghost computation is absorbed into the [erased] type),
   while the nonlinear 5-factor positivity is discharged ONCE here in isolation.
   [bkpre1]/[bkpost1] use [Ghost.reveal (bt2d_nall_e ...)] as the [fA /. _] and
   [fB /. _] divisor: because [bt2d_nall_e] is not [unfold], the divisor stays an
   OPAQUE [pos] symbol in their VC, so the perm obligation is trivial ([>= 1] by
   refinement) instead of re-deriving the nonlinear product positivity.  This is
   what keeps the merged [--split_queries no] kernel-descriptor record VC from
   cascade-failing on [real <: perm] / nonzero-divisor obligations. *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
[@@ "opaque_to_smt"]
let bt2d_nall_e (batch m n : szp)
  (bm : szp{bm /?+ m}) (bn : szp{bn /?+ n})
  (tm : szp{tm /?+ bm}) (tn : szp{tn /?+ bn})
  : Ghost.erased pos
  = FStar.Math.Lemmas.lemma_div_mod m bm;
    FStar.Math.Lemmas.lemma_div_mod n bn;
    FStar.Math.Lemmas.lemma_div_mod bm tm;
    FStar.Math.Lemmas.lemma_div_mod bn tn;
    (batch * ((m/bm) * (n/bn))) * ((bm/tm) * (bn/tn))
#pop-options

(* Reveal bridge for [bt2d_nall_e]: exposes its value as the concrete product so
   [bsetup]/[bteardown] can bridge their [fA /. n_total] shares to [bkpre1]'s
   opaque [fA /. (Ghost.reveal (bt2d_nall_e ...))] perm divisor. *)
let bt2d_nall_e_eq (batch m n : szp)
  (bm : szp{bm /?+ m}) (bn : szp{bn /?+ n})
  (tm : szp{tm /?+ bm}) (tn : szp{tn /?+ bn})
  : Lemma (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn)
             == (batch * ((m/bm) * (n/bn))) * ((bm/tm) * (bn/tn)))
  = reveal_opaque (`%bt2d_nall_e) (bt2d_nall_e batch m n bm bn tm tn)

unfold
let bkpre1
  (#ta #tb #tc #tacc : Type0) {| scalar ta, scalar tb, scalar tc, scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  (gA : array3 ta lA)
  (eA : chest3 ta batch m k)
  (gB : array3 tb lB)
  (eB : chest3 tb batch k n)
  (gC : array3 tc lC)
  (eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (fA fB : perm)
  (bid : natlt (batch * ((m/bm) * (n/bn))))
  (tid : natlt ((bm/tm) * (bn/tn)))
  : slprop
  =
  (gA |-> Frac (fA /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) eA) **
  (gB |-> Frac (fB /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) eB) **
  forall+ (ii : natlt (tm * tn)).
    tensor_pts_to_cell gC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 bid tid ii)
      (Chest.acc eC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 bid tid ii))

(* Opaque wrapper for the per-thread output-cell resource of a page.  Keeping the
   [forall+ bt2d_cell_idx] body folded across the C-independent [kf_compute] call
   prevents the SMT frame-inference query from expanding [bt2d_cell_idx]'s nested
   Euclidean divisions (which otherwise blows the rlimit).  [fold] before the call,
   [unfold] after, to expose the cells again for the epilogue. *)
let bt2d_c_cells
  (#tc : Type0) {| scalar tc |}
  (#batch #m #n : szp)
  (#lC : layout3 batch m n)
  (gC : array3 tc lC)
  (e : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (bid : natlt (batch * ((m/bm) * (n/bn))))
  (tid : natlt ((bm/tm) * (bn/tn)))
  : slprop
  =
  forall+ (ii : natlt (tm * tn)).
    tensor_pts_to_cell gC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)
      (Chest.acc e (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii))

unfold
let bkpost1
  (#ta #tb #tc #tacc : Type0) {| scalar ta, scalar tb, scalar tc, scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  (gA : array3 ta lA)
  (eA : chest3 ta batch m k)
  (gB : array3 tb lB)
  (eB : chest3 tb batch k n)
  (gC : array3 tc lC)
  (eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (fA fB : perm)
  (bid : natlt (batch * ((m/bm) * (n/bn))))
  (tid : natlt ((bm/tm) * (bn/tn)))
  : slprop
  =
  (gA |-> Frac (fA /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) eA) **
  (gB |-> Frac (fB /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) eB) **
  forall+ (ii : natlt (tm * tn)).
    tensor_pts_to_cell gC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 bid tid ii)
      (Chest.acc (MS.gbmmcomb mapA mapB comb eC eA eB)
        (bt2d_cptr batch m n bm bn tm tn sq1 sq2 bid tid ii))

unfold
let bkpre
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, scalar tb, scalar tc, scalar tacc, va : has_vec_cpy ta, vb : has_vec_cpy tb |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  (gA : array3 ta lA)
  (eA : chest3 ta batch m k)
  (gB : array3 tb lB)
  (eB : chest3 tb batch k n)
  (gC : array3 tc lC)
  (eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#sqf: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc ta tb bm bn bk))
  (bid : natlt (batch * ((m/bm) * (n/bn))))
  (tid : natlt ((bm/tm) * (bn/tn)))
  : slprop
  =
  bkpre1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid **
  live_c_shmems sh #(1.0R /. (bm/tm * (bn/tn)))

unfold
let bkpost
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, scalar tb, scalar tc, scalar tacc, va : has_vec_cpy ta, vb : has_vec_cpy tb |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  (gA : array3 ta lA)
  (eA : chest3 ta batch m k)
  (gB : array3 tb lB)
  (eB : chest3 tb batch k n)
  (gC : array3 tc lC)
  (eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#sqf: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc ta tb bm bn bk))
  (bid : natlt (batch * ((m/bm) * (n/bn))))
  (tid : natlt ((bm/tm) * (bn/tn)))
  : slprop
  =
  bkpost1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid **
  live_c_shmems sh #(1.0R /. (bm/tm * (bn/tn)))


(* ─── batched sendables ─── *)
#push-options "--z3rlimit_factor 10 --fuel 1 --ifuel 1 --split_queries no"
#push-options "--z3rlimit 100"
instance bkpre_block_sendable
  (#ta #tb #tc #tacc : Type0)
  (_a:scalar ta) (_b:scalar tb) (_c:scalar tc) (_acc:scalar tacc)
  (va : has_vec_cpy ta) (vb : has_vec_cpy tb)
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  (gA : array3 ta lA { is_global gA })
  (eA : chest3 ta batch m k)
  (gB : array3 tb lB { is_global gB })
  (eB : chest3 tb batch k n)
  (gC : array3 tc lC { is_global gC })
  (eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#sqf: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc ta tb bm bn bk))
  (pf : c_shmems_inv sh)
  (i : natlt (batch * ((m/bm) * (n/bn))))
  (j : natlt ((bm/tm) * (bn/tn)))
: is_send_across block_of
  (bkpre mapA mapB comb gA eA gB eB gC eC bm bn bk #sqf slA slB tm tn sq1 sq2 fA fB sh i j)
= solve

instance bkpost_block_sendable
  (#ta #tb #tc #tacc : Type0)
  (_a:scalar ta) (_b:scalar tb) (_c:scalar tc) (_acc:scalar tacc)
  (va : has_vec_cpy ta) (vb : has_vec_cpy tb)
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  (gA : array3 ta lA { is_global gA })
  (eA : chest3 ta batch m k)
  (gB : array3 tb lB { is_global gB })
  (eB : chest3 tb batch k n)
  (gC : array3 tc lC { is_global gC })
  (eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#sqf: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc ta tb bm bn bk))
  (pf : c_shmems_inv sh)
  (i : natlt (batch * ((m/bm) * (n/bn))))
  (j : natlt ((bm/tm) * (bn/tn)))
: is_send_across block_of
  (bkpost mapA mapB comb gA eA gB eB gC eC bm bn bk #sqf slA slB tm tn sq1 sq2 fA fB sh i j)
= solve
#pop-options
#pop-options

(* Term-level sz bridge: retype a [szlt nthr_v] thread index to the product form
   [szlt (bm/tm * (bn/tn))] that [kf_compute] expects.  Identity on the value,
   backed by [nthr_v]'s refinement so SMT connects the two bounds. *)
inline_for_extraction noextract
let nthr_to_prod_sz
  (bm tm bn tn : szp)
  (nthr_v : szp{SZ.v nthr_v == bm/tm * (bn/tn)})
  (tid : szlt nthr_v)
  : szlt (bm/tm * (bn/tn))
  = tid

(* ─── batched per-block compute (page-minor decode + cell writes) ─── *)
#push-options "--z3rlimit 800 --z3rlimit_factor 4 --fuel 2 --ifuel 2 --split_queries no"
inline_for_extraction noextract
fn bkf
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, scalar tb, scalar tc, scalar tacc, has_vec_cpy ta, has_vec_cpy tb |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| s3A : strided_row_major_3 lA, s3B : strided_row_major_3 lB |}
  (#_ : squash (chunk ta /?+ s3A.rstride3 /\ chunk ta /?+ s3A.offset3 /\ chunk ta /?+ s3A.pstride3))
  (#_ : squash (chunk tb /?+ s3B.rstride3 /\ chunk tb /?+ s3B.offset3 /\ chunk tb /?+ s3B.pstride3))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3A.offset3 + s3A.pstride3 * p)))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3B.offset3 + s3B.pstride3 * p)))
  (gA : array3 ta lA)
  (#eA : chest3 ta batch m k)
  (gB : array3 tb lB)
  (#eB : chest3 tb batch k n)
  (gC : array3 tc lC)
  (#eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk tb /?+ bn))
  (#_ : squash (chunk ta /?+ bk))
  (#sqf : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (chunk ta * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk tb * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (#_ : squash (aligned 16 (core gA) /\ aligned 16 (core gB)))
  (#_ : squash (SZ.fits (m * n)))
  (#fA #fB : perm)
  (nthr : nat {nthr == bm/tm * (bn/tn)})
  (nblk_v : szp{SZ.v nblk_v == batch * (m/bm * (n/bn))})
  (nthr_v : szp{SZ.v nthr_v == bm/tm * (bn/tn)})
  (sh : c_shmems (shmems_desc ta tb bm bn bk))
  (bid : szlt nblk_v)
  (tid : szlt nthr_v)
  ()
  norewrite
  requires
    gpu **
    bkpre mapA mapB comb gA eA gB eB gC eC bm bn bk #sqf slA slB tm tn sq1 sq2 fA fB sh (SZ.v bid) (SZ.v tid) **
    thread_id (SZ.v nthr_v) tid **
    block_id (SZ.v nblk_v) bid **
    B.barrier_tok (FB.contract (slice_page eA (SZ.v bid % batch)) (slice_page eB (SZ.v bid % batch)) slA slB (fst sh) (fst (snd sh)) nthr (SZ.v bid / batch)) **
    B.barrier_state 0
  ensures
    gpu **
    bkpost mapA mapB comb gA eA gB eB gC eC bm bn bk #sqf slA slB tm tn sq1 sq2 fA fB sh (SZ.v bid) (SZ.v tid) **
    thread_id (SZ.v nthr_v) tid **
    block_id (SZ.v nblk_v) bid **
    B.barrier_tok (FB.contract (slice_page eA (SZ.v bid % batch)) (slice_page eB (SZ.v bid % batch)) slA slB (fst sh) (fst (snd sh)) nthr (SZ.v bid / batch)) **
    B.barrier_state (2 * (k / bk))
{
  (* Decode the page-minor block index. *)
  let page : szlt batch = bid %^ batch;
  let rest : szlt (m/^bm *^ (n/^bn)) = bid /^ batch;
  assert (pure (SZ.v page == SZ.v bid % batch));
  assert (pure (SZ.v rest == SZ.v bid / batch));

  (* Fold the output cells to an opaque token BEFORE normalizing bid%%batch, so the
     token wraps them in [bid%%batch] form and the kf_compute frame query does not
     expand bt2d_cell_idx's Euclidean divisions. *)
  fold (bt2d_c_cells gC eC bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid));

  (* Normalize the incoming barrier token to the decoded [page]/[rest] names so
     it later unifies with [kf_compute]'s contract over the page slice.  The gC
     cells are now folded, so this only rewrites the barrier token. *)
  rewrite each (SZ.v bid % batch) as (SZ.v page);
  rewrite each (SZ.v bid / batch) as (SZ.v rest);

  (* Page-slice views of the operands. *)
  let eA_p : chest2 ta m k = slice_page eA (SZ.v page);
  let eB_p : chest2 tb k n = slice_page eB (SZ.v page);
  let eC_p : chest2 tc m n = slice_page eC (SZ.v page);
  assert rewrites_to eA_p (slice_page eA (SZ.v page));
  assert rewrites_to eB_p (slice_page eB (SZ.v page));
  assert rewrites_to eC_p (slice_page eC (SZ.v page));

  (* Slice out the page-th rank-2 views of A and B (read-only). *)
  tensor_extract_slice_ro gA 0 (SZ.v page);
  tensor_extract_slice_ro gB 0 (SZ.v page);
  let gA_p = sliceof gA 0 (SZ.v page);
  rewrite each sliceof gA 0 (SZ.v page) as gA_p;
  let gB_p = sliceof gB 0 (SZ.v page);
  rewrite each sliceof gB 0 (SZ.v page) as gB_p;

  lem_sliceof_core gA 0 (SZ.v page);
  lem_sliceof_core gB 0 (SZ.v page);
  assert pure (aligned 16 (core gA_p));
  assert pure (aligned 16 (core gB_p));

  (* Alignment facts for the vectorized copy. *)
  lemma_aligned_slice_of_3 _ _ _ lA #s3A (SZ.v page) (chunk ta);
  lemma_aligned_slice_of_3 _ _ _ lB #s3B (SZ.v page) (chunk tb);

  (* Product buffer, then run the C-independent compute core over the page. *)
  let mut rchProd : Pulse.Lib.Array.array tacc = [| zero #tacc #_ ; tm*^tn |];
  pts_to_len rchProd;

  (* Bridge the [thread_id]/[tid] bound from [nthr_v] to the product form
     [kf_compute] expects.  Both bounds are equal by [nthr_v]'s refinement. *)
  rewrite (thread_id (SZ.v nthr_v) tid) as (thread_id (bm/tm * (bn/tn)) (nthr_to_prod_sz bm tm bn tn nthr_v tid));

  (* Bridge the A/B page chests from [chest_slice 0 page _] (as produced by
     [tensor_extract_slice_ro]) to the definitionally-equal [slice_page _ page]
     form ([eA_p]/[eB_p]) that [kf_compute] expects.  [slice_page] is a plain
     [let] over [chest_slice], so the two forms are equal by delta. *)
  rewrite (gA_p |-> Frac (fA /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) (chest_slice 0 (SZ.v page) eA))
       as (gA_p |-> Frac (fA /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) eA_p);
  rewrite (gB_p |-> Frac (fB /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) (chest_slice 0 (SZ.v page) eB))
       as (gB_p |-> Frac (fB /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) eB_p);

  kf_compute mapA mapB #m #n #k #_ #_ #_ #_
    #(slice_of_3 _ _ _ lA #s3A (SZ.v page) ())
    #(slice_of_3 _ _ _ lB #s3B (SZ.v page) ())
    gA_p #eA_p gB_p #eB_p
    #(fA /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn)))
    #(fB /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn)))
    bm bn bk slA slB tm tn nthr sh rest (nthr_to_prod_sz bm tm bn tn nthr_v tid) rchProd ();

  (* Restore the [thread_id] bound to [nthr_v] form for the postcondition. *)
  rewrite (thread_id (bm/tm * (bn/tn)) (nthr_to_prod_sz bm tm bn tn nthr_v tid)) as (thread_id (SZ.v nthr_v) tid);

  (* Restore the A/B page chests to [chest_slice 0 page _] form for the trades. *)
  rewrite (gA_p |-> Frac (fA /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) eA_p)
       as (gA_p |-> Frac (fA /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) (chest_slice 0 (SZ.v page) eA));
  rewrite (gB_p |-> Frac (fB /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) eB_p)
       as (gB_p |-> Frac (fB /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) (chest_slice 0 (SZ.v page) eB));

  unfold (bt2d_c_cells gC eC bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid));

  with vrch_val. assert (rchProd |-> vrch_val);
  pts_to_len rchProd;

  (* Restore A and B page slices (read-only trades). *)
  Trade.elim_trade
    (gA_p |-> Frac (fA /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) (chest_slice 0 (SZ.v page) eA))
    (gA |-> Frac (fA /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) eA);
  Trade.elim_trade
    (gB_p |-> Frac (fB /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) (chest_slice 0 (SZ.v page) eB))
    (gB |-> Frac (fB /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) eB);

  (* Write the accumulated products into the rank-3 output cells owned directly
     (by [bt2d_cell_idx]) in [bkpre1].  Track the exact target value per cell via
     a threshold-conditional [forall+] invariant. *)
  let gbmm = MS.gbmmcomb mapA mapB comb eC eA eB;
  assert rewrites_to gbmm (MS.gbmmcomb mapA mapB comb eC eA eB);
  assert pure (SZ.v bm == (SZ.v bm / SZ.v tm) * SZ.v tm);
  assert pure (SZ.v bn == (SZ.v bn / SZ.v tn) * SZ.v tn);

  (* Convert the plain output cells (indexed by [bt2d_cell_idx]) into the
     threshold-conditional form phrased over the SMT-opaque [bt2d_cptr] (with
     threshold 0), so the while loop below can pick its counter [c == 0] at
     entry.  Using [bt2d_cptr] keeps the quantified [forall+] frame inert
     across the loop's [elim_forall] re-insertion.  The per-cell reveal is
     scoped to each ghost step and never enters the loop's quantified goal. *)
  forevery_map
    (fun (ii : natlt (tm * tn)) ->
      tensor_pts_to_cell gC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
        (Chest.acc eC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)))
    (fun (ii : natlt (tm * tn)) ->
      tensor_pts_to_cell gC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
        (if ii < 0
         then Chest.acc gbmm (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
         else Chest.acc eC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)))
    fn (ii : natlt (tm * tn)) {
      bt2d_cptr_eq batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii;
      rewrite (tensor_pts_to_cell gC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
                 (Chest.acc eC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)))
           as (tensor_pts_to_cell gC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
                (if ii < 0
                 then Chest.acc gbmm (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
                 else Chest.acc eC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)));
    };

  let mut cctr : sz = 0sz;
  while (!cctr <^ tm *^ tn)
    invariant live cctr
    invariant exists* (c : nat) (vr : seq tacc).
      (rchProd |-> vr) **
      pure (SZ.v (!cctr) == c /\ c <= tm * tn /\ Seq.length vr == tm * tn /\
        (forall (idx : natlt (tm * tn)).
          vr @! idx ==
            acc2 (ettile (MS.matmul (Kuiper.Chest.chest_map mapA eA_p) (Kuiper.Chest.chest_map mapB eB_p)) bm bn tm tn (SZ.v rest) (SZ.v tid)) (idx / tn) (idx % tn))) **
      (forall+ (ii : natlt (tm * tn)).
        tensor_pts_to_cell gC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
          (if ii < c
           then Chest.acc gbmm (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
           else Chest.acc eC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)))
    decreases (tm * tn - SZ.v (!cctr))
  {
    open Pulse.Lib.Array;
    pts_to_len rchProd;
    let ci0 = !cctr;
    assert pure (SZ.v ci0 < tm * tn);

    (* Bind the invariant counter [c] and identify it with [SZ.v ci0] so the
       [forevery_extract'] threshold below unifies with the loop invariant. *)
    with cval vrval. assert (
      (rchProd |-> vrval) **
      (forall+ (ii : natlt (tm * tn)).
        tensor_pts_to_cell gC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
          (if ii < cval
           then Chest.acc gbmm (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
           else Chest.acc eC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii))));
    rewrite each cval as (SZ.v ci0);

    forevery_extract' #(natlt (tm * tn)) (SZ.v ci0)
      (fun (ii : natlt (tm * tn)) ->
        tensor_pts_to_cell gC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
          (if ii < SZ.v ci0
           then Chest.acc gbmm (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
           else Chest.acc eC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)));

    (* Arithmetic decode of the cell's global (row, col). *)
    let br = rest /^ (n/^bn);
    let bc = rest %^ (n/^bn);
    let tr = tid /^ (bn/^tn);
    let tc = tid %^ (bn/^tn);
    let irow = ci0 /^ tn;
    let icol = ci0 %^ tn;
    (* Tiling bounds so [grow_sz : szlt m] / [gcol_sz : szlt n] and the SZ ops fit. *)
    assert pure (SZ.v bm == (SZ.v bm / SZ.v tm) * SZ.v tm);
    assert pure (SZ.v bn == (SZ.v bn / SZ.v tn) * SZ.v tn);
    assert pure (SZ.v m == (SZ.v m / SZ.v bm) * SZ.v bm);
    assert pure (SZ.v n == (SZ.v n / SZ.v bn) * SZ.v bn);
    assert pure (SZ.v br < SZ.v m / SZ.v bm);
    assert pure (SZ.v bc < SZ.v n / SZ.v bn);
    assert pure (SZ.v tr < SZ.v bm / SZ.v tm);
    assert pure (SZ.v tc < SZ.v bn / SZ.v tn);
    assert pure (SZ.v irow < SZ.v tm);
    assert pure (SZ.v icol < SZ.v tn);
    assert pure (SZ.v tr * SZ.v tm + SZ.v irow < SZ.v bm);
    assert pure (SZ.v br * SZ.v bm + SZ.v tr * SZ.v tm + SZ.v irow < SZ.v m);
    assert pure (SZ.v tc * SZ.v tn + SZ.v icol < SZ.v bn);
    assert pure (SZ.v bc * SZ.v bn + SZ.v tc * SZ.v tn + SZ.v icol < SZ.v n);
    let grow_sz : szlt m = br *^ bm +^ tr *^ tm +^ irow;
    let gcol_sz : szlt n = bc *^ bn +^ tc *^ tn +^ icol;
    let cci : conc (batch @| m @| n @| INil) = (page, (grow_sz, (gcol_sz, ())));
    assert rewrites_to cci (page, (grow_sz, (gcol_sz, ())));
    (* Coordinate equality via a standalone lemma (clean context); [bkf]'s heavy
       VC cannot reliably re-derive the nonlinear substitution inline. *)
    assert pure (SZ.v grow_sz == (SZ.v bid / SZ.v batch / (SZ.v n / SZ.v bn)) * SZ.v bm
                   + ((SZ.v tid / (SZ.v bn / SZ.v tn)) * SZ.v tm + SZ.v ci0 / SZ.v tn));
    assert pure (SZ.v gcol_sz == (SZ.v bid / SZ.v batch % (SZ.v n / SZ.v bn)) * SZ.v bn
                   + ((SZ.v tid % (SZ.v bn / SZ.v tn)) * SZ.v tn + SZ.v ci0 % SZ.v tn));
    bt2d_coord_eq batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) (SZ.v ci0) page grow_sz gcol_sz;
    assert pure (up cci == bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) (SZ.v ci0));
    (* Ground reveal of the opaque index at the single written cell; this is a
       ground fact about [SZ.v ci0] only and does NOT enter the quantified
       [elim_forall] goal below (whose index is bound). *)
    bt2d_cptr_eq batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) (SZ.v ci0);

    with vold.
      rewrite (tensor_pts_to_cell gC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) (SZ.v ci0)) vold)
           as (tensor_pts_to_cell gC (up cci) vold);

    let v0 = tensor_read_cell gC (page, (grow_sz, (gcol_sz, ())));
    let v1 = rchProd.(ci0);
    let v' = comb v0 v1;
    tensor_write_cell gC (page, (grow_sz, (gcol_sz, ()))) v';

    (* Exact reconstruction: v' == acc gbmm cell. *)
    bt2d_acc_bridge batch m n bm bn tm tn sq1 sq2 eC (SZ.v bid) (SZ.v tid) (SZ.v ci0);
    bt2d_acc_bridge batch m n bm bn tm tn sq1 sq2 gbmm (SZ.v bid) (SZ.v tid) (SZ.v ci0);
    MU.gbmmcomb_slice_page mapA mapB comb eC eA eB (SZ.v page);
    ettile_matmul_pointwise (Kuiper.Chest.chest_map mapA eA_p) (Kuiper.Chest.chest_map mapB eB_p)
      bm bn tm tn (SZ.v rest) (SZ.v tid) (SZ.v ci0 / tn) (SZ.v ci0 % tn);
    MS.lemma_matmul_index (Kuiper.Chest.chest_map mapA eA_p) (Kuiper.Chest.chest_map mapB eB_p)
      ((SZ.v rest / (n/bn)) * bm + (SZ.v tid / (bn/tn)) * tm + SZ.v ci0 / tn)
      ((SZ.v rest % (n/bn)) * bn + (SZ.v tid % (bn/tn)) * tn + SZ.v ci0 % tn);
    assert pure (v' == Chest.acc gbmm (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) (SZ.v ci0)));

    rewrite (tensor_pts_to_cell gC (up cci) v')
         as (tensor_pts_to_cell gC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) (SZ.v ci0))
              (Chest.acc gbmm (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) (SZ.v ci0))));

    (* Re-insert with the incremented threshold predicate (opaque [bt2d_cptr],
       so the universal-trade instantiation stays inert). *)
    Pulse.Lib.Forall.elim_forall
      #(natlt (tm * tn) -> slprop)
      (fun (ii : natlt (tm * tn)) ->
        tensor_pts_to_cell gC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
          (if ii < SZ.v ci0 + 1
           then Chest.acc gbmm (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
           else Chest.acc eC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)));

    rewrite (tensor_pts_to_cell gC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) (SZ.v ci0))
              (Chest.acc gbmm (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) (SZ.v ci0))))
         as (tensor_pts_to_cell gC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) (SZ.v ci0))
              (if SZ.v ci0 < SZ.v ci0 + 1
               then Chest.acc gbmm (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) (SZ.v ci0))
               else Chest.acc eC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) (SZ.v ci0))));

    (* For x =!= ci0, the two threshold conditions coincide ((x < ci0+1) iff
       (x < ci0) on nats), so the [if]-selected values are equal and hence the
       two [tensor_pts_to_cell] slprops are equal by congruence.  Proved by a
       standalone lemma in a clean context; the ambient [bkf] VC is too heavy
       for Z3 to re-derive this trivial fact inline.  The lemma's conclusion is
       exactly [elim_trade]'s pure side-condition. *)
    bt2d_ite_thresh_slprop #_ #_ #_ #_ gC (tm * tn) (SZ.v ci0)
      (fun (x : natlt (tm * tn)) -> bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) x)
      (fun (x : natlt (tm * tn)) -> Chest.acc gbmm (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) x))
      (fun (x : natlt (tm * tn)) -> Chest.acc eC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) x));

    Pulse.Lib.Trade.elim_trade _ _;

    cctr := !cctr +^ 1sz;
  };

  (* After the loop c == tm*tn, so every cell holds the gbmm value.  Substitute
     the concrete threshold so the per-cell [forevery_map] step can discharge
     [ii < tm*tn] from [ii]'s type alone (no captured pure fact needed inside
     the closure), then drop the [bt2d_cptr] wrapper back to [bt2d_cell_idx]
     (the form [bkpost1] expects) with a scoped per-cell reveal. *)
  with c vrf. assert (
    (rchProd |-> vrf) **
    (forall+ (ii : natlt (tm * tn)).
      tensor_pts_to_cell gC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
        (if ii < c
         then Chest.acc gbmm (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
         else Chest.acc eC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii))));
  assert pure (c == tm * tn);
  rewrite each c as (tm * tn);
  forevery_map
    (fun (ii : natlt (tm * tn)) ->
      tensor_pts_to_cell gC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
        (if ii < tm * tn
         then Chest.acc gbmm (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
         else Chest.acc eC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)))
    (fun (ii : natlt (tm * tn)) ->
      tensor_pts_to_cell gC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
        (Chest.acc (MS.gbmmcomb mapA mapB comb eC eA eB) (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)))
    fn (ii : natlt (tm * tn)) {
      bt2d_cptr_eq batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii;
      rewrite (tensor_pts_to_cell gC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
                 (if ii < tm * tn
                  then Chest.acc gbmm (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
                  else Chest.acc eC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)))
           as (tensor_pts_to_cell gC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)
                (Chest.acc (MS.gbmmcomb mapA mapB comb eC eA eB) (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 (SZ.v bid) (SZ.v tid) ii)));
    };

  (* Fold [bkpost]; bridge the decoded page/rest names back to the arithmetic form. *)
  rewrite each eA_p as (slice_page eA (SZ.v bid % batch) <: chest2 ta m k);
  rewrite each eB_p as (slice_page eB (SZ.v bid % batch) <: chest2 tb k n);
  rewrite each (SZ.v rest) as (SZ.v bid / batch);
  rewrite each (SZ.v page) as (SZ.v bid % batch);
}
#pop-options

(* ─── opaque carriers for [bkpre1] / [bkpost1] ─────────────────────────────────
   [bkpre1] / [bkpost1] are [unfold] and inline [bt2d_cell_idx]'s nested
   Euclidean divisions.  [forevery_rw_size2] (which re-refines the [bid]/[tid]
   quantifier types from the nat-product bounds to the [szp] bounds [nblk]/[nthr])
   would re-elaborate that division term at the new type, forcing a non-linear
   div/mod coercion that F* cannot insert at elaboration time ("Ill-typed term").
   Wrapping the per-thread predicate in an [opaque_to_smt] symbol keeps the body
   inert across the size rewrite: the only obligation is the *clean* arg coercion
   [natlt nblk <: natlt (batch*((m/bm)*(n/bn)))], discharged from [nblk]'s
   refinement.  [_eq] reveals the wrapper per concrete index for the fold/unfold
   bridges around the rewrite. *)
[@@ "opaque_to_smt"]
let bt2d_cpre1
  (#ta #tb #tc #tacc : Type0) {| scalar ta, scalar tb, scalar tc, scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  (gA : array3 ta lA)
  (eA : chest3 ta batch m k)
  (gB : array3 tb lB)
  (eB : chest3 tb batch k n)
  (gC : array3 tc lC)
  (eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (fA fB : perm)
  (bid : natlt (batch * ((m/bm) * (n/bn))))
  (tid : natlt ((bm/tm) * (bn/tn)))
  : slprop
  = bkpre1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid

let bt2d_cpre1_eq
  (#ta #tb #tc #tacc : Type0) {| scalar ta, scalar tb, scalar tc, scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  (gA : array3 ta lA)
  (eA : chest3 ta batch m k)
  (gB : array3 tb lB)
  (eB : chest3 tb batch k n)
  (gC : array3 tc lC)
  (eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (fA fB : perm)
  (bid : natlt (batch * ((m/bm) * (n/bn))))
  (tid : natlt ((bm/tm) * (bn/tn)))
  : Lemma (bt2d_cpre1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid
             == bkpre1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid)
  = reveal_opaque (`%bt2d_cpre1)
      (bt2d_cpre1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid)

let bt2d_cpre1_eq_all
  (#ta #tb #tc #tacc : Type0) {| scalar ta, scalar tb, scalar tc, scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  (gA : array3 ta lA)
  (eA : chest3 ta batch m k)
  (gB : array3 tb lB)
  (eB : chest3 tb batch k n)
  (gC : array3 tc lC)
  (eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (fA fB : perm)
  : Lemma
    (forall (bid : natlt (batch * ((m/bm) * (n/bn))))
            (tid : natlt ((bm/tm) * (bn/tn))).
      {:pattern (bt2d_cpre1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid)}
      bkpre1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid
        == bt2d_cpre1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid)
  = introduce forall (bid : natlt (batch * ((m/bm) * (n/bn))))
                     (tid : natlt ((bm/tm) * (bn/tn))).
      bkpre1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid
        == bt2d_cpre1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid
    with (bt2d_cpre1_eq mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid)

[@@ "opaque_to_smt"]
let bt2d_cpost1
  (#ta #tb #tc #tacc : Type0) {| scalar ta, scalar tb, scalar tc, scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  (gA : array3 ta lA)
  (eA : chest3 ta batch m k)
  (gB : array3 tb lB)
  (eB : chest3 tb batch k n)
  (gC : array3 tc lC)
  (eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (fA fB : perm)
  (bid : natlt (batch * ((m/bm) * (n/bn))))
  (tid : natlt ((bm/tm) * (bn/tn)))
  : slprop
  = bkpost1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid

let bt2d_cpost1_eq
  (#ta #tb #tc #tacc : Type0) {| scalar ta, scalar tb, scalar tc, scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  (gA : array3 ta lA)
  (eA : chest3 ta batch m k)
  (gB : array3 tb lB)
  (eB : chest3 tb batch k n)
  (gC : array3 tc lC)
  (eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (fA fB : perm)
  (bid : natlt (batch * ((m/bm) * (n/bn))))
  (tid : natlt ((bm/tm) * (bn/tn)))
  : Lemma (bt2d_cpost1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid
             == bkpost1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid)
  = reveal_opaque (`%bt2d_cpost1)
      (bt2d_cpost1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid)

let bt2d_cpost1_eq_all
  (#ta #tb #tc #tacc : Type0) {| scalar ta, scalar tb, scalar tc, scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  (gA : array3 ta lA)
  (eA : chest3 ta batch m k)
  (gB : array3 tb lB)
  (eB : chest3 tb batch k n)
  (gC : array3 tc lC)
  (eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (fA fB : perm)
  : Lemma
    (forall (bid : natlt (batch * ((m/bm) * (n/bn))))
            (tid : natlt ((bm/tm) * (bn/tn))).
      {:pattern (bt2d_cpost1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid)}
      bkpost1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid
        == bt2d_cpost1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid)
  = introduce forall (bid : natlt (batch * ((m/bm) * (n/bn))))
                     (tid : natlt ((bm/tm) * (bn/tn))).
      bkpost1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid
        == bt2d_cpost1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid
    with (bt2d_cpost1_eq mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid)

(* ─── batched setup / teardown (EXACT, cell-level) ─────────────────────────── *)
#push-options "--z3rlimit 200 --fuel 2 --ifuel 5"
ghost
fn bsetup
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, scalar tb, scalar tc, scalar tacc, has_vec_cpy ta, has_vec_cpy tb |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array3 ta lA)
  (eA : chest3 ta batch m k)
  (gB : array3 tb lB)
  (eB : chest3 tb batch k n)
  (gC : array3 tc lC)
  (eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (SZ.fits (m * n)))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (nblk_v : szp{SZ.v nblk_v == batch * (m/bm * (n/bn))})
  (nthr_v : szp{SZ.v nthr_v == bm/tm * (bn/tn)})
  (fA fB : perm)
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (bid : natlt nblk_v)
             (tid : natlt nthr_v).
      bkpre1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid) **
    emp
{
  let n_total : nat = (batch * (m/bm * (n/bn))) * (bm/tm * (bn/tn));

  (* Step 1: share gA/gB (one copy per (bid,tid)). *)
  tensor_share_n gA n_total;
  tensor_share_n gB n_total;

  (* Step 2: explode gC to cells and reindex via the block/thread/cell bijection. *)
  tensor_explode gC;
  forevery_iso (bt2d_idx_bij batch m n bm bn tm tn sq1 sq2)
    (fun (idx : abs (batch @| m @| n @| INil)) ->
       tensor_pts_to_cell gC idx (Chest.acc eC idx));
  forevery_unflatten' _;

  bt2d_gg_all batch m n bm bn tm tn sq1 sq2;

  (* Step 3: per block, split (tid,ii) and rewrite the cell index to the direct
     arithmetic form (values stay EXACT — no existential). *)
  forevery_map
    (fun (bid : natlt (batch * (m/bm * (n/bn)))) ->
       forall+ (tt : natlt (bm/tm * (bn/tn)) & natlt (tm * tn)).
         tensor_pts_to_cell gC ((bt2d_idx_bij batch m n bm bn tm tn sq1 sq2).gg (bid, tt))
           (Chest.acc eC ((bt2d_idx_bij batch m n bm bn tm tn sq1 sq2).gg (bid, tt))))
    (fun (bid : natlt (batch * (m/bm * (n/bn)))) ->
       forall+ (tid : natlt (bm/tm * (bn/tn))) (ii : natlt (tm * tn)).
         tensor_pts_to_cell gC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)
           (Chest.acc eC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)))
    fn bid {
      forevery_unflatten' _;
      forevery_ext_2 _
        (fun (tid : natlt (bm/tm * (bn/tn))) (ii : natlt (tm * tn)) ->
           tensor_pts_to_cell gC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)
             (Chest.acc eC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)));
    };

  (* Step 4: factor gA/gB shares to (bid,tid) grid. *)
  forevery_factor n_total (batch * (m/bm * (n/bn))) (bm/tm * (bn/tn))
    (fun _ -> gA |-> Frac (fA /. n_total) eA);
  forevery_factor n_total (batch * (m/bm * (n/bn))) (bm/tm * (bn/tn))
    (fun _ -> gB |-> Frac (fB /. n_total) eB);

  (* Step 5: zip A/B/C-cells into bkpre1. *)
  forevery_zip3_2
    (fun (_ : natlt (batch * (m/bm * (n/bn)))) (_ : natlt (bm/tm * (bn/tn))) ->
       gA |-> Frac (fA /. n_total) eA)
    (fun (_ : natlt (batch * (m/bm * (n/bn)))) (_ : natlt (bm/tm * (bn/tn))) ->
       gB |-> Frac (fB /. n_total) eB)
    (fun (bid : natlt (batch * (m/bm * (n/bn)))) (tid : natlt (bm/tm * (bn/tn))) ->
       forall+ (ii : natlt (tm * tn)).
         tensor_pts_to_cell gC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)
           (Chest.acc eC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)));

  (* Step 6: fold to [bkpre1], bridge through the opaque [bt2d_cpre1] carrier so
     the grid-size rewrite [natlt (batch*..)/(bm/tm*..)] → [natlt nblk/nthr] does
     not re-elaborate [bt2d_cell_idx]'s divisions.  First fold the explicit form
     into [bkpre1] ([bkpre1] is [unfold], so this is a definitional identity),
     then rewrite into the opaque carrier per instance, do the clean size
     rewrite, and reveal back into [bkpre1]. *)
  forevery_map_2
    (fun (bid : natlt (batch * (m/bm * (n/bn)))) (tid : natlt (bm/tm * (bn/tn))) ->
       gA |-> Frac (fA /. n_total) eA **
       gB |-> Frac (fB /. n_total) eB **
       (forall+ (ii : natlt (tm * tn)).
          tensor_pts_to_cell gC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)
            (Chest.acc eC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii))))
    (fun (bid : natlt (batch * (m/bm * (n/bn)))) (tid : natlt (bm/tm * (bn/tn))) ->
       bt2d_cpre1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid)
    fn bid tid {
      (* Bridge the share perm from the concrete [n_total] to [bkpre1]'s opaque
         [Ghost.reveal (bt2d_nall_e ...)] divisor (equal by [bt2d_nall_e_eq]). *)
      bt2d_nall_e_eq batch m n bm bn tm tn;
      rewrite (gA |-> Frac (fA /. n_total) eA)
           as (gA |-> Frac (fA /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) eA);
      rewrite (gB |-> Frac (fB /. n_total) eB)
           as (gB |-> Frac (fB /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) eB);
      (* Bridge the C-cell index from raw [bt2d_cell_idx] to the SMT-opaque
         [bt2d_cptr] that [bkpre1] carries (per-cell reveal via [bt2d_cptr_eq]). *)
      forevery_map
        (fun (ii : natlt (tm * tn)) ->
           tensor_pts_to_cell gC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)
             (Chest.acc eC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)))
        (fun (ii : natlt (tm * tn)) ->
           tensor_pts_to_cell gC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 bid tid ii)
             (Chest.acc eC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 bid tid ii)))
        fn (ii : natlt (tm * tn)) {
          bt2d_cptr_eq batch m n bm bn tm tn sq1 sq2 bid tid ii;
          rewrite (tensor_pts_to_cell gC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)
                     (Chest.acc eC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)))
               as (tensor_pts_to_cell gC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 bid tid ii)
                    (Chest.acc eC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 bid tid ii)));
        };
      bt2d_cpre1_eq mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid;
      rewrite (gA |-> Frac (fA /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) eA **
               gB |-> Frac (fB /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) eB **
               (forall+ (ii : natlt (tm * tn)).
                  tensor_pts_to_cell gC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 bid tid ii)
                    (Chest.acc eC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 bid tid ii))))
        as (bt2d_cpre1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid);
    };

  forevery_map_2
    (fun (bid : natlt (batch * (m/bm * (n/bn)))) (tid : natlt (bm/tm * (bn/tn))) ->
       bt2d_cpre1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid)
    (fun (bid : natlt (batch * (m/bm * (n/bn)))) (tid : natlt (bm/tm * (bn/tn))) ->
       bkpre1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid)
    fn bid tid {
      bt2d_cpre1_eq mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid;
      rewrite (bt2d_cpre1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid)
        as (bkpre1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid);
    };
  (* Bridge the [forevery] grid size from the product form to the [nblk_v]/[nthr_v]
     bounds the kernel descriptor expects (equal by their refinements). *)
  forevery_rw_size2 (batch * (m/bm * (n/bn))) (SZ.v nblk_v) (bm/tm * (bn/tn)) (SZ.v nthr_v)
    #(fun (bid : natlt (batch * (m/bm * (n/bn)))) (tid : natlt (bm/tm * (bn/tn))) ->
        bkpre1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid);
  ();
}
#pop-options

#push-options "--z3rlimit 200 --fuel 2 --ifuel 5"
ghost
fn bteardown
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, scalar tb, scalar tc, scalar tacc, has_vec_cpy ta, has_vec_cpy tb |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array3 ta lA)
  (eA : chest3 ta batch m k)
  (gB : array3 tb lB)
  (eB : chest3 tb batch k n)
  (gC : array3 tc lC)
  (eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (SZ.fits (m * n)))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (nblk_v : szp{SZ.v nblk_v == batch * (m/bm * (n/bn))})
  (nthr_v : szp{SZ.v nthr_v == bm/tm * (bn/tn)})
  (fA fB : perm)
  ()
  norewrite
  requires
    (forall+ (bid : natlt nblk_v)
             (tid : natlt nthr_v).
      bkpost1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid) **
    emp
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> MS.gbmmcomb mapA mapB comb eC eA eB
{
  let n_total : nat = (batch * (m/bm * (n/bn))) * (bm/tm * (bn/tn));

  (* Bridge the [forevery] grid size from the [nblk_v]/[nthr_v] bounds back to the
     product form the gather machinery uses (equal by their refinements). *)
  forevery_rw_size2 (SZ.v nblk_v) (batch * (m/bm * (n/bn))) (SZ.v nthr_v) (bm/tm * (bn/tn))
    #(fun (bid : natlt (SZ.v nblk_v)) (tid : natlt (SZ.v nthr_v)) ->
        bkpost1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid);

  (* Step 2: unfold bkpost1 to its explicit A-share / B-share / C-cells form.
     [bkpost1] carries the SMT-opaque [bt2d_cptr] index and the opaque
     [Ghost.reveal (bt2d_nall_e ...)] perm divisor (from the record-VC cascade
     fix); bridge both back per instance to the raw [bt2d_cell_idx] / [n_total]
     form the gather machinery below consumes. *)
  forevery_map_2
    (fun (bid : natlt (batch * (m/bm * (n/bn)))) (tid : natlt (bm/tm * (bn/tn))) ->
      bkpost1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid)
    (fun (bid : natlt (batch * (m/bm * (n/bn)))) (tid : natlt (bm/tm * (bn/tn))) ->
      gA |-> Frac (fA /. n_total) eA **
      gB |-> Frac (fB /. n_total) eB **
      forall+ (ii : natlt (tm * tn)).
        tensor_pts_to_cell gC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)
          (Chest.acc (MS.gbmmcomb mapA mapB comb eC eA eB)
            (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)))
    fn bid tid {
      (* Bridge the share perm from the opaque [Ghost.reveal (bt2d_nall_e ...)]
         divisor back to the concrete [n_total] (equal by [bt2d_nall_e_eq]). *)
      bt2d_nall_e_eq batch m n bm bn tm tn;
      rewrite (gA |-> Frac (fA /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) eA)
           as (gA |-> Frac (fA /. n_total) eA);
      rewrite (gB |-> Frac (fB /. (Ghost.reveal (bt2d_nall_e batch m n bm bn tm tn))) eB)
           as (gB |-> Frac (fB /. n_total) eB);
      (* Bridge the C-cell index from the SMT-opaque [bt2d_cptr] back to the raw
         [bt2d_cell_idx] the gather bijection consumes (per-cell [bt2d_cptr_eq]). *)
      forevery_map
        (fun (ii : natlt (tm * tn)) ->
           tensor_pts_to_cell gC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 bid tid ii)
             (Chest.acc (MS.gbmmcomb mapA mapB comb eC eA eB)
               (bt2d_cptr batch m n bm bn tm tn sq1 sq2 bid tid ii)))
        (fun (ii : natlt (tm * tn)) ->
           tensor_pts_to_cell gC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)
             (Chest.acc (MS.gbmmcomb mapA mapB comb eC eA eB)
               (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)))
        fn (ii : natlt (tm * tn)) {
          bt2d_cptr_eq batch m n bm bn tm tn sq1 sq2 bid tid ii;
          rewrite (tensor_pts_to_cell gC (bt2d_cptr batch m n bm bn tm tn sq1 sq2 bid tid ii)
                     (Chest.acc (MS.gbmmcomb mapA mapB comb eC eA eB)
                       (bt2d_cptr batch m n bm bn tm tn sq1 sq2 bid tid ii)))
               as (tensor_pts_to_cell gC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)
                    (Chest.acc (MS.gbmmcomb mapA mapB comb eC eA eB)
                      (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)));
        };
    };

  (* Step 3: peel off and gather the A/B shares.  Give the residual predicate
     explicitly so the [forall+ ii] cell body is not re-inferred (which trips
     the [natlt] div-bound coercion). *)
  forevery_unzip_2
    (fun (_ : natlt (batch * (m/bm * (n/bn)))) (_ : natlt (bm/tm * (bn/tn))) ->
      gA |-> Frac (fA /. n_total) eA)
    (fun (bid : natlt (batch * (m/bm * (n/bn)))) (tid : natlt (bm/tm * (bn/tn))) ->
      gB |-> Frac (fB /. n_total) eB **
      (forall+ (ii : natlt (tm * tn)).
        tensor_pts_to_cell gC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)
          (Chest.acc (MS.gbmmcomb mapA mapB comb eC eA eB)
            (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii))));
  forevery_unzip_2
    (fun (_ : natlt (batch * (m/bm * (n/bn)))) (_ : natlt (bm/tm * (bn/tn))) ->
      gB |-> Frac (fB /. n_total) eB)
    (fun (bid : natlt (batch * (m/bm * (n/bn)))) (tid : natlt (bm/tm * (bn/tn))) ->
      forall+ (ii : natlt (tm * tn)).
        tensor_pts_to_cell gC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)
          (Chest.acc (MS.gbmmcomb mapA mapB comb eC eA eB)
            (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)));
  forevery_unfactor' n_total (batch * (m/bm * (n/bn))) (bm/tm * (bn/tn))
    (fun (_ : natlt (batch * (m/bm * (n/bn)))) (_ : natlt (bm/tm * (bn/tn))) ->
      gA |-> Frac (fA /. n_total) eA);
  tensor_gather_n gA n_total;
  forevery_unfactor' n_total (batch * (m/bm * (n/bn))) (bm/tm * (bn/tn))
    (fun (_ : natlt (batch * (m/bm * (n/bn)))) (_ : natlt (bm/tm * (bn/tn))) ->
      gB |-> Frac (fB /. n_total) eB);
  tensor_gather_n gB n_total;

  (* Step 4: reconstruct gC from the C-cells (values are EXACT [acc gbmm]). *)
  bt2d_gg_all batch m n bm bn tm tn sq1 sq2;
  forevery_map
    (fun (bid : natlt (batch * (m/bm * (n/bn)))) ->
      forall+ (tid : natlt (bm/tm * (bn/tn))) (ii : natlt (tm * tn)).
        tensor_pts_to_cell gC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)
          (Chest.acc (MS.gbmmcomb mapA mapB comb eC eA eB)
            (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)))
    (fun (bid : natlt (batch * (m/bm * (n/bn)))) ->
      forall+ (tt : natlt (bm/tm * (bn/tn)) & natlt (tm * tn)).
        tensor_pts_to_cell gC ((bt2d_idx_bij batch m n bm bn tm tn sq1 sq2).gg (bid, tt))
          (Chest.acc (MS.gbmmcomb mapA mapB comb eC eA eB)
            ((bt2d_idx_bij batch m n bm bn tm tn sq1 sq2).gg (bid, tt))))
    fn bid {
      forevery_ext_2
        (fun (tid : natlt (bm/tm * (bn/tn))) (ii : natlt (tm * tn)) ->
          tensor_pts_to_cell gC (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)
            (Chest.acc (MS.gbmmcomb mapA mapB comb eC eA eB)
              (bt2d_cell_idx batch m n bm bn tm tn sq1 sq2 bid tid ii)))
        (fun (tid : natlt (bm/tm * (bn/tn))) (ii : natlt (tm * tn)) ->
          tensor_pts_to_cell gC ((bt2d_idx_bij batch m n bm bn tm tn sq1 sq2).gg (bid, (tid, ii)))
            (Chest.acc (MS.gbmmcomb mapA mapB comb eC eA eB)
              ((bt2d_idx_bij batch m n bm bn tm tn sq1 sq2).gg (bid, (tid, ii)))));
      forevery_flatten'
        (fun (tt : natlt (bm/tm * (bn/tn)) & natlt (tm * tn)) ->
          tensor_pts_to_cell gC ((bt2d_idx_bij batch m n bm bn tm tn sq1 sq2).gg (bid, tt))
            (Chest.acc (MS.gbmmcomb mapA mapB comb eC eA eB)
              ((bt2d_idx_bij batch m n bm bn tm tn sq1 sq2).gg (bid, tt))));
    };

  forevery_flatten'
    (fun (xy : natlt (batch * (m/bm * (n/bn))) & (natlt (bm/tm * (bn/tn)) & natlt (tm * tn))) ->
      tensor_pts_to_cell gC ((bt2d_idx_bij batch m n bm bn tm tn sq1 sq2).gg xy)
        (Chest.acc (MS.gbmmcomb mapA mapB comb eC eA eB)
          ((bt2d_idx_bij batch m n bm bn tm tn sq1 sq2).gg xy)));
  forevery_iso_back (bt2d_idx_bij batch m n bm bn tm tn sq1 sq2)
    (fun (idx : abs (batch @| m @| n @| INil)) ->
      tensor_pts_to_cell gC idx (Chest.acc (MS.gbmmcomb mapA mapB comb eC eA eB) idx));
  tensor_implode gC;
}
#pop-options

(* ─── batched per-block distribute / gather of the shared tiles ─── *)
#push-options "--z3rlimit 100 --fuel 1 --ifuel 1"
ghost
fn bblock_setup
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, scalar tb, scalar tc, scalar tacc, has_vec_cpy ta, has_vec_cpy tb |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  (gA : array3 ta lA)
  (eA : chest3 ta batch m k)
  (gB : array3 tb lB)
  (eB : chest3 tb batch k n)
  (gC : array3 tc lC)
  (eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#sqf : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (fA fB : perm)
  (nblk_v : szp{SZ.v nblk_v == batch * (m/bm * (n/bn))})
  (nthr_v : szp{SZ.v nthr_v == bm/tm * (bn/tn)})
  (sh : c_shmems (shmems_desc ta tb bm bn bk))
  (bid : natlt nblk_v)
  ()
  norewrite
  requires
    live_c_shmems sh **
    (forall+ (tid : natlt nthr_v).
      bkpre1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid)
  ensures
    (forall+ (tid : natlt nthr_v).
      bkpre mapA mapB comb gA eA gB eB gC eC bm bn bk #sqf slA slB tm tn sq1 sq2 fA fB sh bid tid) **
    emp (* frame *)
{
  gpu_live_c_shmems_share_underspec sh #1.0R #(bm/tm * (bn/tn));
  forevery_rw_size (bm/tm * (bn/tn)) (SZ.v nthr_v);
  forevery_zip
    (fun (tid : natlt nthr_v) ->
      bkpre1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid)
    _;
}

ghost
fn bblock_teardown
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, scalar tb, scalar tc, scalar tacc, has_vec_cpy ta, has_vec_cpy tb |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  (gA : array3 ta lA)
  (eA : chest3 ta batch m k)
  (gB : array3 tb lB)
  (eB : chest3 tb batch k n)
  (gC : array3 tc lC)
  (eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#sqf : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (fA fB : perm)
  (nblk_v : szp{SZ.v nblk_v == batch * (m/bm * (n/bn))})
  (nthr_v : szp{SZ.v nthr_v == bm/tm * (bn/tn)})
  (sh : c_shmems (shmems_desc ta tb bm bn bk))
  (bid : natlt nblk_v)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr_v).
      bkpost mapA mapB comb gA eA gB eB gC eC bm bn bk #sqf slA slB tm tn sq1 sq2 fA fB sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt nthr_v).
      bkpost1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid)
{
  forevery_unzip
    (fun (tid : natlt nthr_v) ->
      bkpost1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid)
    (fun (_ : natlt nthr_v) -> live_c_shmems sh #(1.0R /. (bm/tm * (bn/tn))));
  forevery_rw_size (SZ.v nthr_v) (bm/tm * (bn/tn))
    #(fun (_ : natlt (SZ.v nthr_v)) -> live_c_shmems sh #(1.0R /. (bm/tm * (bn/tn))));
  gpu_live_c_shmems_gather_underspec sh #1.0R #(bm/tm * (bn/tn));
}
#pop-options

(* ─── batched kernel descriptor (the ONLY kernel description) ─── *)

(* PATH A: the single product↔nblk / product↔nthr bridge, relocated into
   term-level pure calls with an isolated (non-merged) VC so SMT is actually
   consulted with the refinement in scope.  Inline coercions inside the giant
   [--split_queries no] record VC do not get the [nblk_v]/[nthr_v] refinement
   connected to the product bound; these tiny lemmas discharge it locally. *)
unfold
let nblk_to_prod
  (batch m n bm bn : szp)
  (nblk_v : szp{SZ.v nblk_v == batch * (m/bm * (n/bn))})
  (bid : natlt nblk_v)
  : natlt (batch * ((m/bm) * (n/bn)))
  = bid

unfold
let nthr_to_prod
  (bm tm bn tn : szp)
  (nthr_v : szp{SZ.v nthr_v == bm/tm * (bn/tn)})
  (tid : natlt nthr_v)
  : natlt ((bm/tm) * (bn/tn))
  = tid

#push-options "--z3rlimit_factor 4 --split_queries no --fuel 1 --ifuel 1"
inline_for_extraction noextract
let bmk_kernel
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, scalar tb, scalar tc, scalar tacc, has_vec_cpy ta, has_vec_cpy tb |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| s3A : strided_row_major_3 lA, s3B : strided_row_major_3 lB |}
  (#_ : squash (chunk ta /?+ s3A.rstride3 /\ chunk ta /?+ s3A.offset3 /\ chunk ta /?+ s3A.pstride3))
  (#_ : squash (chunk tb /?+ s3B.rstride3 /\ chunk tb /?+ s3B.offset3 /\ chunk tb /?+ s3B.pstride3))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3A.offset3 + s3A.pstride3 * p)))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3B.offset3 + s3B.pstride3 * p)))
  (gA : array3 ta lA { is_global gA })
  (#fA : perm)
  (#eA : chest3 ta batch m k)
  (gB : array3 tb lB { is_global gB })
  (#fB : perm)
  (#eB : chest3 tb batch k n)
  (gC : array3 tc lC { is_global gC })
  (#eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk tb /?+ bn))
  (#_ : squash (chunk ta /?+ bk))
  (#sqf : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (chunk ta * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk tb * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (#_ : squash (SZ.fits (m * n)))
  (nblk_v : szp{SZ.v nblk_v == batch * (m/bm * (n/bn))})
  (nthr_v : szp{SZ.v nthr_v == bm/tm * (bn/tn)})
  (#_ : squash (batch * (m/bm * (n/bn)) <= max_blocks
               /\ (bm/tm * (bn/tn)) <= max_threads))
  (#_ : squash (aligned 16 (core gA) /\ aligned 16 (core gB)))
  ()
  : kernel_desc
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.gbmmcomb mapA mapB comb eC eA eB)
=
  {
  nblk = nblk_v;
  nthr = nthr_v;

  shmems_desc = shmems_desc ta tb bm bn bk;

  barrier_contract = (fun bid ptrs ->
    FB.contract (slice_page eA (bid % SZ.v batch)) (slice_page eB (bid % SZ.v batch))
      slA slB (fst ptrs) (fst (snd ptrs)) (SZ.v bm / SZ.v tm * (SZ.v bn / SZ.v tn)) (bid / SZ.v batch));
  barrier_count    = (fun _bid -> 2 * (SZ.v k / SZ.v bk));
  barrier_ok = (fun bid ptrs ->
    FB.barrier_p_to_q_transform (slice_page eA (bid % SZ.v batch)) (slice_page eB (bid % SZ.v batch))
      slA slB (fst ptrs) (fst (snd ptrs)) (SZ.v bm / SZ.v tm * (SZ.v bn / SZ.v tn)) (bid / SZ.v batch));

  frame = emp;
  block_pre  = (fun (bid:natlt nblk_v) -> forall+ (tid : natlt nthr_v). bkpre1  mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid);
  block_post = (fun (bid:natlt nblk_v) -> forall+ (tid : natlt nthr_v). bkpost1 mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 fA fB bid tid);

  setup      = bsetup    mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 nblk_v nthr_v fA fB;
  teardown   = bteardown mapA mapB comb gA eA gB eB gC eC bm bn bk tm tn sq1 sq2 nblk_v nthr_v fA fB;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = bblock_setup    mapA mapB comb gA eA gB eB gC eC bm bn bk #sqf slA slB tm tn sq1 sq2 fA fB nblk_v nthr_v;
  block_teardown = bblock_teardown mapA mapB comb gA eA gB eB gC eC bm bn bk #sqf slA slB tm tn sq1 sq2 fA fB nblk_v nthr_v;

  kpre      = (fun sh (bid:natlt nblk_v) (tid:natlt nthr_v) -> bkpre  mapA mapB comb gA eA gB eB gC eC bm bn bk #sqf slA slB tm tn sq1 sq2 fA fB sh bid tid);
  kpost     = (fun sh (bid:natlt nblk_v) (tid:natlt nthr_v) -> bkpost mapA mapB comb gA eA gB eB gC eC bm bn bk #sqf slA slB tm tn sq1 sq2 fA fB sh bid tid);

  f = bkf mapA mapB comb gA #eA gB #eB gC #eC bm bn bk slA slB tm tn sq1 sq2 #() #() #fA #fB (SZ.v bm / SZ.v tm * (SZ.v bn / SZ.v tn)) nblk_v nthr_v;

  block_pre_sendable=solve;
  block_post_sendable=solve;
  kpre_sendable=solve;
  kpost_sendable=solve;
}
#pop-options

(* ===================================================================== *)
(* Batched (rank-3) entries.  The batched kernel [bmk_kernel] is the ONLY *)
(* [kernel_desc]; the rank-2 entries below are derived from these at      *)
(* [batch = 1].                                                           *)
(* ===================================================================== *)

inline_for_extraction noextract
fn gbmmcomb_gpu_exact
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, scalar tb, scalar tc, scalar tacc, has_vec_cpy ta, has_vec_cpy tb |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| s3A : strided_row_major_3 lA, s3B : strided_row_major_3 lB |}
  (#_ : squash (chunk ta /?+ s3A.rstride3 /\ chunk ta /?+ s3A.offset3 /\ chunk ta /?+ s3A.pstride3))
  (#_ : squash (chunk tb /?+ s3B.rstride3 /\ chunk tb /?+ s3B.offset3 /\ chunk tb /?+ s3B.pstride3))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3A.offset3 + s3A.pstride3 * p)))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3B.offset3 + s3B.pstride3 * p)))
  (gA : array3 ta lA { is_global gA })
  (#eA : chest3 ta batch m k)
  (gB : array3 tb lB { is_global gB })
  (#eB : chest3 tb batch k n)
  (gC : array3 tc lC { is_global gC })
  (#eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk tb /?+ bn))
  (#_ : squash (chunk ta /?+ bk))
  (#sqf : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (chunk ta * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk tb * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (#_ : squash (SZ.fits (m * n)))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (batch * (m/bm * (n/bn)) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.gbmmcomb mapA mapB comb eC eA eB)
{
  launch_sync (bmk_kernel mapA mapB comb gA gB gC bm bn bk slA slB tm tn sq1 sq2
    (batch *^ (m/^bm *^ (n/^bn))) (bm/^tm *^ (bn/^tn)) ());
}

(* Scalar batched wrapper: [bmmcomb] is [gbmmcomb] at the identity pre-maps. *)
inline_for_extraction noextract
fn bmmcomb_gpu_exact
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (comb : binop et)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| s3A : strided_row_major_3 lA, s3B : strided_row_major_3 lB |}
  (#_ : squash (chunk et /?+ s3A.rstride3 /\ chunk et /?+ s3A.offset3 /\ chunk et /?+ s3A.pstride3))
  (#_ : squash (chunk et /?+ s3B.rstride3 /\ chunk et /?+ s3B.offset3 /\ chunk et /?+ s3B.pstride3))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3A.offset3 + s3A.pstride3 * p)))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3B.offset3 + s3B.pstride3 * p)))
  (gA : array3 et lA { is_global gA })
  (#eA : chest3 et batch m k)
  (gB : array3 et lB { is_global gB })
  (#eB : chest3 et batch k n)
  (gC : array3 et lC { is_global gC })
  (#eC : chest3 et batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#sqf : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (#_ : squash (SZ.fits (m * n)))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (batch * (m/bm * (n/bn)) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.bmmcomb comb eC eA eB)
{
  MS.gbmmcomb_id comb eC eA eB;
  gbmmcomb_gpu_exact (fun (x:et) -> x) (fun (x:et) -> x) comb
    gA gB gC bm bn bk slA slB tm tn sq1 sq2;
}

inline_for_extraction noextract
fn gbmmcomb_gpu_approx
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc,
     has_vec_cpy ta, has_vec_cpy tb |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { approx2 comb comb_r })
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| s3A : strided_row_major_3 lA, s3B : strided_row_major_3 lB |}
  (#_ : squash (chunk ta /?+ s3A.rstride3 /\ chunk ta /?+ s3A.offset3 /\ chunk ta /?+ s3A.pstride3))
  (#_ : squash (chunk tb /?+ s3B.rstride3 /\ chunk tb /?+ s3B.offset3 /\ chunk tb /?+ s3B.pstride3))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3A.offset3 + s3A.pstride3 * p)))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3B.offset3 + s3B.pstride3 * p)))
  (gA : array3 ta lA { is_global gA })
  (#eA : chest3 ta batch m k)
  (gB : array3 tb lB { is_global gB })
  (#eB : chest3 tb batch k n)
  (gC : array3 tc lC { is_global gC })
  (#eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk tb /?+ bn))
  (#_ : squash (chunk ta /?+ bk))
  (#sqf : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (chunk ta * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk tb * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (#_ : squash (SZ.fits (m * n)))
  (#fA #fB : perm)
  (rA : chest3 real batch m k)
  (rB : chest3 real batch k n)
  (rC : chest3 real batch m n)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (batch * (m/bm * (n/bn)) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    pure (MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    exists* (eC' : chest3 tc batch m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.gbmmcomb mapA_r mapB_r comb_r rC rA rB)
{
  gbmmcomb_gpu_exact mapA mapB comb
    gA gB gC bm bn bk slA slB tm tn sq1 sq2;
  MU.gbmmcomb_approx_real mapA mapB comb mapA_r mapB_r comb_r eA eB eC rA rB rC;
  ()
}

inline_for_extraction noextract
fn bmmcomb_gpu_approx
  (#et : Type0) {| scalar et, has_vec_cpy et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| s3A : strided_row_major_3 lA, s3B : strided_row_major_3 lB |}
  (#_ : squash (chunk et /?+ s3A.rstride3 /\ chunk et /?+ s3A.offset3 /\ chunk et /?+ s3A.pstride3))
  (#_ : squash (chunk et /?+ s3B.rstride3 /\ chunk et /?+ s3B.offset3 /\ chunk et /?+ s3B.pstride3))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3A.offset3 + s3A.pstride3 * p)))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3B.offset3 + s3B.pstride3 * p)))
  (gA : array3 et lA { is_global gA })
  (#eA : chest3 et batch m k)
  (gB : array3 et lB { is_global gB })
  (#eB : chest3 et batch k n)
  (gC : array3 et lC { is_global gC })
  (#eC : chest3 et batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#sqf : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (#_ : squash (SZ.fits (m * n)))
  (#fA #fB : perm)
  (rA : chest3 real batch m k)
  (rB : chest3 real batch k n)
  (rC : chest3 real batch m n)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (batch * (m/bm * (n/bn)) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    exists* (eC' : chest3 et batch m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.bmmcomb comb_r rC rA rB)
{
  bmmcomb_gpu_exact comb
    gA gB gC bm bn bk slA slB tm tn sq1 sq2;
  MU.bmmcomb_approx_real comb comb_r eA eB eC rA rB rC;
  ()
}

#push-options "--z3rlimit 600 --fuel 4 --ifuel 4"
inline_for_extraction noextract
fn gmmcomb_gpu_exact
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, scalar tb, scalar tc, scalar tacc, has_vec_cpy ta, has_vec_cpy tb |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk ta) str_A))
  (#_ : squash (aligned_strided_row_major (chunk tb) str_B))
  (gA : array2 ta lA { is_global gA })
  (#eA : chest2 ta m k)
  (gB : array2 tb lB { is_global gB })
  (#eB : chest2 tb k n)
  (gC : array2 tc lC { is_global gC })
  (#eC : chest2 tc m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk tb /?+ bn))
  (#_ : squash (chunk ta /?+ bk))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (chunk ta * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk tb * (bm/tm * (bn/tn)) /?+ (bk * bn)))
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
    on gpu_loc (gC |-> MS.gmmcomb mapA mapB comb eC eA eB)
{
  (* Derive the rank-2 kernel from the batched one at [batch = 1]. *)
  assert pure (SZ.fits (m * n));
  let afA : squash (all_fit (SZ.v m @| SZ.v k @| INil)) = C.layout2_all_fit lA;
  let afB : squash (all_fit (SZ.v k @| SZ.v n @| INil)) = C.layout2_all_fit lB;
  let afC : squash (all_fit (SZ.v m @| SZ.v n @| INil)) = C.layout2_all_fit lC;
  bt2d_size_req_bsize1 (SZ.v m) (SZ.v n) (SZ.v bm) (SZ.v bn) (SZ.v tm) (SZ.v tn);
  let sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)) = bt2d_dim_sq m bm tm;
  let sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)) = bt2d_dim_sq n bn tn;

  (* cast_in: relayout the rank-2 ownership to its batch-one rank-3 view. *)
  map_loc gpu_loc (fun () -> C.t2_to_t3n (SZ.v m) (SZ.v k) afA gA);
  map_loc gpu_loc (fun () -> C.t2_to_t3n (SZ.v k) (SZ.v n) afB gB);
  map_loc gpu_loc (fun () -> C.t2_to_t3n (SZ.v m) (SZ.v n) afC gC);

  (* Expose the batch-one strided characterization fields so the alignment
     obligations for the batched kernel reduce to [str_A]/[str_B] alignment. *)
  lemma_l2_to_l3n_fields #(SZ.v m) #(SZ.v k) #lA #_ #str_A;
  lemma_l2_to_l3n_fields #(SZ.v k) #(SZ.v n) #lB #_ #str_B;

  gbmmcomb_gpu_exact mapA mapB comb
    #1sz #m #n #k
    #(C.l2_to_l3n #(SZ.v m) #(SZ.v k) #lA)
    #(C.l2_to_l3n #(SZ.v k) #(SZ.v n) #lB)
    #(C.l2_to_l3n #(SZ.v m) #(SZ.v n) #lC)
    (relay gA (C.l2_to_l3n #(SZ.v m) #(SZ.v k) #lA))
    #(C.c2_to_c3n (SZ.v m) (SZ.v k) afA eA)
    (relay gB (C.l2_to_l3n #(SZ.v k) #(SZ.v n) #lB))
    #(C.c2_to_c3n (SZ.v k) (SZ.v n) afB eB)
    (relay gC (C.l2_to_l3n #(SZ.v m) #(SZ.v n) #lC))
    #(C.c2_to_c3n (SZ.v m) (SZ.v n) afC eC)
    bm bn bk slA slB tm tn sq1 sq2;

  (* restore the flat rank-2 views of A and B. *)
  map_loc gpu_loc (fun () -> C.t3_to_t2n_ow (SZ.v m) (SZ.v k) afA gA);
  map_loc gpu_loc (fun () -> C.t3_to_t2n_ow (SZ.v k) (SZ.v n) afB gB);

  (* cast_out for C: lower the batched result to the rank-2 gmmcomb post. *)
  map_loc gpu_loc (fun () -> C.t3_to_t2n (SZ.v m) (SZ.v n) afC gC);
  MU.batch1_gmmcomb mapA mapB comb (SZ.v m) (SZ.v k) (SZ.v n) afC afA afB eC eA eB;
  ();
}
#pop-options

(* Scalar rank-2 wrapper: the single-type spec [mmcomb] is [gmmcomb] at the
   identity pre-maps.  The identity-lambda SMTPat does NOT fire on its own, so
   we invoke [MS.gmmcomb_id] explicitly to line up the postconditions. *)
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
  MS.gmmcomb_id comb eC eA eB;
  gmmcomb_gpu_exact (fun (x:et) -> x) (fun (x:et) -> x) comb
    #m #n #k #lA #lB #lC gA #eA gB #eB gC #eC bm bn bk tm tn slA slB #_ #_;
}

inline_for_extraction noextract
fn gmmcomb_gpu_approx
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc,
     has_vec_cpy ta, has_vec_cpy tb |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk ta) str_A))
  (#_ : squash (aligned_strided_row_major (chunk tb) str_B))
  (gA : array2 ta lA { is_global gA })
  (#eA : chest2 ta m k)
  (gB : array2 tb lB { is_global gB })
  (#eB : chest2 tb k n)
  (gC : array2 tc lC { is_global gC })
  (#eC : chest2 tc m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk tb /?+ bn))
  (#_ : squash (chunk ta /?+ bk))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (chunk ta * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk tb * (bm/tm * (bn/tn)) /?+ (bk * bn)))
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
    pure (MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    exists* (eC' : chest2 tc m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.gmmcomb mapA_r mapB_r comb_r rC rA rB)
{
  gmmcomb_gpu_exact mapA mapB comb
    #m #n #k #lA #lB #lC gA #eA gB #eB gC #eC bm bn bk tm tn slA slB #_ #_;
  MU.gmmcomb_approx_real mapA mapB comb mapA_r mapB_r comb_r eC eA eB rA rB rC;
  ()
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
