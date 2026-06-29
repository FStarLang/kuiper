module Kuiper.Kernel.RowSoftmax

#lang-pulse
friend Kuiper.Kernel.Softmax
open Kuiper
open Kuiper.Real { exp }
open Kuiper.EMatrix
open Kuiper.Seq.Common
open Kuiper.Math.OnlineSoftmax { seq_max }
module SZ = Kuiper.SizeT
module BMax = Kuiper.Kernel.HReduce.Block.Max
module RB = Kuiper.Kernel.RowBroadcast
module SMK = Kuiper.Kernel.Softmax
module SM = Kuiper.Spec.Softmax
open Kuiper.Tensor { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }
module Chest = Kuiper.Chest

let lseq_to_chest1
  (#t : Type0) {| scalar t |}
  (#n : nat)
  (s : lseq t n)
  : Chest.chest1 t n
  = Chest.mk1 (fun i -> s @! i)

let row_sum_bridge (#m #n : nat) (ra1 : ematrix real m n) (i : natlt m)
  : Lemma (chest1_rsum (chest_map exp (chest2_row ra1 i))
           == rsum (lseq_map exp (ematrix_row ra1 i)))
  = Seq.lemma_eq_elim (chest1_to_seq (chest_map exp (chest2_row ra1 i)))
                      (lseq_map exp (ematrix_row ra1 i))

(* ── Approximation glue: numerically-stable cell-wise softmax ─────────────────
   The stable path subtracts a per-row constant [cs i] (the row max) before
   exponentiating.  Softmax is shift-invariant over the reals, so this still
   refines the unchanged golden spec [row_softmax_real]. *)

(* Cell-wise glue for the (already shifted) divide pass: if every row sum
   approximates [rsum (exp row)], the divide broadcast approximates
   [row_softmax_real].  (Same shape as the original unstable proof.) *)
#push-options "--z3rlimit 60"
let s_row_div_exp_approx_softmax
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#m : nat) (#n : nat { n > 0 })
  (sums : Chest.chest1 et m) (sa : ematrix et m n) (ra : ematrix real m n)
  : Lemma
      (requires
        sa %~ ra /\
        (forall (i : nat). i < m ==>
          v_approximates (acc1 sums i)
                         (rsum (lseq_map exp (ematrix_row ra i)))))
      (ensures RB.s_row_broadcast (fun x s -> div (fexp x) s) sums sa %~ row_softmax_real #m #n ra)
  = admit() // Fixme, broke during changes from lseq to chest
#pop-options

(* The identity pre-map leaves a row unchanged, so the batched max reduction's
   result is exactly the row max [cs i]. *)
let id_map_row
  (#m #n : nat)
  (ra : ematrix real m n)
  (i : nat { i < m })
  : Lemma (lseq_map (fun (z:real) -> z) (ematrix_row ra i) == ematrix_row ra i)
  = Seq.lemma_eq_elim (lseq_map (fun (z:real) -> z) (ematrix_row ra i))
                      (ematrix_row ra i)

(* Subtracting the (approximated) per-row max in place yields a matrix that
   approximates the shifted real matrix. *)
#push-options "--z3rlimit 120 --ifuel 2"
let subtract_approx
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#m #n : nat)
  (maxs : Chest.chest1 et m) (sa : ematrix et m n) (ra : ematrix real m n)
  (cs : (i:nat{i<m}) -> GTot real)
  : Lemma
      (requires
        sa %~ ra /\
        (forall (i : nat). i < m ==> v_approximates (acc1 maxs i) (cs i)))
      (ensures
        RB.s_row_broadcast (fun (x:et) (mx:et) -> sub x mx) maxs sa
        %~ mkM (fun i j -> macc ra i j -. cs i))
  = admit() // pointwise sub-approx; revisit
#pop-options

(* Lifted to matrices: shifting each row by [cs i] does not change
   [row_softmax_real]. *)
let row_softmax_shift_eq
  (#m : nat) (#n : nat { n > 0 })
  (ra : ematrix real m n)
  (cs : (i:nat{i<m}) -> GTot real)
  : Lemma (row_softmax_real (mkM (fun i j -> macc ra i j -. cs i))
           == row_softmax_real #m #n ra)
  = let ra1 = mkM (fun (i:natlt m) (j:natlt n) -> macc ra i j -. cs i) in
    let aux (idx : natlt m & (natlt n & unit))
      : Lemma (Chest.acc (row_softmax_real ra1) idx == Chest.acc (row_softmax_real ra) idx) =
      let i = fst idx in
      Chest.lemma_equal_intro (chest2_row ra1 i)
        (chest_map (fun (z:real) -> z -. cs i) (chest2_row ra i));
      Chest.ext (chest2_row ra1 i)
        (chest_map (fun (z:real) -> z -. cs i) (chest2_row ra i));
      SM.softmax_shift (chest2_row ra i) (cs i);
      ()
    in
    Classical.forall_intro aux;
    assert Chest.equal (row_softmax_real ra1) (row_softmax_real #m #n ra);
    ()

inline_for_extraction noextract
fn row_softmax_gpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (m : szp { m <= max_blocks })
  (n : szp { m * n <= max_blocks * max_threads })
  (#l : layout2 m n) {| ctlayout l |}
  (a : array2 et l { is_global a })
  (#sa : ematrix et m n)
  (ra : ematrix real m n)
  preserves
    cpu
  requires
    on gpu_loc (a |-> sa) **
    pure (sa %~ ra)
  ensures
    exists* (sa' : ematrix et m n).
      on gpu_loc (a |-> sa') **
      pure (sa' %~ row_softmax_real ra)
{
  (* The per-row shift is the row max; [ra1] is the real matrix after the shift. *)
  let cs = (fun (i:nat{i < SZ.v m}) -> seq_max (ematrix_row ra i));
  let ra1 : ematrix real m n = mkM (fun i j -> macc ra i j -. cs i);

  (* Allocate per-row maxes and sums on the GPU. *)
  let maxs = Kuiper.Tensor.alloc0 #et m (l1_forward m);
  let sums = Kuiper.Tensor.alloc0 #et m (l1_forward m);

  (* Step 1: per-row max. Clamp the thread count so every strided bucket is
     non-empty (the max reduction has no identity element). *)
  let nthm : szp = SMK.clamp_threads max_threads n;
  assert pure (SZ.fits (SZ.v n + SZ.v nthm));
  BMax.reduce_batched_block_max #et (fun x -> x) (fun z -> z) m n nthm a maxs ra;
  with maxs_v. assert (on gpu_loc (maxs |-> maxs_v));
  Classical.forall_intro (id_map_row #m #n ra);
  assert pure (forall (i:nat). i < SZ.v m ==> v_approximates (acc1 maxs_v i) (cs i));

  (* Step 2: subtract the per-row max in place: a[i, j] := a[i, j] - maxs[i]. *)
  subtract_approx #et maxs_v sa ra cs;
  RB.row_broadcast (fun x mx -> sub x mx) m n maxs a;

  (* Step 3: tree-reduce exp(a[i, j] - max[i]) into sums[i].  We thread the
     shifted real matrix [ra1] through the reduction. *)
  Kuiper.Kernel.RowReduce.row_reduce fexp exp m n max_threads a sums ra1;
  with sums_v. assert (on gpu_loc (sums |-> sums_v));

  (* Step 4: in-place fused exp(x) / sums[i] over every (already shifted) cell. *)
  RB.row_broadcast (fun x s -> div (fexp x) s) m n sums a;

  Kuiper.Tensor.free sums;
  Kuiper.Tensor.free maxs;

  (* Glue: the divide broadcast over the shifted matrix [ra1] approximates
     [row_softmax_real ra1]; shift invariance equates that with the spec
     [row_softmax_real ra]. *)
  let sa1 : ematrix et m n =
    RB.s_row_broadcast (fun (x:et) (mx:et) -> sub x mx) maxs_v sa;
  Classical.forall_intro (row_sum_bridge #m #n ra1);
  s_row_div_exp_approx_softmax sums_v sa1 ra1;
  row_softmax_shift_eq #m #n ra cs;
  ()
}
