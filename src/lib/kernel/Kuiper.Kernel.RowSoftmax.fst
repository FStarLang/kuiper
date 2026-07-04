module Kuiper.Kernel.RowSoftmax

#lang-pulse
open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Real { exp }
open Kuiper.EMatrix
open Kuiper.Seq.Common
open Kuiper.Math.OnlineSoftmax { seq_max }
module SZ = Kuiper.SizeT
module BMax = Kuiper.Kernel.HReduce.Block.Max
module RB = Kuiper.Kernel.RowBroadcast
module SM = Kuiper.Spec.Softmax
module Map = Kuiper.Kernel.Map
module Array1 = Kuiper.Array1
module Array2 = Kuiper.Array2
open Kuiper.Tensor 
open Kuiper.Tensor.Layout.Alg { l1_forward }
module Chest = Kuiper.Chest

(* Clamp the block thread count so it never exceeds the row length: this keeps
   every strided bucket of the max reduction non-empty. *)
inline_for_extraction noextract
let clamp_threads (nth lena : szp)
  : (r : szp { SZ.v r <= SZ.v nth /\ SZ.v r <= SZ.v lena })
  = if nth <=^ lena then nth else lena

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
   [row_softmax_real].  Note [s_row_broadcast f a b] applies [f (broadcast i)
   (cell i j)], so the row sum (the broadcast value) is [f]'s FIRST argument and
   we exponentiate the SECOND argument: [div (fexp cell) sum]. *)
#push-options "--z3rlimit 60 --split_queries always"
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
      (ensures RB.s_row_broadcast (fun (s:et) (aij:et) -> div (fexp aij) s) sums sa
               %~ row_softmax_real #m #n ra)
  = let lhs = RB.s_row_broadcast (fun (s:et) (aij:et) -> div (fexp aij) s) sums sa in
    let aux (idx : Kuiper.Shape.abs (m @| n @| INil))
      : Lemma (Chest.acc lhs idx %~ Chest.acc (row_softmax_real #m #n ra) idx) =
      let (i, (j, ())) = idx in
      let denom = chest1_rsum (chest_map exp (chest2_row ra i)) in
      // the row-sum denominator is strictly positive, hence nonzero
      sum_non_zero (lseq_map exp (ematrix_row ra i)) 0.0R;
      // bridge the seq-level row sum to the chest1 sum used by [softmax_real]
      row_sum_bridge ra i;
      assert (denom == rsum (lseq_map exp (ematrix_row ra i)));
      assert (denom >. 0.0R);
      // sa %~ ra at this cell, transported to sums and exp via the _pat lemmas
      assert (v_approximates (acc2 sa i j) (macc ra i j));
      assert (v_approximates (acc1 sums i) denom);
      // exp_approx_pat + div_approx_pat
      assert (v_approximates (div (fexp (acc2 sa i j)) (acc1 sums i))
                             (exp (macc ra i j) /. denom));
      // both sides reduce (acc_pat) to the canonical cell forms
      assert (Chest.acc lhs idx == div (fexp (acc2 sa i j)) (acc1 sums i));
      assert (Chest.acc (row_softmax_real #m #n ra) idx == exp (macc ra i j) /. denom);
      ()
    in
    Classical.forall_intro aux
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
   approximates the shifted real matrix.  Note [s_row_broadcast f a b] applies
   [f (broadcast i) (cell i j)], so to compute [cell - max] the broadcast value
   (the max) must be [f]'s SECOND argument. *)
#push-options "--z3rlimit 60"
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
        RB.s_row_broadcast (fun (mx:et) (aij:et) -> sub aij mx) maxs sa
        %~ mkM (fun i j -> macc ra i j -. cs i))
  = let lhs = RB.s_row_broadcast (fun (mx:et) (aij:et) -> sub aij mx) maxs sa in
    let rhs = mkM #real (fun (i:natlt m) (j:natlt n) -> macc ra i j -. cs i) in
    let aux (idx : Kuiper.Shape.abs (m @| n @| INil))
      : Lemma (Chest.acc lhs idx %~ Chest.acc rhs idx) =
      let (i, (j, ())) = idx in
      assert (v_approximates (acc2 sa i j) (macc ra i j));  // from sa %~ ra
      assert (v_approximates (acc1 maxs i) (cs i));         // from requires
      // sub_approx_pat fires: sub (acc2 sa i j) (acc1 maxs i) %~ (macc ra i j -. cs i)
      ()
    in
    Classical.forall_intro aux
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
  (nth : szp { nth <= max_threads })
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
  let nthm : szp = clamp_threads nth n;
  assert pure (SZ.fits (SZ.v n + SZ.v nthm));
  BMax.reduce_batched_block_max #et (fun x -> x) (fun z -> z) m n nthm a maxs ra;
  with maxs_v. assert (on gpu_loc (maxs |-> maxs_v));
  Classical.forall_intro (id_map_row #m #n ra);
  assert pure (forall (i:nat). i < SZ.v m ==> v_approximates (acc1 maxs_v i) (cs i));

  (* Step 2: subtract the per-row max in place: a[i, j] := a[i, j] - maxs[i].
     [row_broadcast f] writes [f (broadcast i) (cell i j)], so the max is the
     FIRST lambda argument and the cell the SECOND. *)
  subtract_approx #et maxs_v sa ra cs;
  RB.row_broadcast (fun mx aij -> sub aij mx) m n maxs a;

  (* Step 3: tree-reduce exp(a[i, j] - max[i]) into sums[i].  We thread the
     shifted real matrix [ra1] through the reduction. *)
  Kuiper.Kernel.RowReduce.row_reduce fexp exp m n nth a sums ra1;
  with sums_v. assert (on gpu_loc (sums |-> sums_v));

  (* Step 4: in-place fused exp(x) / sums[i] over every (already shifted) cell.
     [row_broadcast f] writes [f (broadcast i) (cell i j)], so the sum is the
     FIRST lambda argument and we exponentiate the SECOND. *)
  RB.row_broadcast (fun s aij -> div (fexp aij) s) m n sums a;

  Kuiper.Tensor.free sums;
  Kuiper.Tensor.free maxs;

  (* Glue: the divide broadcast over the shifted matrix [ra1] approximates
     [row_softmax_real ra1]; shift invariance equates that with the spec
     [row_softmax_real ra]. *)
  let sa1 : ematrix et m n =
    RB.s_row_broadcast (fun (mx:et) (aij:et) -> sub aij mx) maxs_v sa;
  Classical.forall_intro (row_sum_bridge #m #n ra1);
  s_row_div_exp_approx_softmax sums_v sa1 ra1;
  row_softmax_shift_eq #m #n ra cs;
  ()
}

(* ── Unshift: recover the true row sum from the max-shifted one ───────────────
   Step 3 reduces the *shifted* matrix, so [sums] holds
     sum_j exp(ra[i,j] - cs_i) = exp(-cs_i) * sum_j exp(ra[i,j]).
   Multiplying back by exp(cs_i) recovers the true row sum. *)

(* Real-arithmetic core: rsum(exp(row - c)) * exp c == rsum(exp row). *)
let unshift_row_sum_real (row : Seq.seq real) (c : real)
  : Lemma (rsum (seq_map exp (seq_map (fun (z:real) -> z -. c) row)) *. exp c
           == rsum (seq_map exp row))
  = SM.shift_denom row c;
    Seq.lemma_eq_elim (seq_map exp (seq_map (fun (z:real) -> z -. c) row))
                      (seq_map (fun (z:real) -> exp (z -. c)) row);
    exp_positive c

(* Lifted to the array: multiplying each shifted row-sum [sums[i]] by [fexp maxs[i]]
   approximates the true (unshifted) per-row sums. *)
let unshift_sums_correct
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#m #n : nat)
  (sums_v maxs_v : lseq et m)
  (ra : ematrix real m n)
  (cs : (i:nat{i<m}) -> GTot real)
  : Lemma
      (requires
        (forall (i:nat). i < m ==>
          v_approximates (Seq.index sums_v i)
            (rsum (lseq_map exp (ematrix_row
              (mkM #real #m #n (fun i j -> macc ra i j -. cs i)) i)))) /\
        (forall (i:nat). i < m ==> v_approximates (Seq.index maxs_v i) (cs i)))
      (ensures
        Map.lseq_map2 (fun (s:et) (mx:et) -> mul s (fexp mx)) sums_v maxs_v
        %~ Seq.init_ghost m (fun (i:natlt m) -> rsum (lseq_map exp (ematrix_row ra i))))
  = let ra1 = mkM #real #m #n (fun (i:natlt m) (j:natlt n) -> macc ra i j -. cs i) in
    let lhs : lseq et m =
      Map.lseq_map2 (fun (s:et) (mx:et) -> mul s (fexp mx)) sums_v maxs_v in
    let rhs : lseq real m =
      Seq.init_ghost m (fun (i:natlt m) -> rsum (lseq_map exp (ematrix_row ra i))) in
    introduce forall (i:nat). i < m ==> v_approximates (Seq.index lhs i) (Seq.index rhs i)
    with introduce _ ==> _
    with _. (
      Seq.lemma_eq_elim (ematrix_row ra1 i)
                        (seq_map (fun (z:real) -> z -. cs i) (ematrix_row ra i));
      exp_approx (Seq.index maxs_v i) (cs i);
      a_mul (Seq.index sums_v i) (fexp (Seq.index maxs_v i))
            (rsum (lseq_map exp (ematrix_row ra1 i))) (exp (cs i));
      unshift_row_sum_real (ematrix_row ra i) (cs i)
    );
    assert (seq_approximates lhs rhs)

inline_for_extraction noextract
fn row_softmax_gpu_with_sum
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (m : szp { m <= max_blocks })
  (n : szp { m * n <= max_blocks * max_threads })
  (#l : Array2.layout m n) {| ctlayout l |}
  (a : Array2.t et l { Array2.is_global a })
  (#sa : ematrix et m n)
  (ra : ematrix real m n)
  preserves
    cpu
  requires
    on gpu_loc (a |-> sa) **
    pure (sa %~ ra)
  returns 
    sums: (sums: Array1.t et (l1_forward m) { Array1.is_global sums })
  ensures
    exists* (sa' : ematrix et m n) (esums : lseq et m).
      on gpu_loc (a |-> sa') ** 
      on gpu_loc (sums |-> esums) **
      pure (sa' %~ row_softmax_real ra) **
      pure (esums %~ Seq.init_ghost m (fun i -> rsum (lseq_map exp (ematrix_row ra i))))
{
   (* The per-row shift is the row max; [ra1] is the real matrix after the shift. *)
  let cs = (fun (i:nat{i < SZ.v m}) -> seq_max (ematrix_row ra i));
  let ra1 : ematrix real (SZ.v m) (SZ.v n) = mkM (fun i j -> macc ra i j -. cs i);

  (* Allocate per-row maxes and sums on the GPU. *)
  let maxs = Array1.alloc0 #et m (l1_forward m);
  let sums = Array1.alloc0 #et m (l1_forward m);

  (* Step 1: per-row max. Clamp the thread count so every strided bucket is
     non-empty (the max reduction has no identity element). *)
  let nthm : szp = SMK.clamp_threads max_threads n;
  assert pure (SZ.fits (SZ.v n + SZ.v nthm));
  BMax.reduce_batched_block_max #et (fun x -> x) (fun z -> z) m n nthm a maxs ra;
  with maxs_v. assert (on gpu_loc (maxs |-> maxs_v));
  Classical.forall_intro (id_map_row #(SZ.v m) #(SZ.v n) ra);
  assert pure (forall (i:nat). i < SZ.v m ==> v_approximates (Seq.index maxs_v i) (cs i));

  (* Step 2: subtract the per-row max in place: a[i, j] := a[i, j] - maxs[i]. *)
  subtract_approx #et maxs_v sa ra cs;
  RB.row_broadcast (fun x mx -> sub x mx) m n maxs a;

  (* Step 3: tree-reduce exp(a[i, j] - max[i]) into sums[i].  We thread the
     shifted real matrix [ra1] through the reduction. *)
  KB.reduce_batched_block #et fexp exp m n max_threads a sums ra1;
  with sums_v. assert (on gpu_loc (sums |-> sums_v));
  assert pure (forall (r:nat). r < SZ.v m ==>
    v_approximates (Seq.index sums_v r) (rsum (lseq_map exp (ematrix_row ra1 r))));

  (* Step 4: in-place fused exp(x) / sums[i] over every (already shifted) cell. *)
  RB.row_broadcast (fun x s -> div (fexp x) s) m n sums a;

  (* Step 5: unshift the row sums back to the true (unshifted) scale:
     sums[i] := sums[i] * exp(maxs[i]).  Step 3 reduced the max-shifted matrix,
     so [sums] held [exp(-max_i) * true_sum_i]; this recovers [true_sum_i]. *)
  Map.map_gpu2 (fun (s:et) (mx:et) -> mul s (fexp mx)) m sums maxs;
  with sums_v2. assert (on gpu_loc (sums |-> sums_v2));
  assert pure (sums_v2
    == Map.lseq_map2 (fun (s:et) (mx:et) -> mul s (fexp mx)) sums_v maxs_v);

  Array1.free maxs;

  (* Glue: the divide broadcast over the shifted matrix [ra1] approximates
     [row_softmax_real ra1]; shift invariance equates that with the spec
     [row_softmax_real ra]. *)
  let sa1 : ematrix et (SZ.v m) (SZ.v n) =
    RB.s_row_broadcast (fun (x:et) (mx:et) -> sub x mx) maxs_v sa;
  s_row_div_exp_approx_softmax sums_v sa1 ra1;
  row_softmax_shift_eq #(SZ.v m) #(SZ.v n) ra cs;

  (* [sums] now approximates the true per-row sums-of-exp. *)
  unshift_sums_correct sums_v maxs_v ra cs;

  return sums;
}
