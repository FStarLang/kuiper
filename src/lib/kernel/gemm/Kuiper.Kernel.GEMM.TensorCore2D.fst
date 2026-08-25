module Kuiper.Kernel.GEMM.TensorCore2D

module MS = Kuiper.Spec.GEMM
open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc { constraints }
#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 120"

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.Float16
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Kernel.GEMM.Copy.Vec2
open Kuiper.Kernel.GEMM.Tiled.Common.Vec
open Kuiper.Spec.GEMM
open Kuiper.TensorCore
open Pulse.Lib.Array
open Pulse.Lib.Trade

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module T = Kuiper.Tensor
module FB = Kuiper.Kernel.GEMM.FlipFlopBarrier2
module Chest = Kuiper.Chest
module MU = Kuiper.Kernel.GEMM.Util

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.FragmentAcc
open Kuiper.Kernel.GEMM.TensorCore2D.Epilogue
open Kuiper.Kernel.GEMM.TensorCore2D.KLoop
open Kuiper.Kernel.GEMM.TensorCore2D.KernelLoop

let real_add_zero (x : real) : squash (add x zero == x /\ add zero x == x) = ()
let real_add_assoc (x y z : real) : squash (add x (add y z) == add (add x y) z) = ()

let matplus_zero_left (#r #c : nat) (x : chest2 real r c)
  : Lemma (matplus (const _ 0.0R) x `equal` x)
  = ()

let perm_div_pos (f : perm) (n : pos)
  : Lemma (f /. FStar.Real.of_int n >. 0.0R)
  = ()

inline_for_extraction noextract
let size_div (x : sz) (y : szp)
  : Tot (z : sz { SZ.v z == SZ.v x / SZ.v y })
  = x /^ y

inline_for_extraction noextract
let size_rem (x : sz) (y : szp)
  : Tot (z : sz { SZ.v z == SZ.v x % SZ.v y })
  = x %^ y

let block_indices_eq
  (n bn : szp)
  (bid : sz)
  (num_n_tiles : szp {
    SZ.v num_n_tiles == SZ.v n / SZ.v bn })
  (mrow : sz {
    SZ.v mrow == SZ.v bid / SZ.v num_n_tiles })
  (mcol : sz {
    SZ.v mcol == SZ.v bid % SZ.v num_n_tiles })
  : Lemma (
      SZ.v mrow == SZ.v bid / (SZ.v n / SZ.v bn) /\
      SZ.v mcol == SZ.v bid % (SZ.v n / SZ.v bn))
  = ()

#restart-solver
(* At least two statements in [kf] (the [rewrite each sarA/sarB as fst
   sh/fst (snd sh)] pair near the end, folding the shared-memory arrays back
   into [sh]) reliably fail their default fuel-0 attempt -- one burns ~33.5s
   before F*'s automatic retry succeeds at fuel 1 in ~1.3s, wasting ~32s of
   wall time for nothing. Setting [--fuel 1 --ifuel 1] explicitly (matching
   what [mk_kernel] below already does, and what these retries already prove
   sufficient) skips that wasted first attempt for the whole function. *)
#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"
let div_lt_bound (x : nat) (a b : pos)
  : Lemma (requires x < a * b) (ensures x / a < b)
  = FStar.Math.Lemmas.lemma_div_mod x a;
    if x / a >= b then
      FStar.Math.Lemmas.lemma_mult_le_right a b (x / a)

let exact_quotient_pos (x d : pos)
  : Lemma (requires d /?+ x) (ensures x / d > 0)
  = FStar.Math.Lemmas.lemma_div_exact x d;
    assert (x == d * (x / d));
    assert (x / d >= 1)

let self_div (x : pos) : Lemma (x / x == 1)
  = FStar.Math.Lemmas.lemma_div_exact x x
#pop-options

#push-options "--fuel 1 --ifuel 1 --z3rlimit 10"
inline_for_extraction noextract
fn kf
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, sc : scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#m #n #k : szp)
  (#lA : layout2 m k) {| T.ctlayout lA |}
  (gA : array2 et_ab lA)
  (#eA : chest2 et_ab m k)
  (#lB : layout2 k n) {| T.ctlayout lB |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_B))
  (gB : array2 et_ab lB)
  (#eB : chest2 et_ab k n)
  (gC : array2 et_c (rm m n))
  (#eC : chest2 et_c m n)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (SZ.fits (m * k)))
  (#_ : squash (SZ.fits (m * n)))
  (#_ : squash (SZ.fits (k * n)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_c FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_c))
  (#fA #fB : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  // Fused pre-maps / combine (real domain) and their device realizations.
  (mapA mapB : real -> real)
  (comb : real -> real -> real)
  (emA emB : et_ab -> et_ab)
  (ecomb : et_c -> et_c -> et_c)
  (#_ : squash (MU.approx1 emA mapA))
  (#_ : squash (MU.approx1 emB mapB))
  (#_ : squash (Kuiper.Approximates.approx2 ecomb comb))
  (nthr : erased nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm*bk + nthr-1)))
  (#_ : squash (SZ.fits (bk*bn + nthr-1)))
  (#_ : squash (wm * tm /?+ m)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ n)) // idem
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : szlt (m/bm * (n/bn)))
  (tid : szlt nthr)
  ()
  requires
    gpu **
    kpre gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr sh bid tid **
    thread_id nthr tid **
    block_id (m/bm * (n/bn)) bid **
    B.barrier_tok (FB.contract eA eB (rm bm bk) (rm bk bn) (fst sh) (fst (snd sh)) nthr bid) **
    B.barrier_state 0
  ensures
    gpu **
    kpost gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB mapA mapB comb rA rB rC nthr sh bid tid **
    thread_id nthr tid **
    block_id (m/bm * (n/bn)) bid **
    B.barrier_tok (FB.contract eA eB (rm bm bk) (rm bk bn) (fst sh) (fst (snd sh)) nthr bid) **
    B.barrier_state (2 * (k / bk))
{
  exact_quotient_pos m bm;
  exact_quotient_pos n bn;
  exact_quotient_pos bm (wm*tm);
  exact_quotient_pos bn (wn*tn);
  assert pure (wm*tm > 0);
  assert pure (wn*tn > 0);
  assert pure (nthr > 0);
  assert pure (m/bm * (n/bn) * nthr > 0);
  perm_div_pos fA (m/bm * (n/bn) * nthr);
  perm_div_pos fB (m/bm * (n/bn) * nthr);

  unfold_c_shmems sh (`%shmems_desc);
  let (sarA, (sarB, _)) = sh;

  gpu_pts_to_ref sarA;
  gpu_pts_to_ref sarB;

  tensor_abs' (rm bm bk) sarA;
  let sA = from_array (rm bm bk) sarA;
  rewrite each _ as sA; //from_array (rm bm bk) sarA as sA;

  tensor_abs' (rm bk bn) sarB;
  let sB = from_array (rm bk bn) sarB;
  rewrite each _ as sB; //from_array (rm bk bn) sarB as sB;

  let num_n_tiles : szp = n /^ bn;
  let mrow = size_div bid num_n_tiles;
  assert pure (mrow < m / bm);
  let mcol = size_rem bid num_n_tiles;
  assert pure (mcol < n / bn);
  block_indices_eq n bn bid num_n_tiles mrow mcol;

  FStar.Math.Lemmas.swap_mul
    (bm / (wm*tm) * (bn / (wn*tn))) warp_size;
  div_lt_bound (SZ.v tid) warp_size
    (bm / (wm*tm) * (bn / (wn*tn)));
  let wid = tid /^ warp_size;
  div_lt_bound (SZ.v wid) (bn / (wn*tn)) (bm / (wm*tm));
  let warpRow : szlt (bm / (wm*tm)) = wid /^ (bn/^(wn*^tn));
  let warpCol : szlt (bn / (wn*tn)) = wid %^ (bn/^(wn*^tn));

  (* Tensor core fragments *)
  let aFrags = __alloc_array_fragment et_ab FragA tm tn tk FragLRM wm;
  let bFrags = __alloc_array_fragment et_ab FragB tm tn tk FragLRM wn;
  let accFrags = __alloc_array_fragment et_c FragAcc tm tn tk FragLAcc (wm *^ wn);

  // Fill accumulators with 0
  populate_acc_with_zero tm tn tk wm wn accFrags;
  let rAcc0 : chest2 real (wm*tm) (wn*tn) = const _ 0.0R;
  assert (rewrites_to rAcc0 (const _ 0.0R));

  Kuiper.Divides.lemma_div_product (wm*tm) bm m;
  FStar.Math.Lemmas.lemma_eucl_div_bound
    warpRow mrow (bm/(wm*tm));
  FStar.Math.Lemmas.lemma_mult_le_left
    (bm/(wm*tm)) (mrow + 1) (m/bm);
  assert pure (mrow * (bm/(wm*tm)) + warpRow < m/(wm*tm));
  Kuiper.Divides.lemma_div_product (wn*tn) bn n;
  FStar.Math.Lemmas.lemma_eucl_div_bound
    warpCol mcol (bn/(wn*tn));
  FStar.Math.Lemmas.lemma_mult_le_left
    (bn/(wn*tn)) (mcol + 1) (n/bn);
  assert pure (mcol * (bn/(wn*tn)) + warpCol < n/(wn*tn));
  let gwRow : enatlt (m/(wm*tm)) = mrow * (bm/(wm*tm)) + warpRow;
  let gwCol : enatlt (n/(wn*tn)) = mcol * (bn/(wn*tn)) + warpCol;
  assert pure (gwRow ==
    warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size));
  assert pure (gwCol ==
    warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size));

  // Establish the accumulator invariant in the FOLDED, opaque [kacc_inv] form.
  // [kacc_inv] is [@@ "opaque_to_smt"], so it stays syntactically inert in kf's
  // large context: the [macc_ematrix_tiled] SMTPat (acc2 (ematrix_tiled ..))
  // never sees the [ematrix_tiled (chest_map ..)] terms buried inside
  // [__gmatmul_single], which previously drove a non-terminating (multi-GB) Z3
  // blowup at the loop invariant check and the [ktile_advance] call framing.
  // [kacc_inv_eq] exposes the definition only here (vk = 0) and once after the
  // loop -- never inside the loop body.
  kacc_inv_eq m n k bm bn bk tm tn tk wm wn
    (Chest.chest_map mapA rA) (Chest.chest_map mapB rB) rAcc0 gwRow gwCol 0;
  rewrite fragarrayAcc_approximates wm wn accFrags rAcc0
       as fragarrayAcc_approximates wm wn accFrags
            (kacc_inv m n k bm bn bk tm tn tk wm wn
              (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
              rAcc0 gwRow gwCol 0);

  rewrite
    (exists* (x : chest2 _ _ _). sA |-> Frac (1.0R /. nthr) x) **
    (exists* (x : chest2 _ _ _). sB |-> Frac (1.0R /. nthr) x)
  as
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr);

  run_loop gA #eA gB #eB bm bn bk tm tn tk wm wn
    rA rB mapA mapB emA emB nthr
    sarA sarB sA sB bid tid mrow mcol warpRow warpCol gwRow gwCol
    rAcc0 aFrags bFrags accFrags ();

  // Expose the opaque final accumulator invariant back to the raw
  // [__gmatmul_single] form once, outside the loop module.
  kacc_inv_eq m n k bm bn bk tm tn tk wm wn
    (Chest.chest_map mapA rA) (Chest.chest_map mapB rB) rAcc0 gwRow gwCol (k / bk);
  rewrite fragarrayAcc_approximates wm wn accFrags
            (kacc_inv m n k bm bn bk tm tn tk wm wn
              (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
              rAcc0 gwRow gwCol (k / bk))
       as fragarrayAcc_approximates wm wn accFrags
            (__gmatmul_single rAcc0 matmul matplus
              (ematrix_tiled (Chest.chest_map mapA rA) (wm*tm) bk)
              (ematrix_tiled (Chest.chest_map mapB rB) bk (wn*tn))
                gwRow gwCol (k / bk));

  assert
        fragarrayAcc_approximates wm wn accFrags
          (__gmatmul_single rAcc0 matmul matplus
            (ematrix_tiled (Chest.chest_map mapA rA) (wm*tm) bk)
            (ematrix_tiled (Chest.chest_map mapB rB) bk (wn*tn))
              gwRow // (mrow * bm/(wm*tm) + warpRow)
              gwCol // (mcol * bn/(wn*tn) + warpCol)
              (k / bk));

  matmul_tiles_lemma real_add_zero real_add_assoc
    (wm*tm) (wn*tn) bk
    rAcc0 (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
    gwRow gwCol;

  let rAcc' : chest2 real (wm*tm) (wn*tn) =
    gmatmul_single rAcc0 matmul matplus
     (ematrix_tiled (Chest.chest_map mapA rA) (wm*tm) bk)
     (ematrix_tiled (Chest.chest_map mapB rB) bk (wn*tn))
       gwRow gwCol;

  assert pure (
      (__gmatmul_single rAcc0 matmul matplus
        (ematrix_tiled (Chest.chest_map mapA rA) (wm*tm) bk)
        (ematrix_tiled (Chest.chest_map mapB rB) bk (wn*tn)) gwRow gwCol (k / bk))
      == rAcc');

  // The per-warp matmul over the PRE-MAPPED inputs (mapA/mapB applied
  // elementwise via chest_map).  This is the matmul component of [wt_target].
  self_div (SZ.v k);
  let rAcc'' : chest2 real (wm*tm) (wn*tn) =
    MS.matmul (ematrix_subtile (Chest.chest_map mapA rA) (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
              (ematrix_subtile (Chest.chest_map mapB rB) k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)));

  matplus_zero_left rAcc'';
  // ^ This is needed so we can use the result of the matmul_tiles_lemma
  // above...  very boring.

  assert pure (rAcc' == rAcc'');
  rewrite
    fragarrayAcc_approximates wm wn accFrags
      (__gmatmul_single rAcc0 matmul matplus
        (ematrix_tiled (Chest.chest_map mapA rA) (wm*tm) bk)
        (ematrix_tiled (Chest.chest_map mapB rB) bk (wn*tn)) gwRow gwCol (k / bk))
  as
    fragarrayAcc_approximates wm wn accFrags rAcc'';

  with em1. unfold FB.bp_sharing sA em1 nthr;
  with em2. unfold FB.bp_sharing sB em2 nthr;

  rewrite each (tid / 32) as wid;
  // The C-input warp tile (from [kpre1]) approximates this subtile of [rC];
  // [epilogue] reads it back to fuse the output combine.  Bind it so the fold
  // back into [wt_target] below has a stable syntactic form.
  let rCtile : chest2 real (wm*tm) (wn*tn) =
    ematrix_subtile rC (wm*tm) (wn*tn)
      (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid (SZ.v wid))
      (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid (SZ.v wid));
  assert (rewrites_to rCtile
    (ematrix_subtile rC (wm*tm) (wn*tn)
      (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid (SZ.v wid))
      (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid (SZ.v wid))));
  epilogue bm bn bk tm tn tk wm wn accFrags rAcc'' gC comb ecomb rCtile bid wid;
  rewrite each v wid as (tid / 32);

  with vaFrags. assert aFrags |-> vaFrags; drop_ (aFrags |-> vaFrags);
  with vbFrags. assert bFrags |-> vbFrags; drop_ (bFrags |-> vbFrags);
  unfold fragarrayAcc_approximates wm wn accFrags rAcc'';
  with vaccumFrags. assert accFrags |-> vaccumFrags; drop_ (accFrags |-> vaccumFrags);

  tensor_concr sA; rewrite each core sA as sarA;
  tensor_concr sB; rewrite each core sB as sarB;

  rewrite each sarA as fst sh;
  rewrite each sarB as fst (snd sh);

  // Fold the combined output tile into the opaque [wt_target] that [kpost1]
  // expects.  The [rewrite each v wid as (tid / 32)] above already inlined
  // [rCtile], so match on the explicit combined form here.
  rewrite each
    (Chest.chest_comb comb
      (ematrix_subtile rC (wm*tm) (wn*tn)
        (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size))
        (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)))
      rAcc'')
    as (wt_target mapA mapB comb bm bn bk tm tn tk wm wn rA rB rC nthr bid (tid / warp_size));

  fold_c_shmems sh (`%shmems_desc);

  ()
}
#pop-options

#restart-solver
let live_frame_sendable
  (#p : slprop)
  (#ds : list shmem_desc)
  (sh : c_shmems ds)
  (#f : perm)
  (sh_inv : squash (c_shmems_inv sh))
  (send_p : is_send_across block_of p)
  : is_send_across block_of (p ** live_c_shmems sh #f)
= let send_sh = is_send_across_live_c_shmems sh #f sh_inv in
  is_send_across_star p (live_c_shmems sh #f) #send_p #send_sh

let kpre1_sendable
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#m #n #k : szp)
  (#lA : layout2 m k) (#lB : layout2 k n) (#lC : layout2 m n)
  (gA : array2 et_ab lA { is_global gA }) (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB { is_global gB }) (eB : chest2 et_ab k n)
  (gC : array2 et_c lC { is_global gC }) (eC : chest2 et_c m n)
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n /\ bk /?+ k /\
                wm * tm /?+ m /\ wn * tn /?+ n))
  (fA fB : perm)
  (rA : chest2 real m k) (rB : chest2 real k n) (rC : chest2 real m n)
  (nthr : szp {
    SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size })
  (bid : natlt (m/bm * (n/bn))) (tid : natlt nthr)
  : is_send_across gpu_of
      (kpre1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn
        fA fB rA rB rC nthr bid tid)
= solve

let kpost1_sendable
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#m #n #k : szp)
  (#lA : layout2 m k) (#lB : layout2 k n) (#lC : layout2 m n)
  (gA : array2 et_ab lA { is_global gA }) (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB { is_global gB }) (eB : chest2 et_ab k n)
  (gC : array2 et_c lC { is_global gC }) (eC : chest2 et_c m n)
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n /\ bk /?+ k /\
                wm * tm /?+ m /\ wn * tn /?+ n))
  (fA fB : perm)
  (mapA mapB : real -> real) (comb : real -> real -> real)
  (rA : chest2 real m k) (rB : chest2 real k n) (rC : chest2 real m n)
  (nthr : szp {
    SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size })
  (bid : natlt (m/bm * (n/bn))) (tid : natlt nthr)
  : is_send_across gpu_of
      (kpost1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn
        fA fB mapA mapB comb rA rB rC nthr bid tid)
= solve

let block_kpre_sendable
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#m #n #k : szp)
  (#lA : layout2 m k) (#lB : layout2 k n) (#lC : layout2 m n)
  (gA : array2 et_ab lA { is_global gA }) (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB { is_global gB }) (eB : chest2 et_ab k n)
  (gC : array2 et_c lC { is_global gC }) (eC : chest2 et_c m n)
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n /\ bk /?+ k /\
                wm * tm /?+ m /\ wn * tn /?+ n))
  (fA fB : perm)
  (rA : chest2 real m k) (rB : chest2 real k n) (rC : chest2 real m n)
  (nthr : szp {
    SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size })
  (bid : natlt (m/bm * (n/bn)))
  : is_send_across gpu_of
      (forall+ (tid : natlt nthr).
        kpre1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn
          fA fB rA rB rC nthr bid tid)
= let send_tid (tid : natlt nthr) : is_send_across gpu_of
    (kpre1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn
      fA fB rA rB rC nthr bid tid) =
    kpre1_sendable gA eA gB eB gC eC bm bn bk tm tn tk wm wn
      fA fB rA rB rC nthr bid tid in
  is_send_across_forevery
    (fun tid -> kpre1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn
      fA fB rA rB rC nthr bid tid) gpu_of #send_tid

let block_kpost_sendable
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#m #n #k : szp)
  (#lA : layout2 m k) (#lB : layout2 k n) (#lC : layout2 m n)
  (gA : array2 et_ab lA { is_global gA }) (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB { is_global gB }) (eB : chest2 et_ab k n)
  (gC : array2 et_c lC { is_global gC }) (eC : chest2 et_c m n)
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n /\ bk /?+ k /\
                wm * tm /?+ m /\ wn * tn /?+ n))
  (fA fB : perm)
  (mapA mapB : real -> real) (comb : real -> real -> real)
  (rA : chest2 real m k) (rB : chest2 real k n) (rC : chest2 real m n)
  (nthr : szp {
    SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size })
  (bid : natlt (m/bm * (n/bn)))
  : is_send_across gpu_of
      (forall+ (tid : natlt nthr).
        kpost1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn
          fA fB mapA mapB comb rA rB rC nthr bid tid)
= let send_tid (tid : natlt nthr) : is_send_across gpu_of
    (kpost1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn
      fA fB mapA mapB comb rA rB rC nthr bid tid) =
    kpost1_sendable gA eA gB eB gC eC bm bn bk tm tn tk wm wn
      fA fB mapA mapB comb rA rB rC nthr bid tid in
  is_send_across_forevery
    (fun tid -> kpost1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn
      fA fB mapA mapB comb rA rB rC nthr bid tid) gpu_of #send_tid

#push-options "--fuel 1 --ifuel 1 --z3rlimit_factor 10"
inline_for_extraction noextract
let mk_kernel
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#m #n #k : szp)
  (#lA : layout2 m k) {| T.ctlayout lA |}
  (gA : array2 et_ab lA  { is_global gA })
  (#eA : chest2 et_ab m k)
  (#lB : layout2 k n) {| T.ctlayout lB |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_B))
  (gB : array2 et_ab lB { is_global gB })
  (#eB : chest2 et_ab k n)
  (gC : array2 et_c (rm m n) { is_global gC })
  // ^ Why does this have a fixed layout?
  (#_ : squash (SZ.fits (m * n)))
  (#eC : chest2 et_c m n)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (#fA #fB : perm)
  (nblk : szp{SZ.v nblk == m/bm * (n/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (wm * wn)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_c FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_c))
  (#_ : squash (SZ.fits (bm*bk + nthr-1)))
  (#_ : squash (SZ.fits (bk*bn + nthr-1)))
  (#_ : squash (nblk <= max_blocks))
  (#_ : squash (nthr <= max_threads))
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  // Fused elementwise pre-maps on the inputs and combine on the output, threaded
  // in the REAL domain, plus their approximation-compatible DEVICE realizations.
  (mapA mapB : real -> real)
  (comb : real -> real -> real)
  (emA emB : et_ab -> et_ab)
  (ecomb : et_c -> et_c -> et_c)
  (#_ : squash (MU.approx1 emA mapA))
  (#_ : squash (MU.approx1 emB mapB))
  (#_ : squash (Kuiper.Approximates.approx2 ecomb comb))
  (#_ : squash (wm * tm /?+ m)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ n)) // idem
  ()
  : kernel_desc
      (gA |-> Frac fA eA ** pure (eA %~ rA) **
       gB |-> Frac fB eB ** pure (eB %~ rB) **
       gC |-> eC ** pure (eC %~ rC))
      (gA |-> Frac fA eA **
       gB |-> Frac fB eB **
       (exists* (eC' : chest2 et_c m n).
         gC |-> eC' ** pure (eC' %~ MS.gmmcomb mapA mapB comb rC rA rB)))
= {
  nblk;
  nthr;

  shmems_desc = shmems_desc et_ab bm bn bk;

  barrier_contract = (fun bid ptrs -> FB.contract eA eB (rm bm bk) (rm bk bn) (fst ptrs) (fst (snd ptrs)) nthr bid);
  barrier_count    = (fun _bid -> 2 * (SZ.v k / SZ.v bk));
  barrier_ok = (fun bid ptrs -> FB.barrier_p_to_q_transform eA eB (rm bm bk) (rm bk bn) (fst ptrs) (fst (snd ptrs)) nthr bid);

  frame = pure (SZ.fits ((rm m n).ulen));
  block_pre  = (fun bid -> forall+ (tid : natlt nthr). kpre1  gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid);
  block_post = (fun bid -> forall+ (tid : natlt nthr). kpost1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB mapA mapB comb rA rB rC nthr bid tid);

  setup      = setup    gA eA gB eB gC eC bm bn bk tm tn tk wm wn nblk nthr fA fB rA rB rC;
  teardown   = teardown gA eA gB eB gC eC bm bn bk tm tn tk wm wn nblk nthr fA fB mapA mapB comb rA rB rC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup    gA eA gB eB gC eC bm bn bk tm tn tk wm wn nblk nthr fA fB rA rB rC;
  block_teardown = block_teardown gA eA gB eB gC eC bm bn bk tm tn tk wm wn nblk nthr fA fB mapA mapB comb rA rB rC;

  kpre      = kpre  gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr;
  kpost     = kpost gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB mapA mapB comb rA rB rC nthr;

  f = kf gA #eA gB #eB gC #eC bm bn bk tm tn tk wm wn rA rB rC mapA mapB comb emA emB ecomb (SZ.v nthr);

  block_pre_sendable=block_kpre_sendable gA eA gB eB gC eC
    bm bn bk tm tn tk wm wn fA fB rA rB rC nthr;
  block_post_sendable=block_kpost_sendable gA eA gB eB gC eC
    bm bn bk tm tn tk wm wn fA fB mapA mapB comb rA rB rC nthr;
  kpre_sendable=(fun sh sh_inv bid tid ->
    let p = kpre1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn
      fA fB rA rB rC nthr bid tid in
    let gpu_send = kpre1_sendable gA eA gB eB gC eC
      bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid in
    let base_send = send_across_if_send_across_gpu p gpu_send in
    live_frame_sendable sh sh_inv base_send);
  kpost_sendable=(fun sh sh_inv bid tid ->
    let p = kpost1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn
      fA fB mapA mapB comb rA rB rC nthr bid tid in
    let gpu_send = kpost1_sendable gA eA gB eB gC eC
      bm bn bk tm tn tk wm wn fA fB mapA mapB comb rA rB rC nthr bid tid in
    let base_send = send_across_if_send_across_gpu p gpu_send in
    live_frame_sendable sh sh_inv base_send);
}
#pop-options
