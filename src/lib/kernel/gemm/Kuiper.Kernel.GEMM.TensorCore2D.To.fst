module Kuiper.Kernel.GEMM.TensorCore2D.To

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15"

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.Float16
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.TensorRO { vtlayout_of_tlayout }
module RO = Kuiper.TensorRO
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Kernel.GEMM.Copy.Vec2
open Kuiper.Kernel.GEMM.Tiled.Common.Vec
open Kuiper.Spec.GEMM
open Kuiper.TensorCore
open Pulse.Lib.Array
open Pulse.Lib.Trade

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module FB = Kuiper.Kernel.GEMM.FlipFlopBarrier2

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc


open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.Teardown
open Kuiper.Kernel.GEMM.TensorCore2D.To.BaseSendable
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelBody

let scratch_fits_of_kernel_constraints
  (m n : szp)
  (bm bn bk tm tn tk wm wn : szp{
    constraints bm bn bk tm tn tk wm wn})
  (#_ : squash (SZ.fits (m * n)))
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  : squash (SZ.fits ((nthr / warp_size) * tm * tn))
=
  FStar.Math.Lemmas.cancel_mul_div
    (SZ.v bm / (SZ.v wm * SZ.v tm) *
      (SZ.v bn / (SZ.v wn * SZ.v tn)))
    (SZ.v warp_size);
  assert (SZ.v nthr / SZ.v warp_size ==
    SZ.v bm / (SZ.v wm * SZ.v tm) *
    (SZ.v bn / (SZ.v wn * SZ.v tn)));
  FStar.Math.Lemmas.div_exact_r
    (SZ.v bm) (SZ.v wm * SZ.v tm);
  assert (
    SZ.v wm * SZ.v tm *
    (SZ.v bm / (SZ.v wm * SZ.v tm)) == SZ.v bm);
  FStar.Math.Lemmas.div_exact_r
    (SZ.v bn) (SZ.v wn * SZ.v tn);
  assert (
    SZ.v wn * SZ.v tn *
    (SZ.v bn / (SZ.v wn * SZ.v tn)) == SZ.v bn);
  assert (
    (SZ.v nthr / SZ.v warp_size) * SZ.v tm * SZ.v tn <=
    SZ.v bm * SZ.v bn);
  Kuiper.Divides.lemma_nat_divides_pos_divides (SZ.v bm) (SZ.v m);
  Kuiper.Divides.lemma_divides_le (SZ.v bm) (SZ.v m);
  Kuiper.Divides.lemma_nat_divides_pos_divides (SZ.v bn) (SZ.v n);
  Kuiper.Divides.lemma_divides_le (SZ.v bn) (SZ.v n);
  assert (SZ.v bm * SZ.v bn <= SZ.v m * SZ.v n);
  ()

let squash_and
  (#p #q : prop)
  (#_ : squash p)
  (#_ : squash q)
  : squash (p /\ q)
= ()

let warp_divides_thread_count
  (bm bn tm tn wm wn : szp)
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  : squash (warp_size /?+ nthr)
=
  FStar.Math.Lemmas.cancel_mul_mod
    (SZ.v bm / (SZ.v wm * SZ.v tm) *
      (SZ.v bn / (SZ.v wn * SZ.v tn)))
    (SZ.v warp_size);
  ()

let shmem_inv_components_to
  (#et_ab #et_acc : Type0)
  {| sized et_ab, sized et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (#_ : squash (c_shmems_inv sh))
  : squash (
      c_shmem_inv (fst sh) /\
      c_shmem_inv (fst (snd sh)) /\
      c_shmem_inv (fst (snd (snd sh))))
=
  assert (c_shmem_inv (fst sh));
  assert (c_shmems_inv (snd sh));
  assert (c_shmem_inv (fst (snd sh)));
  assert (c_shmems_inv (snd (snd sh)));
  assert (c_shmem_inv (fst (snd (snd sh))));
  ()

let scratch_tile_sendable_to
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, scalar et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (#_ : squash (c_shmem_inv (fst (snd (snd sh)))))
  (tid : natlt nthr)
  (eAcc : chest2 et_acc tm tn)
  : is_send_across block_of (
      scratch_tile bm bn bk tm tn nthr sh (tid / warp_size)
        |-> Frac (1.0R /. warp_size) eAcc)
= is_send_across_tensor
    (scratch_tile bm bn bk tm tn nthr sh (tid / warp_size))
    block_of #_ #(1.0R /. warp_size) eAcc

let scratch_tile_live_sendable_to
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, scalar et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (#_ : squash (c_shmem_inv (fst (snd (snd sh)))))
  (tid : natlt nthr)
  : is_send_across block_of (
      scratch_tile_live bm bn bk tm tn nthr sh tid)
=
  let ff (eAcc : chest2 et_acc tm tn) :
    is_send_across block_of (
      scratch_tile bm bn bk tm tn nthr sh (tid / warp_size)
        |-> Frac (1.0R /. warp_size) eAcc) =
    scratch_tile_sendable_to bm bn bk tm tn nthr sh #_ tid eAcc in
  is_send_across_exists
    (fun eAcc ->
      scratch_tile bm bn bk tm tn nthr sh (tid / warp_size)
        |-> Frac (1.0R /. warp_size) eAcc)
    #ff

let shared_thread_live_sendable_to
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, scalar et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (#_ : squash (c_shmems_inv sh))
  (tid : natlt nthr)
  : is_send_across block_of (
      shared_thread_live bm bn bk tm tn nthr sh tid)
=
  let sh_invs =
    shmem_inv_components_to bm bn bk tm tn nthr sh #_ in
  let scratch_send =
    scratch_tile_live_sendable_to bm bn bk tm tn nthr sh #_ tid in
  solve

let shared_thread_frame_sendable_to
  (#p : slprop)
  (#_ : is_send_across block_of p)
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, scalar et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (#_ : squash (c_shmems_inv sh))
  (tid : natlt nthr)
  : is_send_across block_of (
      p ** shared_thread_live bm bn bk tm tn nthr sh tid)
=
  let shared_send =
    shared_thread_live_sendable_to bm bn bk tm tn nthr sh #_ tid in
  solve

let mk_kernel_arithmetic_facts
  (m n : szp)
  (bm bn bk tm tn tk wm wn : szp)
  (#_ : squash (constraints bm bn bk tm tn tk wm wn))
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (nblk : szp)
  (nthr : szp)
  : Lemma (
      SZ.v wm * SZ.v tm / SZ.v tm == SZ.v wm /\
      SZ.v wn * SZ.v tn / SZ.v tn == SZ.v wn /\
      constraints bm bn bk tm tn tk wm wn /\
      (bm /?+ m /\ bn /?+ n /\
       wm * tm /?+ bm /\ wn * tn /?+ bn) /\
      SZ.v nblk > 0 /\
      SZ.v nthr > 0)
=
  FStar.Math.Lemmas.cancel_mul_div (SZ.v wm) (SZ.v tm);
  FStar.Math.Lemmas.cancel_mul_div (SZ.v wn) (SZ.v tn)

#push-options "--fuel 1 --ifuel 1 --split_queries no --z3rlimit_factor 4"
inline_for_extraction noextract
let mk_kernel
  (#et_ab #et_cd #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab |}
  {| scalar et_acc, real_like et_acc |}
  {| scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  (comb : et_cd -> et_acc -> et_cd)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #n #k : szp)
  (#lA : layout2 m k) {| T.ctlayout lA |}
  (gA : array2 et_ab lA  { is_global gA })
  (#eA : chest2 et_ab m k)
  (#lB : layout2 k n) {| T.ctlayout lB |}
  {| str_A : strided_row_major (vtlayout_of_tlayout lA),
     str_B : strided_row_major (vtlayout_of_tlayout lB) |}
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_B))
  (gB : array2 et_ab lB { is_global gB })
  (#eB : chest2 et_ab k n)
  (#lC : RO.vlayout2 m n)
  {| strC : strided_row_major lC |}
  (#_ : squash (aligned_strided_row_major (chunk et_cd) strC))
  (gC : RO.roarray2 et_cd lC { RO.is_global gC })
  (#mn_fits : squash (SZ.fits (m * n)))
  (#eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n) { is_global gD })
  (#eD : chest2 et_cd m n)
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#bm_div_m : squash (bm /?+ m))
  (#bn_div_n : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#shared_fits : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (aligned 16 (RO.core gC)))
  (#_ : squash (aligned 16 (core gD)))
  (#_ : squash (chunk et_cd /?+ n))
  (#_ : squash (chunk et_cd /?+ tn))
  (#fA #fB #fC : perm)
  (nblk : szp{SZ.v nblk == m/bm * (n/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (wm * wn)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (SZ.fits (bm*bk + nthr-1)))
  (#_ : squash (SZ.fits (bk*bn + nthr-1)))
  (#_ : squash (nblk <= max_blocks))
  (#threads_limit : squash (nthr <= max_threads))
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (#_ : squash (wm * tm /?+ m)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ n)) // idem
  ()
  : kernel_desc
      (gA |-> Frac fA eA ** pure (eA %~ rA) **
       gB |-> Frac fB eB ** pure (eB %~ rB) **
       gC |-> Frac fC eC ** pure (eC %~ rC) **
       live gD)
      (gA |-> Frac fA eA **
       gB |-> Frac fB eB **
       gC |-> Frac fC eC **
       (exists* (eD' : chest2 et_cd m n).
         gD |-> eD' ** pure (eD' %~ MS.mmcomb comb_r rC rA rB)))
=
 let kernel_constraints :
   squash (constraints bm bn bk tm tn tk wm wn) = () in
 mk_kernel_arithmetic_facts
   m n bm bn bk tm tn tk wm wn
   #kernel_constraints #bm_div_m #bn_div_n nblk nthr;
 let block_divides =
   squash_and
     #(bm /?+ m) #(bn /?+ n)
     #bm_div_m #bn_div_n in
 let scratch_fits =
   scratch_fits_of_kernel_constraints
     m n bm bn bk tm tn tk wm wn
     #mn_fits #bm_div_m #bn_div_n nthr in
 let warp_divides =
   warp_divides_thread_count bm bn tm tn wm wn nthr in
{
  nblk;
  nthr;

  shmems_desc = shmems_desc_to et_ab et_acc bm bn bk tm tn nthr
    #shared_fits #scratch_fits #warp_divides;

  barrier_contract = (fun bid ptrs -> FB.contract eA eB (rm bm bk) (rm bk bn) (fst ptrs) (fst (snd ptrs)) nthr bid);
  barrier_count    = (fun _bid -> 2 * (SZ.v k / SZ.v bk));
  barrier_ok = (fun bid ptrs -> FB.barrier_p_to_q_transform eA eB (rm bm bk) (rm bk bn) (fst ptrs) (fst (snd ptrs)) nthr bid);

  frame = pure (SZ.fits ((rm m n).ulen));
  block_pre  = (fun bid -> forall+ (tid : natlt nthr).
    kpre1_to gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn #block_divides fA fB fC rA rB rC
      nblk nthr bid tid);
  block_post = (fun bid -> forall+ (tid : natlt nthr).
    kpost1_to comb_r gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn #block_divides fA fB fC rA rB rC
      nblk nthr bid tid);

  setup = setup_to
    gA eA gB eB #_ #_ gC eC gD #mn_fits
    bm bn bk tm tn tk wm wn #block_divides nblk nthr
    fA fB fC rA rB rC;
  teardown = teardown_to comb_r
    gA eA gB eB gC eC gD #mn_fits
    bm bn bk tm tn tk wm wn #block_divides nblk nthr
    fA fB fC rA rB rC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup = block_setup_to
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn
    #block_divides #shared_fits #scratch_fits #warp_divides
    fA fB fC rA rB rC nblk nthr;
  block_teardown = block_teardown_to comb_r
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn
    #block_divides #shared_fits #scratch_fits #warp_divides
    fA fB fC rA rB rC nblk nthr;

  kpre = kpre_to
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn
    #block_divides #shared_fits #scratch_fits #warp_divides
    fA fB fC rA rB rC nblk nthr;
  kpost = kpost_to comb_r
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn
    #block_divides #shared_fits #scratch_fits #warp_divides
    fA fB fC rA rB rC nblk nthr;

  f = kf comb comb_r
    gA #eA gB #eB gC #eC gD
    bm bn bk tm tn tk wm wn
    rA rB rC nblk nthr #warp_divides;

  block_pre_sendable=solve;
  block_post_sendable=solve;
  kpre_sendable = (fun sh sh_inv _bid _tid ->
    let base_send = kpre1_sendable_to
      gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn #block_divides
      fA fB fC rA rB rC nblk nthr _bid _tid in
    shared_thread_frame_sendable_to #_ #base_send
      bm bn bk tm tn nthr
      #shared_fits #scratch_fits #warp_divides sh #sh_inv _tid);
  kpost_sendable = (fun sh sh_inv _bid _tid ->
    let base_send = kpost1_sendable_to comb_r
      gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn #block_divides
      fA fB fC rA rB rC nblk nthr _bid _tid in
    shared_thread_frame_sendable_to #_ #base_send
      bm bn bk tm tn nthr
      #shared_fits #scratch_fits #warp_divides sh #sh_inv _tid);
}
#pop-options
