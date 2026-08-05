module Kuiper.Kernel.UGS.Round

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Float.Casts.Base
open Kuiper.EMatrix

module SZ = Kuiper.SizeT
module Math = FStar.Math.Lemmas

(* See [Kuiper.Kernel.UGS.Round.fsti] for the doc comment. Every cell of
   [vPf32] approximating the corresponding cell of [rM] (via [vPf32 %~
   rM]'s [chest_approximates] definition, which quantifies over the
   generic flattened [abs] index) gives, for any concrete row/col pair,
   [acc2 vPf32 r c %~ acc2 rM r c] by simple forall-instantiation.
   [Kuiper.Float.Casts.Base.cast_f32_to_f16_ok] (an [SMTPat]-triggered
   lemma) then turns each such fact into
   [cast_f32_to_f16 (acc2 vPf32 r c) %~ acc2 rM r c], i.e. (since
   [chest_map]'s [acc2] reduces via [Kuiper.Chest.acc_pat]'s SMTPat)
   [acc2 (chest_map cast_f32_to_f16 vPf32) r c %~ acc2 rM r c]. Collecting
   this over all [(r, c)] and feeding it to
   [Kuiper.EMatrix.lemma_approximates_intro] gives the matrix-level
   conclusion. *)
let round_preserves_approx (#m #n : nat) (vPf32 : chest2 float m n) (rM : chest2 real m n)
  : Lemma (requires vPf32 %~ rM)
          (ensures chest_map cast_f32_to_f16 vPf32 %~ rM)
  = introduce forall (r : natlt m) (c : natlt n).
      acc2 (chest_map cast_f32_to_f16 vPf32) r c %~ acc2 rM r c
    with (
      assert (acc2 vPf32 r c %~ acc2 rM r c);
      cast_f32_to_f16_ok (acc2 vPf32 r c) (acc2 rM r c)
    );
    lemma_approximates_intro (chest_map cast_f32_to_f16 vPf32) rM

(* [converted_before n k r c] holds iff the row-major-flattened index of
   cell [(r,c)] (with row length [n]) is strictly less than the linear
   loop counter [k], i.e. iff cell [(r,c)] has already been converted by
   the time the round loop's counter reaches [k]. *)
let converted_before (n k r c : nat) : bool =
  r * n + c < k

(* If [k < m * n] and [n > 0] then [k / n < m]. Used to justify that the
   quotient of the linear loop counter is a valid row index. *)
let lemma_div_bound (k m n : nat)
  : Lemma (requires n > 0 /\ k < m * n)
          (ensures k / n < m)
  = if m = 0 then ()
    else begin
      Math.small_div (n - 1) n;
      Math.lemma_div_plus (n - 1) (m - 1) n;
      Math.lemma_div_le k (m * n - 1) n
    end

(* Every cell [(r,c)] of an [m]-by-[n] chest has been "converted before"
   the linear counter reaches [m * n] (i.e. after the loop has run to
   completion). Kept as a standalone (non-Pulse) lemma, per this
   codebase's convention (see e.g. [Kuiper.Kernel.GEMM.BlockTiling2D]'s
   comment on why such forall-facts are proved outside imperative Pulse
   code) of proving quantified facts as ordinary F* lemmas rather than
   inline [introduce forall] steps inside Pulse function bodies. *)
let converted_before_last (m n r c : nat)
  : Lemma (requires n > 0 /\ r < m /\ c < n)
          (ensures converted_before n (m * n) r c == true)
  = Math.lemma_mult_le_left n r (m - 1);
    Math.distributivity_sub_right n m 1

(* Once the round loop's linear counter has reached [m * n] (i.e. the loop
   has finished), the pointwise invariant property implies every cell of
   [vP] equals the corresponding cell of [chest_map cast_f32_to_f16 vPf32]. *)
let round_final_forall (m n : nat) (vP : chest2 half m n) (vPf32 : chest2 float m n) (vP0 : chest2 half m n)
  : Lemma
      (requires
        (forall (r : natlt m) (c : natlt n).
           acc2 vP r c ==
             (if converted_before n (m * n) r c
              then cast_f32_to_f16 (acc2 vPf32 r c)
              else acc2 vP0 r c)))
      (ensures forall (r : natlt m) (c : natlt n). acc2 vP r c == acc2 (chest_map cast_f32_to_f16 vPf32) r c)
  = introduce forall (r : natlt m) (c : natlt n).
      acc2 vP r c == acc2 (chest_map cast_f32_to_f16 vPf32) r c
    with converted_before_last m n r c

(* Restatement of [round_final_forall]'s conclusion using the flattened
   [abs]-indexed [acc] (rather than the curried [acc2]), which is the
   exact shape [Kuiper.Chest.lemma_equal_intro] needs. *)
let round_final_pointwise (m n : nat) (vP : chest2 half m n) (vPf32 : chest2 float m n) (vP0 : chest2 half m n)
  : Lemma
      (requires
        (forall (r : natlt m) (c : natlt n).
           acc2 vP r c ==
             (if converted_before n (m * n) r c
              then cast_f32_to_f16 (acc2 vPf32 r c)
              else acc2 vP0 r c)))
      (ensures
        forall (i : Kuiper.Shape.abs (Kuiper.Shape.ICons m (Kuiper.Shape.ICons n Kuiper.Shape.INil))).
          acc vP i == acc (chest_map cast_f32_to_f16 vPf32) i)
  = round_final_forall m n vP vPf32 vP0;
    introduce forall (i : Kuiper.Shape.abs (Kuiper.Shape.ICons m (Kuiper.Shape.ICons n Kuiper.Shape.INil))).
      acc vP i == acc (chest_map cast_f32_to_f16 vPf32) i
    with (
      let (r, (c, ())) = i in
      assert (acc2 vP r c == acc2 (chest_map cast_f32_to_f16 vPf32) r c)
    )

inline_for_extraction noextract
fn round
  (#m #n : szp)
  (#_ : squash (SZ.fits (SZ.v m * SZ.v n)))
  (gPf32 : array2 float (rm m n) { is_global gPf32 })
  (gP    : array2 half  (rm m n) { is_global gP })
  (#fP32 : perm)
  (#vPf32 : chest2 float (SZ.v m) (SZ.v n))
  (#vP0 : chest2 half (SZ.v m) (SZ.v n))
  requires
    gpu ** gPf32 |-> Frac fP32 vPf32 ** gP |-> vP0
  ensures
    gpu ** gPf32 |-> Frac fP32 vPf32 ** gP |-> chest_map cast_f32_to_f16 vPf32
{
  let mut k : sz = 0sz;
  let kend = m *^ n;

  while (!k <^ kend)
    invariant
      gpu ** gPf32 |-> Frac fP32 vPf32 **
      live k **
      pure (!k <= kend) **
      (exists* (vP : chest2 half m n).
        gP |-> vP **
        pure (
          forall (r : natlt m) (c : natlt n).
            acc2 vP r c ==
              (if converted_before n !k r c
               then cast_f32_to_f16 (acc2 vPf32 r c)
               else acc2 vP0 r c)))
    decreases (kend - !k)
  {
    let kv = !k;
    assert pure (kv < kend);
    assert pure (kv < m * n);
    let ri : sz = kv /^ n;
    let ci : szlt n = kv %^ n;
    lemma_div_bound kv m n;
    assert pure (ri < m);
    let ri : szlt m = ri;
    let x = tensor_read gPf32 (cidx2 ri ci);
    let y = cast_f32_to_f16 x;
    tensor_write gP (cidx2 ri ci) y;
    k := kv +^ 1sz;
    dassert (!k <=^ kend);
  };

  with vP. assert (gP |-> vP);
  round_final_pointwise m n vP vPf32 vP0;
  Kuiper.Chest.lemma_equal_intro vP (chest_map cast_f32_to_f16 vPf32);
  Kuiper.Chest.ext vP (chest_map cast_f32_to_f16 vPf32);
  rewrite (gP |-> vP) as (gP |-> chest_map cast_f32_to_f16 vPf32);
}
