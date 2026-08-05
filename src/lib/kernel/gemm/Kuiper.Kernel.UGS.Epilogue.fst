module Kuiper.Kernel.UGS.Epilogue

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Float.Casts.Base
open Kuiper.EMatrix

module SZ = Kuiper.SizeT
module Math = FStar.Math.Lemmas

(* Proof of [epilogue_cell_approx]: walks [epilogue_cell]'s definition
   step by step, calling the matching [Kuiper.Approximates.Base] algebra
   lemma at each step (each of which also has an [SMTPat]-triggered
   variant, so most of these calls are redundant with automatic firing,
   but are spelled out explicitly for robustness/readability, mirroring
   this codebase's existing style e.g. in [Kuiper.Kernel.RowSoftmax]). *)
let epilogue_cell_approx (gate_h up_h : half) (gR uR : real)
  : Lemma (requires gate_h %~ gR /\ up_h %~ uR)
          (ensures epilogue_cell gate_h up_h %~ real_epilogue_cell gR uR)
  = let g : float = cast_f16_to_f32 gate_h in
    cast_f16_to_f32_ok gate_h gR;
    assert (v_approximates (zero #float) 0.0R);
    assert (v_approximates (one #float) 1.0R);
    sub_approx (zero #float) g 0.0R gR;
    let negG : float = zero `sub` g in
    exp_approx negG (0.0R -. gR);
    assert (exp (0.0R -. gR) >. 0.0R);
    a_add (one #float) (fexp negG) 1.0R (exp (0.0R -. gR));
    let denom : float = add (one #float) (fexp negG) in
    div_approx (one #float) denom 1.0R (1.0R +. exp (0.0R -. gR));
    let sgm : float = div (one #float) denom in
    a_mul g sgm gR (1.0R /. (1.0R +. exp (0.0R -. gR)));
    let silu_f : float = mul g sgm in
    cast_f32_to_f16_ok silu_f (gR *. (1.0R /. (1.0R +. exp (0.0R -. gR))));
    let silu_h : half = cast_f32_to_f16 silu_f in
    a_mul silu_h up_h (gR *. (1.0R /. (1.0R +. exp (0.0R -. gR)))) uR

(* If [n > 0] and [k < m * n] then [k / n < m]. (Same shape as
   [Kuiper.Kernel.UGS.Round]'s private helper of the same purpose; kept
   local here since that one isn't exposed outside its own module.) *)
let lemma_div_bound (k m n : nat)
  : Lemma (requires n > 0 /\ k < m * n)
          (ensures k / n < m)
  = if m = 0 then ()
    else begin
      Math.small_div (n - 1) n;
      Math.lemma_div_plus (n - 1) (m - 1) n;
      Math.lemma_div_le k (m * n - 1) n
    end

(* If [n] is a positive multiple of [pair_group_n] and [c < n], then
   [c / pair_group_n < n / pair_group_n]. *)
let lemma_group_lt (n c : nat)
  : Lemma (requires n % pair_group_n == 0 /\ n > 0 /\ c < n)
          (ensures c / pair_group_n < n / pair_group_n)
  = let ng = n / pair_group_n in
    Math.lemma_div_mod n pair_group_n;
    lemma_div_bound c ng pair_group_n

let lemma_gate_up_bound (n c : nat)
  : Lemma (requires n % pair_group_n == 0 /\ c < n)
          (ensures gate_col_of c < 2 * n /\ up_col_of c < 2 * n)
  = lemma_group_lt n c;
    Math.lemma_div_mod n pair_group_n;
    Math.lemma_mod_lt c pair_group_n

(* Proof of [epilogue_matrix_approx]: for each output cell, extract the
   gate/up cells' individual approximates facts from [vP %~ rP] (a
   forall-quantified fact at [Kuiper.Chest]'s generic flattened [abs]
   index, instantiated here at concrete row/gate-col and row/up-col
   pairs), feed them to [epilogue_cell_approx], and collect the result
   with [Kuiper.EMatrix.lemma_approximates_intro]. *)
let epilogue_matrix_approx
  (#m : nat) (n : nat { n % pair_group_n == 0 })
  (vP : chest2 half m (2 * n))
  (rP : chest2 real m (2 * n))
  (eOut : chest2 half m n)
  : Lemma (requires
             vP %~ rP /\
             (forall (r : natlt m) (c : natlt n).
                acc2 eOut r c ==
                  epilogue_cell (acc2 vP r (gate_col_idx n c)) (acc2 vP r (up_col_idx n c))))
          (ensures eOut %~ real_epilogue n rP)
  = introduce forall (r : natlt m) (c : natlt n).
      acc2 eOut r c %~ acc2 (real_epilogue n rP) r c
    with (
      let gc = gate_col_idx n c in
      let uc = up_col_idx n c in
      assert (acc2 vP r gc %~ acc2 rP r gc);
      assert (acc2 vP r uc %~ acc2 rP r uc);
      epilogue_cell_approx (acc2 vP r gc) (acc2 vP r uc) (acc2 rP r gc) (acc2 rP r uc)
    );
    lemma_approximates_intro eOut (real_epilogue n rP)

(* [converted_before n2 k r c] holds iff the row-major-flattened index of
   cell [(r,c)] (in the [n]-wide output) is strictly less than the linear
   loop counter [k], i.e. iff cell [(r,c)] has already been overwritten by
   the epilogue by the time the loop's counter reaches [k]. *)
let converted_before (n k r c : nat) : bool =
  r * n + c < k

let converted_before_last (m n r c : nat)
  : Lemma (requires n > 0 /\ r < m /\ c < n)
          (ensures converted_before n (m * n) r c == true)
  = Math.lemma_mult_le_left n r (m - 1);
    Math.distributivity_sub_right n m 1

let epilogue_final_forall
  (m n n2 : nat) (vOut : chest2 half m n) (vP : chest2 half m n2) (vOut0 : chest2 half m n)
  (pf : squash (n2 == 2 * n /\ n % pair_group_n == 0))
  : Lemma
      (requires
        (forall (r : natlt m) (c : natlt n).
           acc2 vOut r c ==
             (if converted_before n (m * n) r c
              then epilogue_cell (acc2 vP r (gate_col_idx n c)) (acc2 vP r (up_col_idx n c))
              else acc2 vOut0 r c)))
      (ensures
        forall (r : natlt m) (c : natlt n).
          acc2 vOut r c ==
            epilogue_cell (acc2 vP r (gate_col_idx n c)) (acc2 vP r (up_col_idx n c)))
  = introduce forall (r : natlt m) (c : natlt n).
      acc2 vOut r c == epilogue_cell (acc2 vP r (gate_col_idx n c)) (acc2 vP r (up_col_idx n c))
    with converted_before_last m n r c

inline_for_extraction noextract
fn epilogue
  (#m #n #n2 : szp)
  (#_ : squash (SZ.v n2 == 2 * SZ.v n))
  (#_ : squash (SZ.v n % pair_group_n == 0))
  (#_ : squash (SZ.fits (SZ.v m * SZ.v n)))
  (gP : array2 half (rm m n2) { is_global gP })
  (gOut : array2 half (rm m n) { is_global gOut })
  (#fP : perm)
  (#vP : chest2 half (SZ.v m) (SZ.v n2))
  (#vOut0 : chest2 half (SZ.v m) (SZ.v n))
  requires
    gpu ** gP |-> Frac fP vP ** gOut |-> vOut0
  ensures
    gpu ** gP |-> Frac fP vP **
    (exists* (vOut : chest2 half (SZ.v m) (SZ.v n)).
      gOut |-> vOut **
      pure (
        forall (r : natlt (SZ.v m)) (c : natlt (SZ.v n)).
          acc2 vOut r c ==
            epilogue_cell
              (acc2 vP r (gate_col_idx (SZ.v n) c))
              (acc2 vP r (up_col_idx (SZ.v n) c))))
{
  let mut k : sz = 0sz;
  let kend = m *^ n;

  while (!k <^ kend)
    invariant
      gpu ** gP |-> Frac fP vP **
      live k **
      pure (!k <= kend) **
      (exists* (vOut : chest2 half m n).
        gOut |-> vOut **
        pure (
          forall (r : natlt m) (c : natlt n).
            acc2 vOut r c ==
              (if converted_before n !k r c
               then epilogue_cell (acc2 vP r (gate_col_idx n c)) (acc2 vP r (up_col_idx n c))
               else acc2 vOut0 r c)))
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
    let group : sz = ci /^ 64sz;
    let pos : sz = ci %^ 64sz;
    let gc0 : sz = (128sz *^ group) +^ pos;
    let uc0 : sz = gc0 +^ 64sz;
    lemma_gate_up_bound n ci;
    assert pure (SZ.v gc0 == gate_col_of ci);
    assert pure (SZ.v uc0 == up_col_of ci);
    assert pure (SZ.v gc0 < SZ.v n2);
    assert pure (SZ.v uc0 < SZ.v n2);
    let gc : szlt n2 = gc0;
    let uc : szlt n2 = uc0;
    let gate_h = tensor_read gP (cidx2 ri gc);
    let up_h = tensor_read gP (cidx2 ri uc);
    let y = epilogue_cell gate_h up_h;
    tensor_write gOut (cidx2 ri ci) y;
    k := kv +^ 1sz;
    dassert (!k <=^ kend);
  };

  with vOut. assert (gOut |-> vOut);
  epilogue_final_forall m n n2 vOut vP vOut0 ();
}
