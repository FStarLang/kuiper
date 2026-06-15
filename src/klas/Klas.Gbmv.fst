module Klas.Gbmv

(* cuBLAS gbmv (no transpose) : general banded matrix-vector product (in place)

       y := alpha * A * x + beta * y,

   with A an n x n GENERAL BAND matrix with kl sub-diagonals and ku
   super-diagonals, stored in the cuBLAS column-major band layout (leading dim
   kl+ku+1): A(i,j) with i-kl<=j<=i+ku at AB[(ku+i-j) + j*(kl+ku+1)] =
   AB[ku + i + j*(kl+ku)]. AB has length np = (kl+ku+1)*n; x, y are length n.

       y[i] = alpha * (sum_{i-kl<=j<=i+ku} AB[ku+i+j*(kl+ku)] * x[j]) + beta*y[i].

   This is the square (m = n) case of gbmv. The band index ku+i+j*(kl+ku) is
   <= np-(kl+1) < np for every i,j < n, so reads are unconditional and the band
   condition (i<=j+kl && j<=i+ku) only gates the term; a single running index
   with constant increment kl+ku tracks it. In place; output is a full vector. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* ----------------------------------------------------------------------- *)
(* Band index and its bound.                                                 *)
(* ----------------------------------------------------------------------- *)

noextract
let gb_index (kl ku i j : nat) : nat = ku + i + j * (kl + ku)

let gb_bound (n kl ku i j : nat)
  : Lemma (requires i < n /\ j < n) (ensures gb_index kl ku i j < (kl + ku + 1) * n)
  = ML.lemma_mult_le_right (kl + ku) j (n - 1);
    ML.lemma_mult_le_right (kl + ku + 1) n n

(* Guarded form, callable for j up to n: only the in-range case bounds. *)
let gb_bound_g (n kl ku i j : nat)
  : Lemma (requires i < n) (ensures j < n ==> gb_index kl ku i j < (kl + ku + 1) * n)
  = if j < n then gb_bound n kl ku i j

noextract
let inband (kl ku i j : nat) : bool = (i <= j + kl) && (j <= i + ku)

(* ----------------------------------------------------------------------- *)
(* Spec: general banded row dot and the output cell.                         *)
(* ----------------------------------------------------------------------- *)

let rec fgb_dot (#et:Type0) {| floating et |}
  (n:nat) (kl ku : nat) (np:nat{np == (kl + ku + 1) * n}) (sA : lseq et np) (sx : lseq et n)
  (i:nat{i < n}) (m:nat{m <= n})
  : Tot et (decreases m)
  = if m = 0 then zero
    else (let j = m - 1 in
          let term : et =
            if inband kl ku i j then (gb_bound n kl ku i j; (Seq.index sA (gb_index kl ku i j)) `mul` (Seq.index sx j))
            else zero in
          (fgb_dot n kl ku np sA sx i (m - 1)) `add` term)

let fgbmv_at (#et:Type0) {| floating et |}
  (n:nat) (kl ku : nat) (np:nat{np == (kl + ku + 1) * n}) (alpha beta : et)
  (sA : lseq et np) (sx sy0 : lseq et n) (i:nat{i < n}) : et
  = (alpha `mul` (fgb_dot n kl ku np sA sx i n)) `add` (beta `mul` (Seq.index sy0 i))

noextract
let fgbmv (#et:Type0) {| floating et |}
  (n:nat) (kl ku : nat) (np:nat{np == (kl + ku + 1) * n}) (alpha beta : et)
  (sA : lseq et np) (sx sy0 : lseq et n) : lseq et n
  = Seq.init n (fun (i:nat{i < n}) -> fgbmv_at n kl ku np alpha beta sA sx sy0 i)

let fgbmv_index (#et:Type0) {| floating et |}
  (n:nat) (kl ku : nat) (np:nat{np == (kl + ku + 1) * n}) (alpha beta : et)
  (sA : lseq et np) (sx sy0 : lseq et n) (i:nat{i < n})
  : Lemma (ensures Seq.index (fgbmv n kl ku np alpha beta sA sx sy0) i
                   == fgbmv_at n kl ku np alpha beta sA sx sy0 i)
          [SMTPat (Seq.index (fgbmv n kl ku np alpha beta sA sx sy0) i)]
  = ()

(* ----------------------------------------------------------------------- *)
(* Per-row general banded dot (running band index bi == ku + i + j*(kl+ku)). *)
(* ----------------------------------------------------------------------- *)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 200"
inline_for_extraction noextract
fn gbmv_row
  (#et:Type0) {| floating et |}
  (#n : szp)
  (kl ku : sz)
  (np : szp { SZ.v np == (SZ.v kl + SZ.v ku + 1) * SZ.v n })
  (gA : array1 et (l1_forward np))
  (gx : array1 et (l1_forward n))
  (i : szlt n)
  (#sA : erased (lseq et (SZ.v np)))
  (#sx : erased (lseq et (SZ.v n)))
  (#fa #fx : perm)
  preserves
    gpu ** gA |-> Frac fa sA ** gx |-> Frac fx sx
  returns
    s : et
  ensures
    pure (s == fgb_dot (SZ.v n) (SZ.v kl) (SZ.v ku) (SZ.v np) sA sx (SZ.v i) (SZ.v n))
{
  let mut s : et = zero;
  let mut j : sz = 0sz;
  let mut bi : sz = ku +^ i;

  while (!j <^ n)
    invariant live s
    invariant live j
    invariant live bi
    invariant pure (SZ.v !j <= SZ.v n /\
                    (SZ.v !j < SZ.v n ==> SZ.v !bi == gb_index (SZ.v kl) (SZ.v ku) (SZ.v i) (SZ.v !j)) /\
                    !s == fgb_dot (SZ.v n) (SZ.v kl) (SZ.v ku) (SZ.v np) sA sx (SZ.v i) (SZ.v !j))
    decreases (SZ.v n - SZ.v !j)
  {
    let vj = !j;
    let z : et = zero;
    let vbi = !bi;
    gb_bound (SZ.v n) (SZ.v kl) (SZ.v ku) (SZ.v i) (SZ.v vj);
    let a = Array1.(gA.(vbi));
    let xj = Array1.(gx.(vj));
    let cond = (i <=^ (vj +^ kl)) && (vj <=^ (i +^ ku));
    let term = if cond { a `mul` xj } else { z };
    s := !s `add` term;
    gb_bound_g (SZ.v n) (SZ.v kl) (SZ.v ku) (SZ.v i) (SZ.v vj + 1);
    let nb = if ((vj +^ 1sz) <^ n) { (vbi +^ (kl +^ ku) <: sz) } else { vbi };
    bi := nb;
    j := !j +^ 1sz;
  };
  !s
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: one row per output entry, in place.                 *)
(* ----------------------------------------------------------------------- *)

let gbmv_upd
  (#et:Type0) {| floating et |}
  (n:nat) (kl ku : nat) (np:nat{np == (kl + ku + 1) * n}) (alpha beta : et)
  (sA : lseq et np) (sx sy0 : lseq et n) (sy : lseq et n) (i:nat{i < n}) (yi : et)
  : Lemma
      (requires
        (forall (l:nat). l < i ==> Seq.index sy l == fgbmv_at n kl ku np alpha beta sA sx sy0 l) /\
        (forall (l:nat). i <= l /\ l < n ==> Seq.index sy l == Seq.index sy0 l) /\
        yi == fgbmv_at n kl ku np alpha beta sA sx sy0 i)
      (ensures
        (let sy' = Seq.upd sy i yi in
         (forall (l:nat). l < i + 1 ==> Seq.index sy' l == fgbmv_at n kl ku np alpha beta sA sx sy0 l) /\
         (forall (l:nat). i + 1 <= l /\ l < n ==> Seq.index sy' l == Seq.index sy0 l)))
  = let sy' = Seq.upd sy i yi in
    introduce forall (l:nat). l < i + 1 ==> Seq.index sy' l == fgbmv_at n kl ku np alpha beta sA sx sy0 l
    with introduce _ ==> _
    with _. (if l < i then Seq.lemma_index_upd2 sy i yi l
             else Seq.lemma_index_upd1 sy i yi);
    introduce forall (l:nat). i + 1 <= l /\ l < n ==> Seq.index sy' l == Seq.index sy0 l
    with introduce _ ==> _
    with _. Seq.lemma_index_upd2 sy i yi l

#push-options "--fuel 2 --ifuel 1 --z3rlimit 150"
inline_for_extraction noextract
fn gbmv_kf
  (#et:Type0) {| floating et |}
  (#n : szp)
  (kl ku : sz)
  (np : szp { SZ.v np == (SZ.v kl + SZ.v ku + 1) * SZ.v n })
  (alpha beta : et)
  (gA : array1 et (l1_forward np))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (#sA : erased (lseq et (SZ.v np)))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy0 : erased (lseq et (SZ.v n)))
  (#fa #fx : perm)
  ()
  norewrite
  requires
    gpu ** gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0
  ensures
    gpu ** gA |-> Frac fa sA ** gx |-> Frac fx sx **
    gy |-> fgbmv (SZ.v n) (SZ.v kl) (SZ.v ku) (SZ.v np) alpha beta sA sx sy0
{
  let mut i : szle n = 0sz;

  while (!i <^ n)
    invariant live i
    invariant exists* (sy : lseq et (SZ.v n)). gy |-> sy **
      pure (SZ.v !i <= SZ.v n /\
            (forall (l:nat). l < SZ.v !i ==>
               Seq.index sy l == fgbmv_at (SZ.v n) (SZ.v kl) (SZ.v ku) (SZ.v np) alpha beta sA sx sy0 l) /\
            (forall (l:nat). SZ.v !i <= l /\ l < SZ.v n ==>
               Seq.index sy l == Seq.index sy0 l))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let s = gbmv_row kl ku np gA gx vi;
    let yi_old = Array1.(gy.(vi));
    let yi = (alpha `mul` s) `add` (beta `mul` yi_old);
    with syp. assert (gy |-> syp);
    Array1.(gy.(vi) <- yi);
    gbmv_upd (SZ.v n) (SZ.v kl) (SZ.v ku) (SZ.v np) alpha beta sA sx sy0 syp (SZ.v vi) yi;
    i := !i +^ 1sz;
  };

  with syf. assert (gy |-> syf);
  Seq.lemma_eq_intro syf (fgbmv (SZ.v n) (SZ.v kl) (SZ.v ku) (SZ.v np) alpha beta sA sx sy0);
  ()
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper (single-thread kernel_desc_n, like Klas.Sbmv).           *)
(* ----------------------------------------------------------------------- *)

ghost
fn gbmv_setup
  (#et:Type0) {| floating et |}
  (#n : szp)
  (kl ku : sz)
  (np : szp { SZ.v np == (SZ.v kl + SZ.v ku + 1) * SZ.v n })
  (gA : array1 et (l1_forward np))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (#sA : erased (lseq et (SZ.v np)))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy0 : erased (lseq et (SZ.v n)))
  (#fa #fx : perm)
  ()
  norewrite
  requires gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0
  ensures (forall+ (tid : natlt 1sz).
             gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0) ** emp
{
  forevery_singleton_intro #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0);
}

ghost
fn gbmv_teardown
  (#et:Type0) {| floating et |}
  (#n : szp)
  (kl ku : sz)
  (np : szp { SZ.v np == (SZ.v kl + SZ.v ku + 1) * SZ.v n })
  (alpha beta : et)
  (gA : array1 et (l1_forward np))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (#sA : erased (lseq et (SZ.v np)))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy0 : erased (lseq et (SZ.v n)))
  (#fa #fx : perm)
  ()
  norewrite
  requires
    (forall+ (tid : natlt 1sz).
       gA |-> Frac fa sA ** gx |-> Frac fx sx **
       gy |-> fgbmv (SZ.v n) (SZ.v kl) (SZ.v ku) (SZ.v np) alpha beta sA sx sy0) ** emp
  ensures
    gA |-> Frac fa sA ** gx |-> Frac fx sx **
    gy |-> fgbmv (SZ.v n) (SZ.v kl) (SZ.v ku) (SZ.v np) alpha beta sA sx sy0
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx **
       gy |-> fgbmv (SZ.v n) (SZ.v kl) (SZ.v ku) (SZ.v np) alpha beta sA sx sy0);
}

inline_for_extraction noextract
let kamgbmv
  (#et:Type0) {| floating et |}
  (n : szp)
  (kl ku : sz)
  (np : szp { SZ.v np == (SZ.v kl + SZ.v ku + 1) * SZ.v n })
  (alpha beta : et)
  (gA : array1 et (l1_forward np))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (#_ : squash (Array1.is_global gA /\ Array1.is_global gx /\ Array1.is_global gy))
  (#sA : erased (lseq et (SZ.v np)))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy0 : erased (lseq et (SZ.v n)))
  (#fa #fx : perm)
  : kernel_desc
      (requires gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0)
      (ensures  gA |-> Frac fa sA ** gx |-> Frac fx sx **
                gy |-> fgbmv (SZ.v n) (SZ.v kl) (SZ.v ku) (SZ.v np) alpha beta sA sx sy0)
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> gbmv_kf kl ku np alpha beta gA gx gy #sA #sx #sy0 #fa #fx);
    frame    = emp;
    teardown = gbmv_teardown kl ku np alpha beta gA gx gy;
    setup    = gbmv_setup kl ku np gA gx gy;
    kpre  = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0);
    kpost = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx **
                                     gy |-> fgbmv (SZ.v n) (SZ.v kl) (SZ.v ku) (SZ.v np) alpha beta sA sx sy0);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn gbmv_gen
  (#et:Type0) {| floating et |}
  (n : szp)
  (kl ku : sz)
  (np : szp { SZ.v np == (SZ.v kl + SZ.v ku + 1) * SZ.v n })
  (alpha beta : et)
  (gA : array1 et (l1_forward np) { Array1.is_global gA })
  (gx : array1 et (l1_forward n) { Array1.is_global gx })
  (gy : array1 et (l1_forward n) { Array1.is_global gy })
  (#sA : erased (lseq et (SZ.v np)))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy0 : erased (lseq et (SZ.v n)))
  (#fa #fx : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fa sA) **
    on gpu_loc (gx |-> Frac fx sx)
  requires
    on gpu_loc (gy |-> sy0)
  ensures
    on gpu_loc (gy |-> fgbmv (SZ.v n) (SZ.v kl) (SZ.v ku) (SZ.v np) alpha beta sA sx sy0)
{
  on_star_eq gpu_loc (gx |-> Frac fx sx) (gy |-> sy0);
  rewrite (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc (gy |-> sy0))
       as (on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> sy0)));
  on_star_eq gpu_loc (gA |-> Frac fa sA) ((gx |-> Frac fx sx) ** (gy |-> sy0));
  rewrite (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> sy0)))
       as (on gpu_loc ((gA |-> Frac fa sA) ** ((gx |-> Frac fx sx) ** (gy |-> sy0))));

  launch_sync (kamgbmv n kl ku np alpha beta gA gx gy #() #sA #sx #sy0 #fa #fx);

  on_star_eq gpu_loc (gA |-> Frac fa sA)
             ((gx |-> Frac fx sx) ** (gy |-> fgbmv (SZ.v n) (SZ.v kl) (SZ.v ku) (SZ.v np) alpha beta sA sx sy0));
  rewrite (on gpu_loc ((gA |-> Frac fa sA) ** ((gx |-> Frac fx sx) ** (gy |-> fgbmv (SZ.v n) (SZ.v kl) (SZ.v ku) (SZ.v np) alpha beta sA sx sy0))))
       as (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> fgbmv (SZ.v n) (SZ.v kl) (SZ.v ku) (SZ.v np) alpha beta sA sx sy0)));
  on_star_eq gpu_loc (gx |-> Frac fx sx) (gy |-> fgbmv (SZ.v n) (SZ.v kl) (SZ.v ku) (SZ.v np) alpha beta sA sx sy0);
  rewrite (on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> fgbmv (SZ.v n) (SZ.v kl) (SZ.v ku) (SZ.v np) alpha beta sA sx sy0)))
       as (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc (gy |-> fgbmv (SZ.v n) (SZ.v kl) (SZ.v ku) (SZ.v np) alpha beta sA sx sy0));
  ()
}

let gbmv_f32 = gbmv_gen #f32
let gbmv_f64 = gbmv_gen #f64
