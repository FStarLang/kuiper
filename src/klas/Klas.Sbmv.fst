module Klas.Sbmv

(* cuBLAS sbmv (lower) : symmetric banded matrix-vector product (in place)

       y := alpha * A * x + beta * y,

   with A an n x n SYMMETRIC BAND matrix with k off-diagonals, of which only the
   lower band is stored in the cuBLAS column-major band layout: A(i,j) with j<=i,
   i-j<=k, at AB[i + j*k]; for j>i the symmetric entry A(i,j)=A(j,i) is at
   AB[j + i*k]. AB has length np = (k+1)*n; x, y are length n.

       y[i] = alpha * (sum_{|i-j|<=k} A_sym(i,j) * x[j]) + beta * y[i].

   Combines the symmetric reconstruction of Klas.Spmv with the band running
   index of Klas.Tbmv. The symmetric band index (j<=i ? i+j*k : j+i*k) stays
   < np for every j<n, so reads are unconditional and the band condition only
   gates the term. A single running index suffices (increment k while j<i,
   else 1). In place; output is a full vector. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* ----------------------------------------------------------------------- *)
(* Symmetric band index and its bound (any j, in or out of band).            *)
(* ----------------------------------------------------------------------- *)

noextract
let symb_index (k i j : nat) : nat = if j <= i then i + j * k else j + i * k

let symb_bound (n k i j : nat)
  : Lemma (requires i < n /\ j < n) (ensures symb_index k i j < (k + 1) * n)
  = if j <= i
    then (ML.lemma_mult_le_right k j i; ML.lemma_mult_le_right (k + 1) (i + 1) n)
    else (ML.lemma_mult_le_right k i (n - 1); ML.lemma_mult_le_right (k + 1) n n)

(* The post-increment index (j up to n) only needs to fit, i.e. be <= np. *)
let symb_le (n k i j : nat)
  : Lemma (requires i < n /\ j <= n) (ensures symb_index k i j <= (k + 1) * n)
  = if j <= i
    then (ML.lemma_mult_le_right k j i; ML.lemma_mult_le_right (k + 1) (i + 1) n)
    else (ML.lemma_mult_le_right k i (n - 1); ML.lemma_mult_le_right (k + 1) n n)

(* band membership: |i-j| <= k, written without subtraction. *)
noextract
let inband (k i j : nat) : bool = (i <= j + k) && (j <= i + k)

(* ----------------------------------------------------------------------- *)
(* Spec: symmetric banded row dot and the output cell.                       *)
(* ----------------------------------------------------------------------- *)

let rec fsb_dot (#et:Type0) {| floating et |}
  (n:nat) (k:nat) (np:nat{np == (k + 1) * n}) (sA : lseq et np) (sx : lseq et n)
  (i:nat{i < n}) (m:nat{m <= n})
  : Tot et (decreases m)
  = if m = 0 then zero
    else (let j = m - 1 in
          let term : et =
            if inband k i j then (symb_bound n k i j; (Seq.index sA (symb_index k i j)) `mul` (Seq.index sx j))
            else zero in
          (fsb_dot n k np sA sx i (m - 1)) `add` term)

let fsbmv_at (#et:Type0) {| floating et |}
  (n:nat) (k:nat) (np:nat{np == (k + 1) * n}) (alpha beta : et)
  (sA : lseq et np) (sx sy0 : lseq et n) (i:nat{i < n}) : et
  = (alpha `mul` (fsb_dot n k np sA sx i n)) `add` (beta `mul` (Seq.index sy0 i))

noextract
let fsbmv (#et:Type0) {| floating et |}
  (n:nat) (k:nat) (np:nat{np == (k + 1) * n}) (alpha beta : et)
  (sA : lseq et np) (sx sy0 : lseq et n) : lseq et n
  = Seq.init n (fun (i:nat{i < n}) -> fsbmv_at n k np alpha beta sA sx sy0 i)

let fsbmv_index (#et:Type0) {| floating et |}
  (n:nat) (k:nat) (np:nat{np == (k + 1) * n}) (alpha beta : et)
  (sA : lseq et np) (sx sy0 : lseq et n) (i:nat{i < n})
  : Lemma (ensures Seq.index (fsbmv n k np alpha beta sA sx sy0) i
                   == fsbmv_at n k np alpha beta sA sx sy0 i)
          [SMTPat (Seq.index (fsbmv n k np alpha beta sA sx sy0) i)]
  = ()

(* ----------------------------------------------------------------------- *)
(* Per-row symmetric banded dot (running symmetric band index bi).           *)
(* ----------------------------------------------------------------------- *)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 200"
inline_for_extraction noextract
fn sbmv_row
  (#et:Type0) {| floating et |}
  (#n : szp)
  (k : sz)
  (np : szp { SZ.v np == (SZ.v k + 1) * SZ.v n })
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
    pure (s == fsb_dot (SZ.v n) (SZ.v k) (SZ.v np) sA sx (SZ.v i) (SZ.v n))
{
  let mut s : et = zero;
  let mut j : sz = 0sz;
  let mut bi : sz = i;

  while (!j <^ n)
    invariant live s
    invariant live j
    invariant live bi
    invariant pure (SZ.v !j <= SZ.v n /\
                    SZ.v !bi == symb_index (SZ.v k) (SZ.v i) (SZ.v !j) /\
                    !s == fsb_dot (SZ.v n) (SZ.v k) (SZ.v np) sA sx (SZ.v i) (SZ.v !j))
    decreases (SZ.v n - SZ.v !j)
  {
    let vj = !j;
    let z : et = zero;
    let vbi = !bi;
    symb_bound (SZ.v n) (SZ.v k) (SZ.v i) (SZ.v vj);
    let a = Array1.(gA.(vbi));
    let xj = Array1.(gx.(vj));
    let cond = (i <=^ (vj +^ k)) && (vj <=^ (i +^ k));
    let term = if cond { a `mul` xj } else { z };
    s := !s `add` term;
    let inc = if (vj <^ i) { k } else { (1sz <: sz) };
    symb_le (SZ.v n) (SZ.v k) (SZ.v i) (SZ.v vj + 1);
    bi := vbi +^ inc;
    j := !j +^ 1sz;
  };
  !s
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: one row per output entry, in place.                 *)
(* ----------------------------------------------------------------------- *)

let sbmv_upd
  (#et:Type0) {| floating et |}
  (n:nat) (k:nat) (np:nat{np == (k + 1) * n}) (alpha beta : et)
  (sA : lseq et np) (sx sy0 : lseq et n) (sy : lseq et n) (i:nat{i < n}) (yi : et)
  : Lemma
      (requires
        (forall (l:nat). l < i ==> Seq.index sy l == fsbmv_at n k np alpha beta sA sx sy0 l) /\
        (forall (l:nat). i <= l /\ l < n ==> Seq.index sy l == Seq.index sy0 l) /\
        yi == fsbmv_at n k np alpha beta sA sx sy0 i)
      (ensures
        (let sy' = Seq.upd sy i yi in
         (forall (l:nat). l < i + 1 ==> Seq.index sy' l == fsbmv_at n k np alpha beta sA sx sy0 l) /\
         (forall (l:nat). i + 1 <= l /\ l < n ==> Seq.index sy' l == Seq.index sy0 l)))
  = let sy' = Seq.upd sy i yi in
    introduce forall (l:nat). l < i + 1 ==> Seq.index sy' l == fsbmv_at n k np alpha beta sA sx sy0 l
    with introduce _ ==> _
    with _. (if l < i then Seq.lemma_index_upd2 sy i yi l
             else Seq.lemma_index_upd1 sy i yi);
    introduce forall (l:nat). i + 1 <= l /\ l < n ==> Seq.index sy' l == Seq.index sy0 l
    with introduce _ ==> _
    with _. Seq.lemma_index_upd2 sy i yi l

#push-options "--fuel 2 --ifuel 1 --z3rlimit 150"
inline_for_extraction noextract
fn sbmv_kf
  (#et:Type0) {| floating et |}
  (#n : szp)
  (k : sz)
  (np : szp { SZ.v np == (SZ.v k + 1) * SZ.v n })
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
    gy |-> fsbmv (SZ.v n) (SZ.v k) (SZ.v np) alpha beta sA sx sy0
{
  let mut i : szle n = 0sz;

  while (!i <^ n)
    invariant live i
    invariant exists* (sy : lseq et (SZ.v n)). gy |-> sy **
      pure (SZ.v !i <= SZ.v n /\
            (forall (l:nat). l < SZ.v !i ==>
               Seq.index sy l == fsbmv_at (SZ.v n) (SZ.v k) (SZ.v np) alpha beta sA sx sy0 l) /\
            (forall (l:nat). SZ.v !i <= l /\ l < SZ.v n ==>
               Seq.index sy l == Seq.index sy0 l))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let s = sbmv_row k np gA gx vi;
    let yi_old = Array1.(gy.(vi));
    let yi = (alpha `mul` s) `add` (beta `mul` yi_old);
    with syp. assert (gy |-> syp);
    Array1.(gy.(vi) <- yi);
    sbmv_upd (SZ.v n) (SZ.v k) (SZ.v np) alpha beta sA sx sy0 syp (SZ.v vi) yi;
    i := !i +^ 1sz;
  };

  with syf. assert (gy |-> syf);
  Seq.lemma_eq_intro syf (fsbmv (SZ.v n) (SZ.v k) (SZ.v np) alpha beta sA sx sy0);
  ()
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper (single-thread kernel_desc_n, like Klas.Spmv).           *)
(* ----------------------------------------------------------------------- *)

ghost
fn sbmv_setup
  (#et:Type0) {| floating et |}
  (#n : szp)
  (k : sz)
  (np : szp { SZ.v np == (SZ.v k + 1) * SZ.v n })
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
fn sbmv_teardown
  (#et:Type0) {| floating et |}
  (#n : szp)
  (k : sz)
  (np : szp { SZ.v np == (SZ.v k + 1) * SZ.v n })
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
       gy |-> fsbmv (SZ.v n) (SZ.v k) (SZ.v np) alpha beta sA sx sy0) ** emp
  ensures
    gA |-> Frac fa sA ** gx |-> Frac fx sx **
    gy |-> fsbmv (SZ.v n) (SZ.v k) (SZ.v np) alpha beta sA sx sy0
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx **
       gy |-> fsbmv (SZ.v n) (SZ.v k) (SZ.v np) alpha beta sA sx sy0);
}

inline_for_extraction noextract
let kamsbmv
  (#et:Type0) {| floating et |}
  (n : szp)
  (k : sz)
  (np : szp { SZ.v np == (SZ.v k + 1) * SZ.v n })
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
                gy |-> fsbmv (SZ.v n) (SZ.v k) (SZ.v np) alpha beta sA sx sy0)
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> sbmv_kf k np alpha beta gA gx gy #sA #sx #sy0 #fa #fx);
    frame    = emp;
    teardown = sbmv_teardown k np alpha beta gA gx gy;
    setup    = sbmv_setup k np gA gx gy;
    kpre  = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0);
    kpost = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx **
                                     gy |-> fsbmv (SZ.v n) (SZ.v k) (SZ.v np) alpha beta sA sx sy0);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn sbmv_gen
  (#et:Type0) {| floating et |}
  (n : szp)
  (k : sz)
  (np : szp { SZ.v np == (SZ.v k + 1) * SZ.v n })
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
    on gpu_loc (gy |-> fsbmv (SZ.v n) (SZ.v k) (SZ.v np) alpha beta sA sx sy0)
{
  on_star_eq gpu_loc (gx |-> Frac fx sx) (gy |-> sy0);
  rewrite (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc (gy |-> sy0))
       as (on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> sy0)));
  on_star_eq gpu_loc (gA |-> Frac fa sA) ((gx |-> Frac fx sx) ** (gy |-> sy0));
  rewrite (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> sy0)))
       as (on gpu_loc ((gA |-> Frac fa sA) ** ((gx |-> Frac fx sx) ** (gy |-> sy0))));

  launch_sync (kamsbmv n k np alpha beta gA gx gy #() #sA #sx #sy0 #fa #fx);

  on_star_eq gpu_loc (gA |-> Frac fa sA)
             ((gx |-> Frac fx sx) ** (gy |-> fsbmv (SZ.v n) (SZ.v k) (SZ.v np) alpha beta sA sx sy0));
  rewrite (on gpu_loc ((gA |-> Frac fa sA) ** ((gx |-> Frac fx sx) ** (gy |-> fsbmv (SZ.v n) (SZ.v k) (SZ.v np) alpha beta sA sx sy0))))
       as (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> fsbmv (SZ.v n) (SZ.v k) (SZ.v np) alpha beta sA sx sy0)));
  on_star_eq gpu_loc (gx |-> Frac fx sx) (gy |-> fsbmv (SZ.v n) (SZ.v k) (SZ.v np) alpha beta sA sx sy0);
  rewrite (on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> fsbmv (SZ.v n) (SZ.v k) (SZ.v np) alpha beta sA sx sy0)))
       as (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc (gy |-> fsbmv (SZ.v n) (SZ.v k) (SZ.v np) alpha beta sA sx sy0));
  ()
}

let sbmv_f32 = sbmv_gen #f32
let sbmv_f64 = sbmv_gen #f64
