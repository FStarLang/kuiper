module Klas.Tbmv

(* cuBLAS tbmv (lower, non-unit, no transpose) : triangular banded mat-vec

       y := A * x,

   with A an n x n lower-triangular BAND matrix with k sub-diagonals, stored in
   the cuBLAS column-major band layout (leading dim k+1): the entry A(i,j) with
   j <= i <= j+k is stored at AB[(i-j) + j*(k+1)] = AB[i + j*k]. AB has length
   np = (k+1)*n; x and y are length n.

       y[i] = sum_{j: i-k<=j<=i} AB[i + j*k] * x[j].

   The band analog of Klas.Trmv. Output is a full vector. The band index
   simplifies to i + j*k, carried as a running value (+k each step), so no
   in-loop multiplication is needed and there is no underflow. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* ----------------------------------------------------------------------- *)
(* Band index bound.                                                         *)
(* ----------------------------------------------------------------------- *)

let bidx_bound (n k i j : nat)
  : Lemma (requires i < n /\ j <= i) (ensures i + j * k < (k + 1) * n)
  = ML.lemma_mult_le_right k j i;          (* j*k <= i*k *)
    ML.lemma_mult_le_right (k + 1) (i + 1) n (* (i+1)*(k+1) <= n*(k+1) *)

(* ----------------------------------------------------------------------- *)
(* Spec: banded triangular row dot and the output vector.                    *)
(* ----------------------------------------------------------------------- *)

let rec ftbmv_dot (#et:Type0) {| floating et |}
  (n:nat) (k:nat) (np:nat{np == (k + 1) * n}) (sA : lseq et np) (sx : lseq et n)
  (i:nat{i < n}) (m:nat{m <= i + 1})
  : Tot et (decreases m)
  = if m = 0 then zero
    else (let j = m - 1 in
          let term : et =
            if i - j <= k then (bidx_bound n k i j; (Seq.index sA (i + j * k)) `mul` (Seq.index sx j))
            else zero in
          (ftbmv_dot n k np sA sx i (m - 1)) `add` term)

let ftbmv_at (#et:Type0) {| floating et |}
  (n:nat) (k:nat) (np:nat{np == (k + 1) * n}) (sA : lseq et np) (sx : lseq et n) (i:nat{i < n}) : et
  = ftbmv_dot n k np sA sx i (i + 1)

noextract
let ftbmv (#et:Type0) {| floating et |}
  (n:nat) (k:nat) (np:nat{np == (k + 1) * n}) (sA : lseq et np) (sx : lseq et n) : lseq et n
  = Seq.init n (fun (i:nat{i < n}) -> ftbmv_at n k np sA sx i)

let ftbmv_index (#et:Type0) {| floating et |}
  (n:nat) (k:nat) (np:nat{np == (k + 1) * n}) (sA : lseq et np) (sx : lseq et n) (l:nat{l < n})
  : Lemma (ensures Seq.index (ftbmv n k np sA sx) l == ftbmv_at n k np sA sx l)
          [SMTPat (Seq.index (ftbmv n k np sA sx) l)]
  = ()

(* ----------------------------------------------------------------------- *)
(* Per-row banded dot (running band index bi == i + j*k).                    *)
(* ----------------------------------------------------------------------- *)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 150"
inline_for_extraction noextract
fn tbmv_row
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
    yi : et
  ensures
    pure (yi == ftbmv_at (SZ.v n) (SZ.v k) (SZ.v np) sA sx (SZ.v i))
{
  let mut s : et = zero;
  let mut j : sz = 0sz;
  let mut bi : sz = i;

  while (!j <^ (i +^ 1sz))
    invariant live s
    invariant live j
    invariant live bi
    invariant pure (SZ.v !j <= SZ.v i + 1 /\
                    SZ.v !bi == SZ.v i + SZ.v !j * SZ.v k /\
                    !s == ftbmv_dot (SZ.v n) (SZ.v k) (SZ.v np) sA sx (SZ.v i) (SZ.v !j))
    decreases (SZ.v i + 1 - SZ.v !j)
  {
    let vj = !j;
    let z : et = zero;
    let vbi = !bi;
    bidx_bound (SZ.v n) (SZ.v k) (SZ.v i) (SZ.v vj);
    let a = Array1.(gA.(vbi));
    let b = Array1.(gx.(vj));
    let term = if ((i -^ vj) <=^ k) { a `mul` b } else { z };
    s := !s `add` term;
    bi := !bi +^ k;
    j := !j +^ 1sz;
  };
  !s
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: one banded row dot per output entry.                *)
(* ----------------------------------------------------------------------- *)

let tbmv_upd
  (#et:Type0) {| floating et |}
  (n:nat) (k:nat) (np:nat{np == (k + 1) * n}) (sA : lseq et np) (sx : lseq et n)
  (syp : lseq et n) (vi : nat{vi < n}) (yi : et)
  : Lemma (requires (forall (l:nat). l < vi ==> Seq.index syp l == ftbmv_at n k np sA sx l) /\
                    yi == ftbmv_at n k np sA sx vi)
          (ensures (forall (l:nat). l < vi + 1 ==>
                      Seq.index (Seq.upd syp vi yi) l == ftbmv_at n k np sA sx l))
  = introduce forall (l:nat). l < vi + 1 ==>
                Seq.index (Seq.upd syp vi yi) l == ftbmv_at n k np sA sx l
    with introduce _ ==> _
    with _. (if l < vi then Seq.lemma_index_upd2 syp vi yi l
             else Seq.lemma_index_upd1 syp vi yi)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 120"
inline_for_extraction noextract
fn tbmv_kf
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
  requires
    gpu ** gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0
  ensures
    gpu ** gA |-> Frac fa sA ** gx |-> Frac fx sx **
    gy |-> ftbmv (SZ.v n) (SZ.v k) (SZ.v np) sA sx
{
  let mut i : szle n = 0sz;

  while (!i <^ n)
    invariant live i
    invariant exists* (sy : lseq et (SZ.v n)). gy |-> sy **
      pure (SZ.v !i <= SZ.v n /\
            (forall (l:nat). l < SZ.v !i ==>
               Seq.index sy l == ftbmv_at (SZ.v n) (SZ.v k) (SZ.v np) sA sx l))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let yi = tbmv_row k np gA gx vi;
    with syp. assert (gy |-> syp);
    Array1.(gy.(vi) <- yi);
    tbmv_upd (SZ.v n) (SZ.v k) (SZ.v np) sA sx syp (SZ.v vi) yi;
    i := !i +^ 1sz;
  };

  with syf. assert (gy |-> syf);
  Seq.lemma_eq_intro syf (ftbmv (SZ.v n) (SZ.v k) (SZ.v np) sA sx);
  ()
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper (single-thread kernel_desc_n, like Klas.Trmv).           *)
(* ----------------------------------------------------------------------- *)

ghost
fn tbmv_setup
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
fn tbmv_teardown
  (#et:Type0) {| floating et |}
  (#n : szp)
  (k : sz)
  (np : szp { SZ.v np == (SZ.v k + 1) * SZ.v n })
  (gA : array1 et (l1_forward np))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (#sA : erased (lseq et (SZ.v np)))
  (#sx : erased (lseq et (SZ.v n)))
  (#fa #fx : perm)
  ()
  norewrite
  requires
    (forall+ (tid : natlt 1sz).
       gA |-> Frac fa sA ** gx |-> Frac fx sx **
       gy |-> ftbmv (SZ.v n) (SZ.v k) (SZ.v np) sA sx) ** emp
  ensures
    gA |-> Frac fa sA ** gx |-> Frac fx sx **
    gy |-> ftbmv (SZ.v n) (SZ.v k) (SZ.v np) sA sx
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx **
       gy |-> ftbmv (SZ.v n) (SZ.v k) (SZ.v np) sA sx);
}

inline_for_extraction noextract
let kamtbmv
  (#et:Type0) {| floating et |}
  (n : szp)
  (k : sz)
  (np : szp { SZ.v np == (SZ.v k + 1) * SZ.v n })
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
                gy |-> ftbmv (SZ.v n) (SZ.v k) (SZ.v np) sA sx)
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> tbmv_kf k np gA gx gy #sA #sx #sy0 #fa #fx);
    frame    = emp;
    teardown = tbmv_teardown k np gA gx gy;
    setup    = tbmv_setup k np gA gx gy;
    kpre  = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0);
    kpost = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx **
                                     gy |-> ftbmv (SZ.v n) (SZ.v k) (SZ.v np) sA sx);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn tbmv_gen
  (#et:Type0) {| floating et |}
  (n : szp)
  (k : sz)
  (np : szp { SZ.v np == (SZ.v k + 1) * SZ.v n })
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
    on gpu_loc (gy |-> ftbmv (SZ.v n) (SZ.v k) (SZ.v np) sA sx)
{
  on_star_eq gpu_loc (gx |-> Frac fx sx) (gy |-> sy0);
  rewrite (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc (gy |-> sy0))
       as (on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> sy0)));
  on_star_eq gpu_loc (gA |-> Frac fa sA) ((gx |-> Frac fx sx) ** (gy |-> sy0));
  rewrite (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> sy0)))
       as (on gpu_loc ((gA |-> Frac fa sA) ** ((gx |-> Frac fx sx) ** (gy |-> sy0))));

  launch_sync (kamtbmv n k np gA gx gy #() #sA #sx #sy0 #fa #fx);

  on_star_eq gpu_loc (gA |-> Frac fa sA)
             ((gx |-> Frac fx sx) ** (gy |-> ftbmv (SZ.v n) (SZ.v k) (SZ.v np) sA sx));
  rewrite (on gpu_loc ((gA |-> Frac fa sA) ** ((gx |-> Frac fx sx) ** (gy |-> ftbmv (SZ.v n) (SZ.v k) (SZ.v np) sA sx))))
       as (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> ftbmv (SZ.v n) (SZ.v k) (SZ.v np) sA sx)));
  on_star_eq gpu_loc (gx |-> Frac fx sx) (gy |-> ftbmv (SZ.v n) (SZ.v k) (SZ.v np) sA sx);
  rewrite (on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> ftbmv (SZ.v n) (SZ.v k) (SZ.v np) sA sx)))
       as (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc (gy |-> ftbmv (SZ.v n) (SZ.v k) (SZ.v np) sA sx));
  ()
}

let tbmv_f32 = tbmv_gen #f32
let tbmv_f64 = tbmv_gen #f64
