module Klas.Spmv

(* cuBLAS spmv (lower) : symmetric packed matrix-vector product (in place)

       y := alpha * A * x + beta * y,

   with A an n x n SYMMETRIC matrix in PACKED row-major storage of its lower
   triangle: entry (i,j) with j<=i is at offset off(i)+j; for j>i the symmetric
   entry A[i][j] = A[j][i] is at off(j)+i. AP has length np = n*(n+1)/2.

       y[i] = alpha * (sum_j A_sym[i][j] * x[j]) + beta * y[i].

   Combines the symmetric reconstruction of Klas.Symv with the packed offset
   machinery of Klas.Tpmv (recursive off, two running offsets: oi for the
   current row, oj for the inner column). Output is a full vector. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Klas.Tpmv { off, off_mono, poff_bound }
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* ----------------------------------------------------------------------- *)
(* Symmetric packed index and its bounds.                                    *)
(* ----------------------------------------------------------------------- *)

(* The symmetric entry A[i][j]: stored at off(i)+j if j<=i, else at off(j)+i. *)
noextract
let psym_index (n:nat) (i:nat{i < n}) (j:nat{j < n}) : (r:nat{r < off n})
  = if j <= i then (poff_bound n i j; off i + j)
    else (poff_bound n j i; off j + i)

(* Both potential reads land in range; lets the kernel hoist the lemma out of
   the value-returning if (avoids the in-branch lemma Tactic failure). *)
let psym_bounds (n:nat) (i:nat{i < n}) (j:nat{j < n})
  : Lemma (ensures (j <= i ==> off i + j < off n) /\ (i <= j ==> off j + i < off n))
  = (if j <= i then poff_bound n i j);
    (if i <= j then poff_bound n j i)

let rec fpsym_dot (#et:Type0) {| floating et |}
  (n:pos) (np:nat{np == off n}) (sA : lseq et np) (sx : lseq et n)
  (i:nat{i < n}) (m:nat{m <= n})
  : Tot et (decreases m)
  = if m = 0 then zero
    else (fpsym_dot n np sA sx i (m - 1))
         `add` ((Seq.index sA (psym_index n i (m - 1))) `mul` (Seq.index sx (m - 1)))

let fspmv_at (#et:Type0) {| floating et |}
  (n:pos) (np:nat{np == off n}) (alpha beta : et)
  (sA : lseq et np) (sx sy0 : lseq et n) (i:nat{i < n}) : et
  = (alpha `mul` (fpsym_dot n np sA sx i n)) `add` (beta `mul` (Seq.index sy0 i))

noextract
let fspmv (#et:Type0) {| floating et |}
  (n:pos) (np:nat{np == off n}) (alpha beta : et)
  (sA : lseq et np) (sx sy0 : lseq et n) : lseq et n
  = Seq.init n (fun (i:nat{i < n}) -> fspmv_at n np alpha beta sA sx sy0 i)

let fspmv_index (#et:Type0) {| floating et |}
  (n:pos) (np:nat{np == off n}) (alpha beta : et)
  (sA : lseq et np) (sx sy0 : lseq et n) (i:nat{i < n})
  : Lemma (ensures Seq.index (fspmv n np alpha beta sA sx sy0) i
                   == fspmv_at n np alpha beta sA sx sy0 i)
          [SMTPat (Seq.index (fspmv n np alpha beta sA sx sy0) i)]
  = ()

(* ----------------------------------------------------------------------- *)
(* Per-row symmetric packed dot (row offset oi == off i; inner offset oj).   *)
(* ----------------------------------------------------------------------- *)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 150"
inline_for_extraction noextract
fn spmv_row
  (#et:Type0) {| floating et |}
  (#n : szp)
  (#np : szp { SZ.v np == off (SZ.v n) })
  (gA : array1 et (l1_forward np))
  (gx : array1 et (l1_forward n))
  (i : szlt n)
  (oi : sz { SZ.v oi == off (SZ.v i) })
  (#sA : erased (lseq et (SZ.v np)))
  (#sx : erased (lseq et (SZ.v n)))
  (#fa #fx : perm)
  preserves
    gpu ** gA |-> Frac fa sA ** gx |-> Frac fx sx
  returns
    s : et
  ensures
    pure (s == fpsym_dot (SZ.v n) (SZ.v np) sA sx (SZ.v i) (SZ.v n))
{
  let mut s : et = zero;
  let mut j : szle n = 0sz;
  let mut oj : sz = 0sz;

  while (!j <^ n)
    invariant live s
    invariant live j
    invariant live oj
    invariant pure (SZ.v !j <= SZ.v n /\ SZ.v !oj == off (SZ.v !j) /\
                    !s == fpsym_dot (SZ.v n) (SZ.v np) sA sx (SZ.v i) (SZ.v !j))
    decreases (SZ.v n - SZ.v !j)
  {
    let vj = !j;
    let voj = !oj;
    psym_bounds (SZ.v n) (SZ.v i) (SZ.v vj);
    let aij =
      if (vj <=^ i) {
        Array1.(gA.(oi +^ vj))
      } else {
        Array1.(gA.(voj +^ i))
      };
    let xj = Array1.(gx.(vj));
    s := !s `add` (aij `mul` xj);
    off_mono (SZ.v vj + 1) (SZ.v n);
    oj := !oj +^ (vj +^ 1sz);
    j := !j +^ 1sz;
  };
  !s
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: one row per output entry, in place.                 *)
(* ----------------------------------------------------------------------- *)

let spmv_upd
  (#et:Type0) {| floating et |}
  (n:pos) (np:nat{np == off n}) (alpha beta : et)
  (sA : lseq et np) (sx sy0 : lseq et n) (sy : lseq et n) (i:nat{i < n}) (yi : et)
  : Lemma
      (requires
        (forall (k:nat). k < i ==> Seq.index sy k == fspmv_at n np alpha beta sA sx sy0 k) /\
        (forall (k:nat). i <= k /\ k < n ==> Seq.index sy k == Seq.index sy0 k) /\
        yi == fspmv_at n np alpha beta sA sx sy0 i)
      (ensures
        (let sy' = Seq.upd sy i yi in
         (forall (k:nat). k < i + 1 ==> Seq.index sy' k == fspmv_at n np alpha beta sA sx sy0 k) /\
         (forall (k:nat). i + 1 <= k /\ k < n ==> Seq.index sy' k == Seq.index sy0 k)))
  = let sy' = Seq.upd sy i yi in
    introduce forall (k:nat). k < i + 1 ==> Seq.index sy' k == fspmv_at n np alpha beta sA sx sy0 k
    with introduce _ ==> _
    with _. (if k < i then Seq.lemma_index_upd2 sy i yi k
             else Seq.lemma_index_upd1 sy i yi);
    introduce forall (k:nat). i + 1 <= k /\ k < n ==> Seq.index sy' k == Seq.index sy0 k
    with introduce _ ==> _
    with _. Seq.lemma_index_upd2 sy i yi k

#push-options "--fuel 2 --ifuel 1 --z3rlimit 150"
inline_for_extraction noextract
fn spmv_kf
  (#et:Type0) {| floating et |}
  (#n : szp)
  (#np : szp { SZ.v np == off (SZ.v n) })
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
    gy |-> fspmv (SZ.v n) (SZ.v np) alpha beta sA sx sy0
{
  let mut i : szle n = 0sz;
  let mut o : sz = 0sz;

  while (!i <^ n)
    invariant live i
    invariant live o
    invariant exists* (sy : lseq et (SZ.v n)). gy |-> sy **
      pure (SZ.v !i <= SZ.v n /\ SZ.v !o == off (SZ.v !i) /\
            (forall (k:nat). k < SZ.v !i ==>
               Seq.index sy k == fspmv_at (SZ.v n) (SZ.v np) alpha beta sA sx sy0 k) /\
            (forall (k:nat). SZ.v !i <= k /\ k < SZ.v n ==>
               Seq.index sy k == Seq.index sy0 k))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let vo = !o;
    let s = spmv_row gA gx vi vo;
    let yi_old = Array1.(gy.(vi));
    let yi = (alpha `mul` s) `add` (beta `mul` yi_old);
    with syp. assert (gy |-> syp);
    Array1.(gy.(vi) <- yi);
    spmv_upd (SZ.v n) (SZ.v np) alpha beta sA sx sy0 syp (SZ.v vi) yi;
    off_mono (SZ.v vi + 1) (SZ.v n);
    o := !o +^ (vi +^ 1sz);
    i := !i +^ 1sz;
  };

  with syf. assert (gy |-> syf);
  Seq.lemma_eq_intro syf (fspmv (SZ.v n) (SZ.v np) alpha beta sA sx sy0);
  ()
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper (single-thread kernel_desc_n, like Klas.Symv).           *)
(* ----------------------------------------------------------------------- *)

ghost
fn spmv_setup
  (#et:Type0) {| floating et |}
  (#n : szp)
  (#np : szp { SZ.v np == off (SZ.v n) })
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
fn spmv_teardown
  (#et:Type0) {| floating et |}
  (#n : szp)
  (#np : szp { SZ.v np == off (SZ.v n) })
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
       gy |-> fspmv (SZ.v n) (SZ.v np) alpha beta sA sx sy0) ** emp
  ensures
    gA |-> Frac fa sA ** gx |-> Frac fx sx **
    gy |-> fspmv (SZ.v n) (SZ.v np) alpha beta sA sx sy0
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx **
       gy |-> fspmv (SZ.v n) (SZ.v np) alpha beta sA sx sy0);
}

inline_for_extraction noextract
let kamspmv
  (#et:Type0) {| floating et |}
  (n : szp)
  (np : szp { SZ.v np == off (SZ.v n) })
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
                gy |-> fspmv (SZ.v n) (SZ.v np) alpha beta sA sx sy0)
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> spmv_kf alpha beta gA gx gy #sA #sx #sy0 #fa #fx);
    frame    = emp;
    teardown = spmv_teardown alpha beta gA gx gy;
    setup    = spmv_setup gA gx gy;
    kpre  = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0);
    kpost = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx **
                                     gy |-> fspmv (SZ.v n) (SZ.v np) alpha beta sA sx sy0);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn spmv_gen
  (#et:Type0) {| floating et |}
  (n : szp)
  (np : szp { SZ.v np == off (SZ.v n) })
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
    on gpu_loc (gy |-> fspmv (SZ.v n) (SZ.v np) alpha beta sA sx sy0)
{
  on_star_eq gpu_loc (gx |-> Frac fx sx) (gy |-> sy0);
  rewrite (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc (gy |-> sy0))
       as (on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> sy0)));
  on_star_eq gpu_loc (gA |-> Frac fa sA) ((gx |-> Frac fx sx) ** (gy |-> sy0));
  rewrite (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> sy0)))
       as (on gpu_loc ((gA |-> Frac fa sA) ** ((gx |-> Frac fx sx) ** (gy |-> sy0))));

  launch_sync (kamspmv n np alpha beta gA gx gy #() #sA #sx #sy0 #fa #fx);

  on_star_eq gpu_loc (gA |-> Frac fa sA)
             ((gx |-> Frac fx sx) ** (gy |-> fspmv (SZ.v n) (SZ.v np) alpha beta sA sx sy0));
  rewrite (on gpu_loc ((gA |-> Frac fa sA) ** ((gx |-> Frac fx sx) ** (gy |-> fspmv (SZ.v n) (SZ.v np) alpha beta sA sx sy0))))
       as (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> fspmv (SZ.v n) (SZ.v np) alpha beta sA sx sy0)));
  on_star_eq gpu_loc (gx |-> Frac fx sx) (gy |-> fspmv (SZ.v n) (SZ.v np) alpha beta sA sx sy0);
  rewrite (on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> fspmv (SZ.v n) (SZ.v np) alpha beta sA sx sy0)))
       as (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc (gy |-> fspmv (SZ.v n) (SZ.v np) alpha beta sA sx sy0));
  ()
}

let spmv_f32 = spmv_gen #f32
let spmv_f64 = spmv_gen #f64
