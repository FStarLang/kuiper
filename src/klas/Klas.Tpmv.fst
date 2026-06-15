module Klas.Tpmv

(* cuBLAS tpmv (lower, non-unit, no transpose) : triangular packed mat-vec

       y := A * x,

   with A an n x n lower-triangular matrix stored in PACKED row-major form: the
   lower triangle is laid out row by row, so entry (i,j) with j<=i is at offset
   off(i)+j where off(i) = i*(i+1)/2 is the number of entries in rows 0..i-1.
   AP has length off(n) = n*(n+1)/2; x and y are length n.

       y[i] = sum_{j<=i} AP[off(i)+j] * x[j].

   We avoid the i*(i+1)/2 division by defining off recursively (off 0 = 0,
   off (i+1) = off i + (i+1)) and carrying a running offset in the kernel. This
   is the packed analog of Klas.Trmv. Output is a full vector (forward packed
   reads), so no inverse packed->(i,j) index is needed. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* ----------------------------------------------------------------------- *)
(* Triangular packed offset and its bounds (no division).                    *)
(* ----------------------------------------------------------------------- *)

(* off i = number of stored entries in rows 0..i-1 = 0+1+...+i = i*(i+1)/2. *)
noextract
let rec off (i:nat) : nat = if i = 0 then 0 else off (i - 1) + i

let rec off_mono (i j : nat)
  : Lemma (requires i <= j) (ensures off i <= off j) (decreases j)
  = if i = j then () else off_mono i (j - 1)

(* The read bound: entry (i,j) with j<=i<n lands inside the packed array. *)
let poff_bound (n i j : nat)
  : Lemma (requires i < n /\ j <= i) (ensures off i + j < off n)
  = off_mono (i + 1) n

(* ----------------------------------------------------------------------- *)
(* Spec: packed triangular row dot and the output vector.                    *)
(* ----------------------------------------------------------------------- *)

let rec ftpmv_dot (#et:Type0) {| floating et |}
  (n:nat) (np:nat{np == off n}) (sA : lseq et np) (sx : lseq et n)
  (i:nat{i < n}) (m:nat{m <= i + 1})
  : Tot et (decreases m)
  = if m = 0 then zero
    else (poff_bound n i (m - 1);
          (ftpmv_dot n np sA sx i (m - 1))
          `add` ((Seq.index sA (off i + (m - 1))) `mul` (Seq.index sx (m - 1))))

let ftpmv_at (#et:Type0) {| floating et |}
  (n:nat) (np:nat{np == off n}) (sA : lseq et np) (sx : lseq et n) (i:nat{i < n}) : et
  = ftpmv_dot n np sA sx i (i + 1)

noextract
let ftpmv (#et:Type0) {| floating et |}
  (n:nat) (np:nat{np == off n}) (sA : lseq et np) (sx : lseq et n) : lseq et n
  = Seq.init n (fun (i:nat{i < n}) -> ftpmv_at n np sA sx i)

let ftpmv_index (#et:Type0) {| floating et |}
  (n:nat) (np:nat{np == off n}) (sA : lseq et np) (sx : lseq et n) (k:nat{k < n})
  : Lemma (ensures Seq.index (ftpmv n np sA sx) k == ftpmv_at n np sA sx k)
          [SMTPat (Seq.index (ftpmv n np sA sx) k)]
  = ()

(* ----------------------------------------------------------------------- *)
(* Per-row packed triangular dot (offset oi == off i carried in).            *)
(* ----------------------------------------------------------------------- *)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 100"
inline_for_extraction noextract
fn tpmv_row
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
    yi : et
  ensures
    pure (yi == ftpmv_at (SZ.v n) (SZ.v np) sA sx (SZ.v i))
{
  let mut s : et = zero;
  let mut j : sz = 0sz;

  while (!j <^ (i +^ 1sz))
    invariant live s
    invariant live j
    invariant pure (SZ.v !j <= SZ.v i + 1 /\
                    !s == ftpmv_dot (SZ.v n) (SZ.v np) sA sx (SZ.v i) (SZ.v !j))
    decreases (SZ.v i + 1 - SZ.v !j)
  {
    let vj = !j;
    poff_bound (SZ.v n) (SZ.v i) (SZ.v vj);
    let aij = Array1.(gA.(oi +^ vj));
    let xj = Array1.(gx.(vj));
    s := !s `add` (aij `mul` xj);
    j := !j +^ 1sz;
  };
  !s
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: one packed row dot per output entry.                *)
(* ----------------------------------------------------------------------- *)

let tpmv_upd
  (#et:Type0) {| floating et |}
  (n:nat) (np:nat{np == off n}) (sA : lseq et np) (sx : lseq et n)
  (syp : lseq et n) (vi : nat{vi < n}) (yi : et)
  : Lemma (requires (forall (k:nat). k < vi ==> Seq.index syp k == ftpmv_at n np sA sx k) /\
                    yi == ftpmv_at n np sA sx vi)
          (ensures (forall (k:nat). k < vi + 1 ==>
                      Seq.index (Seq.upd syp vi yi) k == ftpmv_at n np sA sx k))
  = introduce forall (k:nat). k < vi + 1 ==>
                Seq.index (Seq.upd syp vi yi) k == ftpmv_at n np sA sx k
    with introduce _ ==> _
    with _. (if k < vi then Seq.lemma_index_upd2 syp vi yi k
             else Seq.lemma_index_upd1 syp vi yi)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 120"
inline_for_extraction noextract
fn tpmv_kf
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
  requires
    gpu ** gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0
  ensures
    gpu ** gA |-> Frac fa sA ** gx |-> Frac fx sx **
    gy |-> ftpmv (SZ.v n) (SZ.v np) sA sx
{
  let mut i : szle n = 0sz;
  let mut o : sz = 0sz;

  while (!i <^ n)
    invariant live i
    invariant live o
    invariant exists* (sy : lseq et (SZ.v n)). gy |-> sy **
      pure (SZ.v !i <= SZ.v n /\ SZ.v !o == off (SZ.v !i) /\
            (forall (k:nat). k < SZ.v !i ==>
               Seq.index sy k == ftpmv_at (SZ.v n) (SZ.v np) sA sx k))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let vo = !o;
    let yi = tpmv_row gA gx vi vo;
    with syp. assert (gy |-> syp);
    Array1.(gy.(vi) <- yi);
    tpmv_upd (SZ.v n) (SZ.v np) sA sx syp (SZ.v vi) yi;
    off_mono (SZ.v vi + 1) (SZ.v n);
    o := !o +^ (vi +^ 1sz);
    i := !i +^ 1sz;
  };

  with syf. assert (gy |-> syf);
  Seq.lemma_eq_intro syf (ftpmv (SZ.v n) (SZ.v np) sA sx);
  ()
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper (single-thread kernel_desc_n, like Klas.Trmv).           *)
(* ----------------------------------------------------------------------- *)

ghost
fn tpmv_setup
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
fn tpmv_teardown
  (#et:Type0) {| floating et |}
  (#n : szp)
  (#np : szp { SZ.v np == off (SZ.v n) })
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
       gy |-> ftpmv (SZ.v n) (SZ.v np) sA sx) ** emp
  ensures
    gA |-> Frac fa sA ** gx |-> Frac fx sx **
    gy |-> ftpmv (SZ.v n) (SZ.v np) sA sx
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx **
       gy |-> ftpmv (SZ.v n) (SZ.v np) sA sx);
}

inline_for_extraction noextract
let kamtpmv
  (#et:Type0) {| floating et |}
  (n : szp)
  (np : szp { SZ.v np == off (SZ.v n) })
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
                gy |-> ftpmv (SZ.v n) (SZ.v np) sA sx)
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> tpmv_kf gA gx gy #sA #sx #sy0 #fa #fx);
    frame    = emp;
    teardown = tpmv_teardown gA gx gy;
    setup    = tpmv_setup gA gx gy;
    kpre  = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0);
    kpost = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx **
                                     gy |-> ftpmv (SZ.v n) (SZ.v np) sA sx);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn tpmv_gen
  (#et:Type0) {| floating et |}
  (n : szp)
  (np : szp { SZ.v np == off (SZ.v n) })
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
    on gpu_loc (gy |-> ftpmv (SZ.v n) (SZ.v np) sA sx)
{
  on_star_eq gpu_loc (gx |-> Frac fx sx) (gy |-> sy0);
  rewrite (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc (gy |-> sy0))
       as (on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> sy0)));
  on_star_eq gpu_loc (gA |-> Frac fa sA) ((gx |-> Frac fx sx) ** (gy |-> sy0));
  rewrite (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> sy0)))
       as (on gpu_loc ((gA |-> Frac fa sA) ** ((gx |-> Frac fx sx) ** (gy |-> sy0))));

  launch_sync (kamtpmv n np gA gx gy #() #sA #sx #sy0 #fa #fx);

  on_star_eq gpu_loc (gA |-> Frac fa sA)
             ((gx |-> Frac fx sx) ** (gy |-> ftpmv (SZ.v n) (SZ.v np) sA sx));
  rewrite (on gpu_loc ((gA |-> Frac fa sA) ** ((gx |-> Frac fx sx) ** (gy |-> ftpmv (SZ.v n) (SZ.v np) sA sx))))
       as (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> ftpmv (SZ.v n) (SZ.v np) sA sx)));
  on_star_eq gpu_loc (gx |-> Frac fx sx) (gy |-> ftpmv (SZ.v n) (SZ.v np) sA sx);
  rewrite (on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> ftpmv (SZ.v n) (SZ.v np) sA sx)))
       as (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc (gy |-> ftpmv (SZ.v n) (SZ.v np) sA sx));
  ()
}

let tpmv_f32 = tpmv_gen #f32
let tpmv_f64 = tpmv_gen #f64
