module Klas.Symv

(* cuBLAS symv : symmetric matrix-vector product (in place)

       y := alpha * A * x + beta * y,

   with A an n x n SYMMETRIC matrix of which only the lower triangle is stored
   (row-major, length n*n); the (i,j) entry with j>i is read from A[j*n+i]. x and
   y are length n. Each output entry y[i] is independent, so a single thread with
   a per-row full dot (using the symmetric reconstruction) suffices. We expose
   the exact float spec; the symmetric row-dot fsym_dot is reused by Klas.Symm.
   Reuses idx_bound from Klas.Trsv for the flat index bound. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Klas.Trsv { idx_bound }
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* ----------------------------------------------------------------------- *)
(* Spec: symmetric (lower-stored) entry, full row dot, and the output cell.  *)
(* ----------------------------------------------------------------------- *)

(* The symmetric entry A[i][j]: stored at i*n+j if j<=i, else at j*n+i. *)
noextract
let sym_index (n:nat) (i:nat{i < n}) (j:nat{j < n}) : (r:nat{r < n * n})
  = if j <= i then (idx_bound n i j; i * n + j)
    else (idx_bound n j i; j * n + i)

let rec fsym_dot (#et:Type0) {| floating et |}
  (n:pos) (na:nat{n * n <= na}) (sA : lseq et na) (sx : lseq et n)
  (i:nat{i < n}) (m:nat{m <= n})
  : Tot et (decreases m)
  = if m = 0 then zero
    else (fsym_dot n na sA sx i (m - 1))
         `add` ((Seq.index sA (sym_index n i (m - 1))) `mul` (Seq.index sx (m - 1)))

let fsymv_at (#et:Type0) {| floating et |}
  (n:pos) (na:nat{n * n <= na}) (alpha beta : et)
  (sA : lseq et na) (sx sy0 : lseq et n) (i:nat{i < n}) : et
  = (alpha `mul` (fsym_dot n na sA sx i n)) `add` (beta `mul` (Seq.index sy0 i))

noextract
let fsymv (#et:Type0) {| floating et |}
  (n:pos) (na:nat{n * n <= na}) (alpha beta : et)
  (sA : lseq et na) (sx sy0 : lseq et n) : lseq et n
  = Seq.init n (fun (i:nat{i < n}) -> fsymv_at n na alpha beta sA sx sy0 i)

let fsymv_index (#et:Type0) {| floating et |}
  (n:pos) (na:nat{n * n <= na}) (alpha beta : et)
  (sA : lseq et na) (sx sy0 : lseq et n) (i:nat{i < n})
  : Lemma (ensures Seq.index (fsymv n na alpha beta sA sx sy0) i
                   == fsymv_at n na alpha beta sA sx sy0 i)
          [SMTPat (Seq.index (fsymv n na alpha beta sA sx sy0) i)]
  = ()

(* ----------------------------------------------------------------------- *)
(* Per-row symmetric dot.                                                    *)
(* ----------------------------------------------------------------------- *)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 100"
inline_for_extraction noextract
fn symv_row
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (gA : array1 et (l1_forward (n *^ n)))
  (gx : array1 et (l1_forward n))
  (i : szlt n)
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sx : erased (lseq et (SZ.v n)))
  (#fa #fx : perm)
  preserves
    gpu ** gA |-> Frac fa sA ** gx |-> Frac fx sx
  returns
    s : et
  ensures
    pure (s == fsym_dot (SZ.v n) (SZ.v (n *^ n)) sA sx (SZ.v i) (SZ.v n))
{
  let mut s : et = zero;
  let mut j : szle n = 0sz;

  while (!j <^ n)
    invariant live s
    invariant live j
    invariant pure (SZ.v !j <= SZ.v n /\
                    !s == fsym_dot (SZ.v n) (SZ.v (n *^ n)) sA sx (SZ.v i) (SZ.v !j))
    decreases (SZ.v n - SZ.v !j)
  {
    let vj = !j;
    idx_bound (SZ.v n) (SZ.v i) (SZ.v vj);
    idx_bound (SZ.v n) (SZ.v vj) (SZ.v i);
    let aij =
      if (vj <=^ i) {
        Array1.(gA.((i *^ n) +^ vj))
      } else {
        Array1.(gA.((vj *^ n) +^ i))
      };
    let xj = Array1.(gx.(vj));
    s := !s `add` (aij `mul` xj);
    j := !j +^ 1sz;
  };
  !s
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: one row per output entry, in place.                 *)
(* ----------------------------------------------------------------------- *)

(* Writing y[i] keeps the already-updated prefix (<i) and shrinks the still-
   original suffix (>i). y[i] read before the write is the original sy0[i]. *)
let symv_upd
  (#et:Type0) {| floating et |}
  (n:pos) (na:nat{n * n <= na}) (alpha beta : et)
  (sA : lseq et na) (sx sy0 : lseq et n) (sy : lseq et n) (i:nat{i < n}) (yi : et)
  : Lemma
      (requires
        (forall (k:nat). k < i ==> Seq.index sy k == fsymv_at n na alpha beta sA sx sy0 k) /\
        (forall (k:nat). i <= k /\ k < n ==> Seq.index sy k == Seq.index sy0 k) /\
        yi == fsymv_at n na alpha beta sA sx sy0 i)
      (ensures
        (let sy' = Seq.upd sy i yi in
         (forall (k:nat). k < i + 1 ==> Seq.index sy' k == fsymv_at n na alpha beta sA sx sy0 k) /\
         (forall (k:nat). i + 1 <= k /\ k < n ==> Seq.index sy' k == Seq.index sy0 k)))
  = let sy' = Seq.upd sy i yi in
    introduce forall (k:nat). k < i + 1 ==> Seq.index sy' k == fsymv_at n na alpha beta sA sx sy0 k
    with introduce _ ==> _
    with _. (if k < i then Seq.lemma_index_upd2 sy i yi k
             else Seq.lemma_index_upd1 sy i yi);
    introduce forall (k:nat). i + 1 <= k /\ k < n ==> Seq.index sy' k == Seq.index sy0 k
    with introduce _ ==> _
    with _. Seq.lemma_index_upd2 sy i yi k

#push-options "--fuel 2 --ifuel 1 --z3rlimit 120"
inline_for_extraction noextract
fn symv_kf
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (alpha beta : et)
  (gA : array1 et (l1_forward (n *^ n)))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy0 : erased (lseq et (SZ.v n)))
  (#fa #fx : perm)
  ()
  norewrite
  requires
    gpu ** gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0
  ensures
    gpu ** gA |-> Frac fa sA ** gx |-> Frac fx sx **
    gy |-> fsymv (SZ.v n) (SZ.v (n *^ n)) alpha beta sA sx sy0
{
  let mut i : szle n = 0sz;

  while (!i <^ n)
    invariant live i
    invariant exists* (sy : lseq et (SZ.v n)). gy |-> sy **
      pure (SZ.v !i <= SZ.v n /\
            (forall (k:nat). k < SZ.v !i ==>
               Seq.index sy k == fsymv_at (SZ.v n) (SZ.v (n *^ n)) alpha beta sA sx sy0 k) /\
            (forall (k:nat). SZ.v !i <= k /\ k < SZ.v n ==>
               Seq.index sy k == Seq.index sy0 k))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let s = symv_row gA gx vi;
    let yi_old = Array1.(gy.(vi));
    let yi = (alpha `mul` s) `add` (beta `mul` yi_old);
    with syp. assert (gy |-> syp);
    Array1.(gy.(vi) <- yi);
    symv_upd (SZ.v n) (SZ.v (n *^ n)) alpha beta sA sx sy0 syp (SZ.v vi) yi;
    i := !i +^ 1sz;
  };

  with syf. assert (gy |-> syf);
  Seq.lemma_eq_intro syf (fsymv (SZ.v n) (SZ.v (n *^ n)) alpha beta sA sx sy0);
  ()
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper (single-thread kernel_desc_n, like Klas.Trmv).           *)
(* ----------------------------------------------------------------------- *)

ghost
fn symv_setup
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (gA : array1 et (l1_forward (n *^ n)))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
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
fn symv_teardown
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (alpha beta : et)
  (gA : array1 et (l1_forward (n *^ n)))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy0 : erased (lseq et (SZ.v n)))
  (#fa #fx : perm)
  ()
  norewrite
  requires
    (forall+ (tid : natlt 1sz).
       gA |-> Frac fa sA ** gx |-> Frac fx sx **
       gy |-> fsymv (SZ.v n) (SZ.v (n *^ n)) alpha beta sA sx sy0) ** emp
  ensures
    gA |-> Frac fa sA ** gx |-> Frac fx sx **
    gy |-> fsymv (SZ.v n) (SZ.v (n *^ n)) alpha beta sA sx sy0
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx **
       gy |-> fsymv (SZ.v n) (SZ.v (n *^ n)) alpha beta sA sx sy0);
}

inline_for_extraction noextract
let kamsymv
  (#et:Type0) {| floating et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (alpha beta : et)
  (gA : array1 et (l1_forward (n *^ n)))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (#_ : squash (Array1.is_global gA /\ Array1.is_global gx /\ Array1.is_global gy))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy0 : erased (lseq et (SZ.v n)))
  (#fa #fx : perm)
  : kernel_desc
      (requires gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0)
      (ensures  gA |-> Frac fa sA ** gx |-> Frac fx sx **
                gy |-> fsymv (SZ.v n) (SZ.v (n *^ n)) alpha beta sA sx sy0)
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> symv_kf alpha beta gA gx gy #sA #sx #sy0 #fa #fx);
    frame    = emp;
    teardown = symv_teardown alpha beta gA gx gy;
    setup    = symv_setup gA gx gy;
    kpre  = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0);
    kpost = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx **
                                     gy |-> fsymv (SZ.v n) (SZ.v (n *^ n)) alpha beta sA sx sy0);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn symv_gen
  (#et:Type0) {| floating et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (alpha beta : et)
  (gA : array1 et (l1_forward (n *^ n)) { Array1.is_global gA })
  (gx : array1 et (l1_forward n) { Array1.is_global gx })
  (gy : array1 et (l1_forward n) { Array1.is_global gy })
  (#sA : erased (lseq et (SZ.v (n *^ n))))
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
    on gpu_loc (gy |-> fsymv (SZ.v n) (SZ.v (n *^ n)) alpha beta sA sx sy0)
{
  on_star_eq gpu_loc (gx |-> Frac fx sx) (gy |-> sy0);
  rewrite (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc (gy |-> sy0))
       as (on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> sy0)));
  on_star_eq gpu_loc (gA |-> Frac fa sA) ((gx |-> Frac fx sx) ** (gy |-> sy0));
  rewrite (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> sy0)))
       as (on gpu_loc ((gA |-> Frac fa sA) ** ((gx |-> Frac fx sx) ** (gy |-> sy0))));

  launch_sync (kamsymv n alpha beta gA gx gy #() #sA #sx #sy0 #fa #fx);

  on_star_eq gpu_loc (gA |-> Frac fa sA)
             ((gx |-> Frac fx sx) ** (gy |-> fsymv (SZ.v n) (SZ.v (n *^ n)) alpha beta sA sx sy0));
  rewrite (on gpu_loc ((gA |-> Frac fa sA) ** ((gx |-> Frac fx sx) ** (gy |-> fsymv (SZ.v n) (SZ.v (n *^ n)) alpha beta sA sx sy0))))
       as (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> fsymv (SZ.v n) (SZ.v (n *^ n)) alpha beta sA sx sy0)));
  on_star_eq gpu_loc (gx |-> Frac fx sx) (gy |-> fsymv (SZ.v n) (SZ.v (n *^ n)) alpha beta sA sx sy0);
  rewrite (on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> fsymv (SZ.v n) (SZ.v (n *^ n)) alpha beta sA sx sy0)))
       as (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc (gy |-> fsymv (SZ.v n) (SZ.v (n *^ n)) alpha beta sA sx sy0));
  ()
}

let symv_f32 = symv_gen #f32
let symv_f64 = symv_gen #f64
