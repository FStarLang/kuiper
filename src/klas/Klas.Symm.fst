module Klas.Symm

(* cuBLAS symm (side = left, uplo = lower) : symmetric matrix-matrix multiply

       X := alpha * A * B + beta * C,

   with A an n x n SYMMETRIC matrix of which only the lower triangle is stored
   (row-major, length n*n; A[i][j] for j>i is read from A[j*n+i]); B, C, X are
   n x k column-major (length n*k). Out-of-place output X (cuBLAS is in-place
   C := alpha*A*B + beta*C; the separate output is functionally equivalent and
   simpler to verify, matching Klas.Trsm / Klas.Trmm). Each cell is independent:

       X[c*n+i] = alpha * (sum_j A_sym[i][j] * B[c*n+j]) + beta * C[c*n+i].

   Reuses the symmetric row-dot fsym_dot / sym_index from Klas.Symv and the
   column-major index helpers fidx / bcol / col_bound / div_lt from Klas.Trsm. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Klas.Trsm { fidx, bcol, bcol_index, col_bound, div_lt }
open Klas.Trsv { idx_bound }
open Klas.Symv { sym_index, fsym_dot }
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* ----------------------------------------------------------------------- *)
(* Spec: the (alpha-scaled symmetric dot + beta*C) cell and the matrix.      *)
(* ----------------------------------------------------------------------- *)

let fsymm_at (#et:Type0) {| floating et |}
  (n:pos) (na:nat{n * n <= na}) (k:nat) (alpha beta : et) (sA : lseq et na)
  (sB sC : lseq et (n * k)) (c:nat{c < k}) (i:nat{i < n}) : et
  = (alpha `mul` (fsym_dot n na sA (bcol n k sB c) i n))
    `add` (beta `mul` (Seq.index sC (fidx n k c i)))

noextract
let fsymm (#et:Type0) {| floating et |}
  (n:pos) (na:nat{n * n <= na}) (k:nat) (nk:nat{nk == n * k}) (alpha beta : et)
  (sA : lseq et na) (sB sC : lseq et nk)
  : lseq et nk
  = Seq.init nk
      (fun (idx:nat{idx < nk}) ->
         div_lt idx n k;
         fsymm_at n na k alpha beta sA (sB <: lseq et (n * k)) (sC <: lseq et (n * k))
           (idx / n) (idx % n))

let fsymm_index (#et:Type0) {| floating et |}
  (n:pos) (na:nat{n * n <= na}) (k:nat) (nk:nat{nk == n * k}) (alpha beta : et)
  (sA : lseq et na) (sB sC : lseq et nk)
  (c:nat{c < k}) (i:nat{i < n})
  : Lemma (ensures Seq.index (fsymm n na k nk alpha beta sA sB sC) (fidx n k c i)
                   == fsymm_at n na k alpha beta sA (sB <: lseq et (n * k)) (sC <: lseq et (n * k)) c i)
          [SMTPat (Seq.index (fsymm n na k nk alpha beta sA sB sC) (fidx n k c i))]
  = col_bound n k c;
    ML.lemma_div_plus i c n;
    ML.small_div i n;
    ML.lemma_mod_plus i c n;
    ML.small_mod i n

let fsymm_ext (#et:Type0) {| floating et |}
  (n:pos) (na:nat{n * n <= na}) (k:nat) (nk:nat{nk == n * k}) (alpha beta : et)
  (sA : lseq et na) (sB sC : lseq et nk) (sX : lseq et nk)
  : Lemma (requires (forall (c':nat) (i':nat). c' < k /\ i' < n ==>
                       Seq.index sX (fidx n k c' i')
                       == fsymm_at n na k alpha beta sA (sB <: lseq et (n * k)) (sC <: lseq et (n * k)) c' i'))
          (ensures sX == fsymm n na k nk alpha beta sA sB sC)
  = introduce forall (idx:nat{idx < nk}).
                Seq.index sX idx == Seq.index (fsymm n na k nk alpha beta sA sB sC) idx
    with (div_lt idx n k; ML.lemma_div_mod idx n;
          fsymm_index n na k nk alpha beta sA sB sC (idx / n) (idx % n));
    Seq.lemma_eq_intro sX (fsymm n na k nk alpha beta sA sB sC)

(* ----------------------------------------------------------------------- *)
(* Per-row symmetric dot against column c of B.                              *)
(* ----------------------------------------------------------------------- *)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 100"
inline_for_extraction noextract
fn symm_row
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (gA : array1 et (l1_forward (n *^ n)))
  (gB : array1 et (l1_forward (n *^ k)))
  (c : szlt k)
  (i : szlt n)
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sB : erased (lseq et (SZ.v (n *^ k))))
  (#fa #fb : perm)
  preserves
    gpu ** gA |-> Frac fa sA ** gB |-> Frac fb sB
  returns
    s : et
  ensures
    pure (s == fsym_dot (SZ.v n) (SZ.v (n *^ n)) sA (bcol (SZ.v n) (SZ.v k) sB (SZ.v c)) (SZ.v i) (SZ.v n))
{
  let co = c *^ n;
  let mut s : et = zero;
  let mut j : szle n = 0sz;

  while (!j <^ n)
    invariant live s
    invariant live j
    invariant pure (SZ.v !j <= SZ.v n /\
                    !s == fsym_dot (SZ.v n) (SZ.v (n *^ n)) sA
                            (bcol (SZ.v n) (SZ.v k) sB (SZ.v c)) (SZ.v i) (SZ.v !j))
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
    let bj = Array1.(gB.(co +^ vj));
    s := !s `add` (aij `mul` bj);
    j := !j +^ 1sz;
  };
  !s
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: column by column, row by row, fresh output X.       *)
(* ----------------------------------------------------------------------- *)

let upd_preserves_2d
  (#et:Type0) {| floating et |}
  (n:pos) (na:nat{n * n <= na}) (k:nat) (alpha beta : et) (sA : lseq et na)
  (sB sC : lseq et (n * k)) (sX : lseq et (n * k))
  (c:nat{c < k}) (i:nat{i < n}) (xi : et)
  : Lemma
      (requires
        (forall (c':nat) (i':nat). c' < c /\ i' < n ==>
           Seq.index sX (fidx n k c' i') == fsymm_at n na k alpha beta sA sB sC c' i') /\
        (forall (j:nat). j < i ==>
           Seq.index sX (fidx n k c j) == fsymm_at n na k alpha beta sA sB sC c j) /\
        xi == fsymm_at n na k alpha beta sA sB sC c i)
      (ensures
        (let sX' = Seq.upd sX (fidx n k c i) xi in
         (forall (c':nat) (i':nat). c' < c /\ i' < n ==>
            Seq.index sX' (fidx n k c' i') == fsymm_at n na k alpha beta sA sB sC c' i') /\
         (forall (j:nat). j < i + 1 ==>
            Seq.index sX' (fidx n k c j) == fsymm_at n na k alpha beta sA sB sC c j)))
  = col_bound n k c;
    let sX' = Seq.upd sX (fidx n k c i) xi in
    introduce forall (c':nat) (i':nat). c' < c /\ i' < n ==>
                Seq.index sX' (fidx n k c' i') == fsymm_at n na k alpha beta sA sB sC c' i'
    with introduce _ ==> _
    with _. (ML.lemma_mult_le_right n (c' + 1) c;
             ML.lemma_mult_le_right n (c' + 1) k;
             Seq.lemma_index_upd2 sX (c * n + i) xi (c' * n + i'));
    introduce forall (j:nat). j < i + 1 ==>
                Seq.index sX' (fidx n k c j) == fsymm_at n na k alpha beta sA sB sC c j
    with introduce _ ==> _
    with _. (col_bound n k c;
             if j < i
             then Seq.lemma_index_upd2 sX (c * n + i) xi (c * n + j)
             else Seq.lemma_index_upd1 sX (c * n + i) xi)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 120"
inline_for_extraction noextract
fn symm_col
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (alpha beta : et)
  (gA : array1 et (l1_forward (n *^ n)))
  (gB : array1 et (l1_forward (n *^ k)))
  (gC : array1 et (l1_forward (n *^ k)))
  (gX : array1 et (l1_forward (n *^ k)))
  (c : szlt k)
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sB : erased (lseq et (SZ.v (n *^ k))))
  (#sC : erased (lseq et (SZ.v (n *^ k))))
  (#sX0 : erased (lseq et (SZ.v (n *^ k))))
  (#fa #fb #fc : perm)
  preserves
    gpu ** gA |-> Frac fa sA ** gB |-> Frac fb sB ** gC |-> Frac fc sC
  requires
    gX |-> sX0 **
    pure (forall (c':nat) (i':nat). c' < SZ.v c /\ i' < SZ.v n ==>
            Seq.index sX0 (fidx (SZ.v n) (SZ.v k) c' i')
            == fsymm_at (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) alpha beta sA sB sC c' i')
  ensures
    exists* (sX : lseq et (SZ.v (n *^ k))). gX |-> sX **
      pure (forall (c':nat) (i':nat). c' <= SZ.v c /\ i' < SZ.v n ==>
              Seq.index sX (fidx (SZ.v n) (SZ.v k) c' i')
              == fsymm_at (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) alpha beta sA sB sC c' i')
{
  let mut i : szle n = 0sz;

  while (!i <^ n)
    invariant live i
    invariant exists* (sX : lseq et (SZ.v (n *^ k))). gX |-> sX **
      pure (SZ.v !i <= SZ.v n /\
            (forall (c':nat) (i':nat). c' < SZ.v c /\ i' < SZ.v n ==>
               Seq.index sX (fidx (SZ.v n) (SZ.v k) c' i')
               == fsymm_at (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) alpha beta sA sB sC c' i') /\
            (forall (j:nat). j < SZ.v !i ==>
               Seq.index sX (fidx (SZ.v n) (SZ.v k) (SZ.v c) j)
               == fsymm_at (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) alpha beta sA sB sC (SZ.v c) j))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let s = symm_row gA gB c vi;
    col_bound (SZ.v n) (SZ.v k) (SZ.v c);
    let ci = Array1.(gC.((c *^ n) +^ vi));
    let xi = (alpha `mul` s) `add` (beta `mul` ci);
    with sxpre. assert (gX |-> sxpre);
    Array1.(gX.((c *^ n) +^ vi) <- xi);
    upd_preserves_2d (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) alpha beta sA sB sC sxpre (SZ.v c) (SZ.v vi) xi;
    i := !i +^ 1sz;
  };
}
#pop-options

#push-options "--fuel 2 --ifuel 1 --z3rlimit 120"
inline_for_extraction noextract
fn symm_kf
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (alpha beta : et)
  (gA : array1 et (l1_forward (n *^ n)))
  (gB : array1 et (l1_forward (n *^ k)))
  (gC : array1 et (l1_forward (n *^ k)))
  (gX : array1 et (l1_forward (n *^ k)))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sB : erased (lseq et (SZ.v (n *^ k))))
  (#sC : erased (lseq et (SZ.v (n *^ k))))
  (#sX0 : erased (lseq et (SZ.v (n *^ k))))
  (#fa #fb #fc : perm)
  ()
  norewrite
  requires
    gpu ** gA |-> Frac fa sA ** gB |-> Frac fb sB ** gC |-> Frac fc sC ** gX |-> sX0
  ensures
    gpu ** gA |-> Frac fa sA ** gB |-> Frac fb sB ** gC |-> Frac fc sC **
    gX |-> fsymm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha beta sA sB sC
{
  let mut c : szle k = 0sz;

  while (!c <^ k)
    invariant live c
    invariant exists* (sX : lseq et (SZ.v (n *^ k))). gX |-> sX **
      pure (SZ.v !c <= SZ.v k /\
            (forall (c':nat) (i':nat). c' < SZ.v !c /\ i' < SZ.v n ==>
               Seq.index sX (fidx (SZ.v n) (SZ.v k) c' i')
               == fsymm_at (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) alpha beta sA sB sC c' i'))
    decreases (SZ.v k - SZ.v !c)
  {
    let vc = !c;
    symm_col alpha beta gA gB gC gX vc;
    c := !c +^ 1sz;
  };

  with sXf. assert (gX |-> sXf);
  fsymm_ext (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha beta sA sB sC sXf;
  ()
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper (single-thread kernel_desc_n, like Klas.Trmm).           *)
(* ----------------------------------------------------------------------- *)

ghost
fn symm_setup
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (gA : array1 et (l1_forward (n *^ n)))
  (gB : array1 et (l1_forward (n *^ k)))
  (gC : array1 et (l1_forward (n *^ k)))
  (gX : array1 et (l1_forward (n *^ k)))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sB : erased (lseq et (SZ.v (n *^ k))))
  (#sC : erased (lseq et (SZ.v (n *^ k))))
  (#sX0 : erased (lseq et (SZ.v (n *^ k))))
  (#fa #fb #fc : perm)
  ()
  norewrite
  requires gA |-> Frac fa sA ** gB |-> Frac fb sB ** gC |-> Frac fc sC ** gX |-> sX0
  ensures (forall+ (tid : natlt 1sz).
             gA |-> Frac fa sA ** gB |-> Frac fb sB ** gC |-> Frac fc sC ** gX |-> sX0) ** emp
{
  forevery_singleton_intro #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gB |-> Frac fb sB ** gC |-> Frac fc sC ** gX |-> sX0);
}

ghost
fn symm_teardown
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (alpha beta : et)
  (gA : array1 et (l1_forward (n *^ n)))
  (gB : array1 et (l1_forward (n *^ k)))
  (gC : array1 et (l1_forward (n *^ k)))
  (gX : array1 et (l1_forward (n *^ k)))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sB : erased (lseq et (SZ.v (n *^ k))))
  (#sC : erased (lseq et (SZ.v (n *^ k))))
  (#fa #fb #fc : perm)
  ()
  norewrite
  requires
    (forall+ (tid : natlt 1sz).
       gA |-> Frac fa sA ** gB |-> Frac fb sB ** gC |-> Frac fc sC **
       gX |-> fsymm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha beta sA sB sC) ** emp
  ensures
    gA |-> Frac fa sA ** gB |-> Frac fb sB ** gC |-> Frac fc sC **
    gX |-> fsymm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha beta sA sB sC
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gB |-> Frac fb sB ** gC |-> Frac fc sC **
       gX |-> fsymm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha beta sA sB sC);
}

inline_for_extraction noextract
let kamsymm
  (#et:Type0) {| floating et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (alpha beta : et)
  (gA : array1 et (l1_forward (n *^ n)))
  (gB : array1 et (l1_forward (n *^ k)))
  (gC : array1 et (l1_forward (n *^ k)))
  (gX : array1 et (l1_forward (n *^ k)))
  (#_ : squash (Array1.is_global gA /\ Array1.is_global gB /\ Array1.is_global gC /\ Array1.is_global gX))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sB : erased (lseq et (SZ.v (n *^ k))))
  (#sC : erased (lseq et (SZ.v (n *^ k))))
  (#sX0 : erased (lseq et (SZ.v (n *^ k))))
  (#fa #fb #fc : perm)
  : kernel_desc
      (requires gA |-> Frac fa sA ** gB |-> Frac fb sB ** gC |-> Frac fc sC ** gX |-> sX0)
      (ensures  gA |-> Frac fa sA ** gB |-> Frac fb sB ** gC |-> Frac fc sC **
                gX |-> fsymm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha beta sA sB sC)
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> symm_kf alpha beta gA gB gC gX #sA #sB #sC #sX0 #fa #fb #fc);
    frame    = emp;
    teardown = symm_teardown alpha beta gA gB gC gX;
    setup    = symm_setup gA gB gC gX;
    kpre  = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gB |-> Frac fb sB ** gC |-> Frac fc sC ** gX |-> sX0);
    kpost = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gB |-> Frac fb sB ** gC |-> Frac fc sC **
                                     gX |-> fsymm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha beta sA sB sC);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn symm_gen
  (#et:Type0) {| floating et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (alpha beta : et)
  (gA : array1 et (l1_forward (n *^ n)) { Array1.is_global gA })
  (gB : array1 et (l1_forward (n *^ k)) { Array1.is_global gB })
  (gC : array1 et (l1_forward (n *^ k)) { Array1.is_global gC })
  (gX : array1 et (l1_forward (n *^ k)) { Array1.is_global gX })
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sB : erased (lseq et (SZ.v (n *^ k))))
  (#sC : erased (lseq et (SZ.v (n *^ k))))
  (#sX0 : erased (lseq et (SZ.v (n *^ k))))
  (#fa #fb #fc : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fa sA) **
    on gpu_loc (gB |-> Frac fb sB) **
    on gpu_loc (gC |-> Frac fc sC)
  requires
    on gpu_loc (gX |-> sX0)
  ensures
    on gpu_loc (gX |-> fsymm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha beta sA sB sC)
{
  on_star_eq gpu_loc (gC |-> Frac fc sC) (gX |-> sX0);
  rewrite (on gpu_loc (gC |-> Frac fc sC) ** on gpu_loc (gX |-> sX0))
       as (on gpu_loc ((gC |-> Frac fc sC) ** (gX |-> sX0)));
  on_star_eq gpu_loc (gB |-> Frac fb sB) ((gC |-> Frac fc sC) ** (gX |-> sX0));
  rewrite (on gpu_loc (gB |-> Frac fb sB) ** on gpu_loc ((gC |-> Frac fc sC) ** (gX |-> sX0)))
       as (on gpu_loc ((gB |-> Frac fb sB) ** ((gC |-> Frac fc sC) ** (gX |-> sX0))));
  on_star_eq gpu_loc (gA |-> Frac fa sA) ((gB |-> Frac fb sB) ** ((gC |-> Frac fc sC) ** (gX |-> sX0)));
  rewrite (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gB |-> Frac fb sB) ** ((gC |-> Frac fc sC) ** (gX |-> sX0))))
       as (on gpu_loc ((gA |-> Frac fa sA) ** ((gB |-> Frac fb sB) ** ((gC |-> Frac fc sC) ** (gX |-> sX0)))));

  launch_sync (kamsymm n k alpha beta gA gB gC gX #() #sA #sB #sC #sX0 #fa #fb #fc);

  on_star_eq gpu_loc (gA |-> Frac fa sA)
             ((gB |-> Frac fb sB) ** ((gC |-> Frac fc sC) ** (gX |-> fsymm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha beta sA sB sC)));
  rewrite (on gpu_loc ((gA |-> Frac fa sA) ** ((gB |-> Frac fb sB) ** ((gC |-> Frac fc sC) ** (gX |-> fsymm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha beta sA sB sC)))))
       as (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gB |-> Frac fb sB) ** ((gC |-> Frac fc sC) ** (gX |-> fsymm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha beta sA sB sC))));
  on_star_eq gpu_loc (gB |-> Frac fb sB) ((gC |-> Frac fc sC) ** (gX |-> fsymm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha beta sA sB sC));
  rewrite (on gpu_loc ((gB |-> Frac fb sB) ** ((gC |-> Frac fc sC) ** (gX |-> fsymm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha beta sA sB sC))))
       as (on gpu_loc (gB |-> Frac fb sB) ** on gpu_loc ((gC |-> Frac fc sC) ** (gX |-> fsymm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha beta sA sB sC)));
  on_star_eq gpu_loc (gC |-> Frac fc sC) (gX |-> fsymm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha beta sA sB sC);
  rewrite (on gpu_loc ((gC |-> Frac fc sC) ** (gX |-> fsymm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha beta sA sB sC)))
       as (on gpu_loc (gC |-> Frac fc sC) ** on gpu_loc (gX |-> fsymm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha beta sA sB sC));
  ()
}

let symm_f32 = symm_gen #f32
let symm_f64 = symm_gen #f64
