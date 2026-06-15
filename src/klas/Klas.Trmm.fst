module Klas.Trmm

(* cuBLAS trmm : triangular matrix-matrix multiply (out-of-place)

       X := alpha * A * B,

   with A an n x n lower-triangular (non-unit) matrix (row-major, length n*n)
   and B, X both n x k column-major (length n*k). Column c of X is
   alpha * (A * column c of B), i.e. each entry is a triangular row dot scaled
   by alpha:

       X[c*n+i] = alpha * sum_{j<=i} A[i*n+j] * B[c*n+j].

   This is the trmv analog of trsm: it solves nothing, it multiplies. We reuse
   the triangular row-dot spec ftrmv_at from Klas.Trmv and the column-major
   index helpers (fidx / bcol / col_bound / div_lt) from Klas.Trsm. A single
   thread with a triple loop (column, row, dot) suffices. cuBLAS trmm is
   in-place (B := alpha*A*B); we use a separate output X, which is functionally
   equivalent and simpler to verify. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Klas.Trsm { fidx, bcol, bcol_index, col_bound, div_lt }
open Klas.Trsv { idx_bound }
open Klas.Trmv { ftrmv_dot, ftrmv_at }
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* ----------------------------------------------------------------------- *)
(* Spec: the (alpha-scaled) triangular product cell and the whole matrix.    *)
(* ----------------------------------------------------------------------- *)

let ftrmm_at (#et:Type0) {| floating et |}
  (n:nat) (na:nat{n * n <= na}) (k:nat) (alpha : et) (sA : lseq et na) (sB : lseq et (n * k))
  (c:nat{c < k}) (i:nat{i < n}) : et
  = alpha `mul` (ftrmv_at n na sA (bcol n k sB c) i)

noextract
let ftrmm (#et:Type0) {| floating et |}
  (n:pos) (na:nat{n * n <= na}) (k:nat) (nk:nat{nk == n * k}) (alpha : et)
  (sA : lseq et na) (sB : lseq et nk)
  : lseq et nk
  = Seq.init nk
      (fun (idx:nat{idx < nk}) ->
         div_lt idx n k;
         ftrmm_at n na k alpha sA (sB <: lseq et (n * k)) (idx / n) (idx % n))

let ftrmm_index (#et:Type0) {| floating et |}
  (n:pos) (na:nat{n * n <= na}) (k:nat) (nk:nat{nk == n * k}) (alpha : et)
  (sA : lseq et na) (sB : lseq et nk)
  (c:nat{c < k}) (i:nat{i < n})
  : Lemma (ensures Seq.index (ftrmm n na k nk alpha sA sB) (fidx n k c i)
                   == ftrmm_at n na k alpha sA (sB <: lseq et (n * k)) c i)
          [SMTPat (Seq.index (ftrmm n na k nk alpha sA sB) (fidx n k c i))]
  = col_bound n k c;
    ML.lemma_div_plus i c n;
    ML.small_div i n;
    ML.lemma_mod_plus i c n;
    ML.small_mod i n

let ftrmm_ext (#et:Type0) {| floating et |}
  (n:pos) (na:nat{n * n <= na}) (k:nat) (nk:nat{nk == n * k}) (alpha : et)
  (sA : lseq et na) (sB : lseq et nk) (sX : lseq et nk)
  : Lemma (requires (forall (c':nat) (i':nat). c' < k /\ i' < n ==>
                       Seq.index sX (fidx n k c' i')
                       == ftrmm_at n na k alpha sA (sB <: lseq et (n * k)) c' i'))
          (ensures sX == ftrmm n na k nk alpha sA sB)
  = introduce forall (idx:nat{idx < nk}).
                Seq.index sX idx == Seq.index (ftrmm n na k nk alpha sA sB) idx
    with (div_lt idx n k; ML.lemma_div_mod idx n;
          ftrmm_index n na k nk alpha sA sB (idx / n) (idx % n));
    Seq.lemma_eq_intro sX (ftrmm n na k nk alpha sA sB)

(* ----------------------------------------------------------------------- *)
(* Per-row triangular dot for one column (offset c*n into B/X), scaled.      *)
(* ----------------------------------------------------------------------- *)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 80"
inline_for_extraction noextract
fn trmm_row
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (alpha : et)
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
    xi : et
  ensures
    pure (xi == ftrmm_at (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) alpha sA sB (SZ.v c) (SZ.v i))
{
  let co = c *^ n;
  let mut s : et = zero;
  let mut j : sz = 0sz;

  while (!j <^ (i +^ 1sz))
    invariant live s
    invariant live j
    invariant pure (SZ.v !j <= SZ.v i + 1 /\
                    !s == ftrmv_dot (SZ.v n) (SZ.v (n *^ n)) sA
                            (bcol (SZ.v n) (SZ.v k) sB (SZ.v c)) (SZ.v i) (SZ.v !j))
    decreases (SZ.v i + 1 - SZ.v !j)
  {
    let vj = !j;
    let aij = Array1.(gA.((i *^ n) +^ vj));
    let bj = Array1.(gB.(co +^ vj));
    s := !s `add` (aij `mul` bj);
    j := !j +^ 1sz;
  };
  alpha `mul` !s
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: column by column, row by row.                       *)
(* ----------------------------------------------------------------------- *)

let upd_preserves_2d
  (#et:Type0) {| floating et |}
  (n:nat) (na:nat{n * n <= na}) (k:nat) (alpha : et) (sA : lseq et na)
  (sB : lseq et (n * k)) (sX : lseq et (n * k))
  (c:nat{c < k}) (i:nat{i < n}) (xi : et)
  : Lemma
      (requires
        (forall (c':nat) (i':nat). c' < c /\ i' < n ==>
           Seq.index sX (fidx n k c' i') == ftrmm_at n na k alpha sA sB c' i') /\
        (forall (j:nat). j < i ==>
           Seq.index sX (fidx n k c j) == ftrmm_at n na k alpha sA sB c j) /\
        xi == ftrmm_at n na k alpha sA sB c i)
      (ensures
        (let sX' = Seq.upd sX (fidx n k c i) xi in
         (forall (c':nat) (i':nat). c' < c /\ i' < n ==>
            Seq.index sX' (fidx n k c' i') == ftrmm_at n na k alpha sA sB c' i') /\
         (forall (j:nat). j < i + 1 ==>
            Seq.index sX' (fidx n k c j) == ftrmm_at n na k alpha sA sB c j)))
  = col_bound n k c;
    let sX' = Seq.upd sX (fidx n k c i) xi in
    introduce forall (c':nat) (i':nat). c' < c /\ i' < n ==>
                Seq.index sX' (fidx n k c' i') == ftrmm_at n na k alpha sA sB c' i'
    with introduce _ ==> _
    with _. (ML.lemma_mult_le_right n (c' + 1) c;
             ML.lemma_mult_le_right n (c' + 1) k;
             Seq.lemma_index_upd2 sX (c * n + i) xi (c' * n + i'));
    introduce forall (j:nat). j < i + 1 ==>
                Seq.index sX' (fidx n k c j) == ftrmm_at n na k alpha sA sB c j
    with introduce _ ==> _
    with _. (col_bound n k c;
             if j < i
             then Seq.lemma_index_upd2 sX (c * n + i) xi (c * n + j)
             else Seq.lemma_index_upd1 sX (c * n + i) xi)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 100"
inline_for_extraction noextract
fn trmm_col
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (alpha : et)
  (gA : array1 et (l1_forward (n *^ n)))
  (gB : array1 et (l1_forward (n *^ k)))
  (gX : array1 et (l1_forward (n *^ k)))
  (c : szlt k)
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sB : erased (lseq et (SZ.v (n *^ k))))
  (#sX0 : erased (lseq et (SZ.v (n *^ k))))
  (#fa #fb : perm)
  preserves
    gpu ** gA |-> Frac fa sA ** gB |-> Frac fb sB
  requires
    gX |-> sX0 **
    pure (forall (c':nat) (i':nat). c' < SZ.v c /\ i' < SZ.v n ==>
            Seq.index sX0 (fidx (SZ.v n) (SZ.v k) c' i')
            == ftrmm_at (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) alpha sA sB c' i')
  ensures
    exists* (sX : lseq et (SZ.v (n *^ k))). gX |-> sX **
      pure (forall (c':nat) (i':nat). c' <= SZ.v c /\ i' < SZ.v n ==>
              Seq.index sX (fidx (SZ.v n) (SZ.v k) c' i')
              == ftrmm_at (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) alpha sA sB c' i')
{
  let mut i : szle n = 0sz;

  while (!i <^ n)
    invariant live i
    invariant exists* (sX : lseq et (SZ.v (n *^ k))). gX |-> sX **
      pure (SZ.v !i <= SZ.v n /\
            (forall (c':nat) (i':nat). c' < SZ.v c /\ i' < SZ.v n ==>
               Seq.index sX (fidx (SZ.v n) (SZ.v k) c' i')
               == ftrmm_at (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) alpha sA sB c' i') /\
            (forall (j:nat). j < SZ.v !i ==>
               Seq.index sX (fidx (SZ.v n) (SZ.v k) (SZ.v c) j)
               == ftrmm_at (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) alpha sA sB (SZ.v c) j))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let xi = trmm_row alpha gA gB c vi;
    with sxpre. assert (gX |-> sxpre);
    Array1.(gX.((c *^ n) +^ vi) <- xi);
    col_bound (SZ.v n) (SZ.v k) (SZ.v c);
    upd_preserves_2d (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) alpha sA sB sxpre (SZ.v c) (SZ.v vi) xi;
    i := !i +^ 1sz;
  };
}
#pop-options

#push-options "--fuel 2 --ifuel 1 --z3rlimit 100"
inline_for_extraction noextract
fn trmm_kf
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (alpha : et)
  (gA : array1 et (l1_forward (n *^ n)))
  (gB : array1 et (l1_forward (n *^ k)))
  (gX : array1 et (l1_forward (n *^ k)))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sB : erased (lseq et (SZ.v (n *^ k))))
  (#sX0 : erased (lseq et (SZ.v (n *^ k))))
  (#fa #fb : perm)
  ()
  norewrite
  requires
    gpu ** gA |-> Frac fa sA ** gB |-> Frac fb sB ** gX |-> sX0
  ensures
    gpu ** gA |-> Frac fa sA ** gB |-> Frac fb sB **
    gX |-> ftrmm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha sA sB
{
  let mut c : szle k = 0sz;

  while (!c <^ k)
    invariant live c
    invariant exists* (sX : lseq et (SZ.v (n *^ k))). gX |-> sX **
      pure (SZ.v !c <= SZ.v k /\
            (forall (c':nat) (i':nat). c' < SZ.v !c /\ i' < SZ.v n ==>
               Seq.index sX (fidx (SZ.v n) (SZ.v k) c' i')
               == ftrmm_at (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) alpha sA sB c' i'))
    decreases (SZ.v k - SZ.v !c)
  {
    let vc = !c;
    trmm_col alpha gA gB gX vc;
    c := !c +^ 1sz;
  };

  with sXf. assert (gX |-> sXf);
  ftrmm_ext (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha sA sB sXf;
  ()
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper (single-thread kernel_desc_n, like Klas.Trsm).           *)
(* ----------------------------------------------------------------------- *)

ghost
fn trmm_setup
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (gA : array1 et (l1_forward (n *^ n)))
  (gB : array1 et (l1_forward (n *^ k)))
  (gX : array1 et (l1_forward (n *^ k)))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sB : erased (lseq et (SZ.v (n *^ k))))
  (#sX0 : erased (lseq et (SZ.v (n *^ k))))
  (#fa #fb : perm)
  ()
  norewrite
  requires gA |-> Frac fa sA ** gB |-> Frac fb sB ** gX |-> sX0
  ensures (forall+ (tid : natlt 1sz).
             gA |-> Frac fa sA ** gB |-> Frac fb sB ** gX |-> sX0) ** emp
{
  forevery_singleton_intro #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gB |-> Frac fb sB ** gX |-> sX0);
}

ghost
fn trmm_teardown
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (alpha : et)
  (gA : array1 et (l1_forward (n *^ n)))
  (gB : array1 et (l1_forward (n *^ k)))
  (gX : array1 et (l1_forward (n *^ k)))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sB : erased (lseq et (SZ.v (n *^ k))))
  (#fa #fb : perm)
  ()
  norewrite
  requires
    (forall+ (tid : natlt 1sz).
       gA |-> Frac fa sA ** gB |-> Frac fb sB **
       gX |-> ftrmm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha sA sB) ** emp
  ensures
    gA |-> Frac fa sA ** gB |-> Frac fb sB **
    gX |-> ftrmm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha sA sB
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gB |-> Frac fb sB **
       gX |-> ftrmm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha sA sB);
}

inline_for_extraction noextract
let kamtrmm
  (#et:Type0) {| floating et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (alpha : et)
  (gA : array1 et (l1_forward (n *^ n)))
  (gB : array1 et (l1_forward (n *^ k)))
  (gX : array1 et (l1_forward (n *^ k)))
  (#_ : squash (Array1.is_global gA /\ Array1.is_global gB /\ Array1.is_global gX))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sB : erased (lseq et (SZ.v (n *^ k))))
  (#sX0 : erased (lseq et (SZ.v (n *^ k))))
  (#fa #fb : perm)
  : kernel_desc
      (requires gA |-> Frac fa sA ** gB |-> Frac fb sB ** gX |-> sX0)
      (ensures  gA |-> Frac fa sA ** gB |-> Frac fb sB **
                gX |-> ftrmm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha sA sB)
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> trmm_kf alpha gA gB gX #sA #sB #sX0 #fa #fb);
    frame    = emp;
    teardown = trmm_teardown alpha gA gB gX;
    setup    = trmm_setup gA gB gX;
    kpre  = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gB |-> Frac fb sB ** gX |-> sX0);
    kpost = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gB |-> Frac fb sB **
                                     gX |-> ftrmm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha sA sB);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn trmm_gen
  (#et:Type0) {| floating et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (alpha : et)
  (gA : array1 et (l1_forward (n *^ n)) { Array1.is_global gA })
  (gB : array1 et (l1_forward (n *^ k)) { Array1.is_global gB })
  (gX : array1 et (l1_forward (n *^ k)) { Array1.is_global gX })
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sB : erased (lseq et (SZ.v (n *^ k))))
  (#sX0 : erased (lseq et (SZ.v (n *^ k))))
  (#fa #fb : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fa sA) **
    on gpu_loc (gB |-> Frac fb sB)
  requires
    on gpu_loc (gX |-> sX0)
  ensures
    on gpu_loc (gX |-> ftrmm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha sA sB) **
    pure (forall (c:nat) (i:nat). c < SZ.v k /\ i < SZ.v n ==>
            Seq.index (ftrmm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha sA sB) (fidx (SZ.v n) (SZ.v k) c i)
            == ftrmm_at (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) alpha sA sB c i)
{
  on_star_eq gpu_loc (gB |-> Frac fb sB) (gX |-> sX0);
  rewrite (on gpu_loc (gB |-> Frac fb sB) ** on gpu_loc (gX |-> sX0))
       as (on gpu_loc ((gB |-> Frac fb sB) ** (gX |-> sX0)));
  on_star_eq gpu_loc (gA |-> Frac fa sA) ((gB |-> Frac fb sB) ** (gX |-> sX0));
  rewrite (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gB |-> Frac fb sB) ** (gX |-> sX0)))
       as (on gpu_loc ((gA |-> Frac fa sA) ** ((gB |-> Frac fb sB) ** (gX |-> sX0))));

  launch_sync (kamtrmm n k alpha gA gB gX #() #sA #sB #sX0 #fa #fb);

  on_star_eq gpu_loc (gA |-> Frac fa sA)
             ((gB |-> Frac fb sB) ** (gX |-> ftrmm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha sA sB));
  rewrite (on gpu_loc ((gA |-> Frac fa sA) ** ((gB |-> Frac fb sB) ** (gX |-> ftrmm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha sA sB))))
       as (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gB |-> Frac fb sB) ** (gX |-> ftrmm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha sA sB)));
  on_star_eq gpu_loc (gB |-> Frac fb sB) (gX |-> ftrmm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha sA sB);
  rewrite (on gpu_loc ((gB |-> Frac fb sB) ** (gX |-> ftrmm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha sA sB)))
       as (on gpu_loc (gB |-> Frac fb sB) ** on gpu_loc (gX |-> ftrmm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) alpha sA sB));
  ()
}

let trmm_f32 = trmm_gen #f32
let trmm_f64 = trmm_gen #f64
