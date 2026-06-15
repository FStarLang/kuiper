module Klas.Trsm

(* cuBLAS trsm (left, lower-triangular, non-unit, no transpose): solve
   A * X = B for a matrix B of k right-hand sides. Each column is an
   independent triangular solve, so trsm is k forward substitutions and we
   reuse the per-column spec from Klas.Trsv. Storage is column-major: A is the
   usual n x n row-major matrix (length n*n); B and X are n x k with column c
   occupying the contiguous slice [c*n, c*n + n). Single thread. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Klas.Trsv
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* ----------------------------------------------------------------------- *)
(* Column slice and its index law.                                           *)
(* ----------------------------------------------------------------------- *)

let col_bound (n k c : nat)
  : Lemma (requires c < k) (ensures c * n + n <= n * k)
  = ML.lemma_mult_le_right n (c + 1) k

noextract
let bcol (#et:Type0) (n k : nat) (sB : lseq et (n * k)) (c:nat{c < k}) : lseq et n
  = col_bound n k c; Seq.slice sB (c * n) (c * n + n)

let bcol_index (#et:Type0) (n k : nat) (sB : lseq et (n * k)) (c:nat{c < k}) (i:nat{i < n})
  : Lemma (ensures Seq.index (bcol n k sB c) i == Seq.index sB (c * n + i))
          [SMTPat (Seq.index (bcol n k sB c) i)]
  = col_bound n k c

(* Bounded flat (column-major) index into an n x k array: column c, row i. *)
noextract
let fidx (n k : nat) (c:nat{c < k}) (i:nat{i < n}) : (r:nat{r < n * k /\ r == c * n + i})
  = col_bound n k c; c * n + i

(* The full trsm solution: column c is the trsv solution of column c of B. *)
let div_lt (idx n k : nat)
  : Lemma (requires idx < n * k /\ n > 0) (ensures idx / n < k /\ idx % n < n)
  = ML.lemma_div_mod idx n; ML.lemma_mod_lt idx n;
    ML.lemma_div_le idx (n * k - 1) n; ML.cancel_mul_div k n

noextract
let ftrsm (#et:Type0) {| floating et |}
  (n:pos) (na:nat{n * n <= na}) (k:nat) (nk:nat{nk == n * k})
  (sA : lseq et na) (sB : lseq et nk)
  : lseq et nk
  = Seq.init nk
      (fun (idx:nat{idx < nk}) ->
         div_lt idx n k;
         ftrsv_at n na sA (bcol n k (sB <: lseq et (n * k)) (idx / n)) (idx % n))

let ftrsm_index (#et:Type0) {| floating et |}
  (n:pos) (na:nat{n * n <= na}) (k:nat) (nk:nat{nk == n * k})
  (sA : lseq et na) (sB : lseq et nk)
  (c:nat{c < k}) (i:nat{i < n})
  : Lemma (ensures Seq.index (ftrsm n na k nk sA sB) (fidx n k c i)
                   == ftrsv_at n na sA (bcol n k (sB <: lseq et (n * k)) c) i)
          [SMTPat (Seq.index (ftrsm n na k nk sA sB) (fidx n k c i))]
  = col_bound n k c;
    ML.lemma_div_plus i c n;   (* (i + c*n)/n = i/n + c = c since i<n *)
    ML.small_div i n;          (* i/n = 0 *)
    ML.lemma_mod_plus i c n;   (* (i + c*n)%n = i%n = i *)
    ML.small_mod i n

(* The exact float trsm output equals ftrsm. *)
let ftrsm_ext (#et:Type0) {| floating et |}
  (n:pos) (na:nat{n * n <= na}) (k:nat) (nk:nat{nk == n * k})
  (sA : lseq et na) (sB : lseq et nk) (sX : lseq et nk)
  : Lemma (requires (forall (c':nat) (i':nat). c' < k /\ i' < n ==>
                       Seq.index sX (fidx n k c' i')
                       == ftrsv_at n na sA (bcol n k (sB <: lseq et (n * k)) c') i'))
          (ensures sX == ftrsm n na k nk sA sB)
  = introduce forall (idx:nat{idx < nk}).
                Seq.index sX idx == Seq.index (ftrsm n na k nk sA sB) idx
    with (div_lt idx n k; ML.lemma_div_mod idx n;
          ftrsm_index n na k nk sA sB (idx / n) (idx % n));
    Seq.lemma_eq_intro sX (ftrsm n na k nk sA sB)

(* ----------------------------------------------------------------------- *)
(* Per-row computation for one column (offset c*n into B/X).                 *)
(* ----------------------------------------------------------------------- *)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 80"
inline_for_extraction noextract
fn trsm_row
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (gA : array1 et (l1_forward (n *^ n)))
  (gB : array1 et (l1_forward (n *^ k)))
  (gX : array1 et (l1_forward (n *^ k)))
  (c : szlt k)
  (i : szlt n)
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sB : erased (lseq et (SZ.v (n *^ k))))
  (#sX : erased (lseq et (SZ.v (n *^ k))))
  (#fa #fb #fx : perm)
  preserves
    gpu **
    gA |-> Frac fa sA **
    gB |-> Frac fb sB **
    gX |-> Frac fx sX
  requires
    pure (forall (j:nat). j < SZ.v i ==>
            Seq.index sX (fidx (SZ.v n) (SZ.v k) (SZ.v c) j)
            == ftrsv_at (SZ.v n) (SZ.v (n *^ n)) sA (bcol (SZ.v n) (SZ.v k) sB (SZ.v c)) j)
  returns
    xi : et
  ensures
    pure (xi == ftrsv_at (SZ.v n) (SZ.v (n *^ n)) sA (bcol (SZ.v n) (SZ.v k) sB (SZ.v c)) (SZ.v i))
{
  let co = c *^ n;
  let mut s : et = Array1.(gB.(co +^ i));
  let mut j : szle i = 0sz;

  while (!j <^ i)
    invariant live s
    invariant live j
    invariant pure (SZ.v !j <= SZ.v i /\
                    !s == srow (SZ.v n) (SZ.v (n *^ n)) sA
                            (bcol (SZ.v n) (SZ.v k) sB (SZ.v c)) (SZ.v i) (SZ.v !j))
    decreases (SZ.v i - SZ.v !j)
  {
    let vj = !j;
    let aij = Array1.(gA.((i *^ n) +^ vj));
    let xj = Array1.(gX.(co +^ vj));
    s := !s `sub` (aij `mul` xj);
    j := !j +^ 1sz;
  };

  let aii = Array1.(gA.((i *^ n) +^ i));
  !s `div` aii
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: solve column by column.                             *)
(* ----------------------------------------------------------------------- *)

(* Writing X[c*n+i] preserves the already-solved columns (< c) and the solved
   prefix of column c, and adds entry i of column c. *)
let upd_preserves_2d
  (#et:Type0) {| floating et |}
  (n:nat) (na:nat{n * n <= na}) (sA : lseq et na) (k:nat)
  (sB : lseq et (n * k)) (sX : lseq et (n * k))
  (c:nat{c < k}) (i:nat{i < n}) (xi : et)
  : Lemma
      (requires
        (forall (c':nat) (i':nat). c' < c /\ i' < n ==>
           Seq.index sX (fidx n k c' i') == ftrsv_at n na sA (bcol n k sB c') i') /\
        (forall (j:nat). j < i ==>
           Seq.index sX (fidx n k c j) == ftrsv_at n na sA (bcol n k sB c) j) /\
        xi == ftrsv_at n na sA (bcol n k sB c) i)
      (ensures
        (let sX' = Seq.upd sX (fidx n k c i) xi in
         (forall (c':nat) (i':nat). c' < c /\ i' < n ==>
            Seq.index sX' (fidx n k c' i') == ftrsv_at n na sA (bcol n k sB c') i') /\
         (forall (j:nat). j < i + 1 ==>
            Seq.index sX' (fidx n k c j) == ftrsv_at n na sA (bcol n k sB c) j)))
  = col_bound n k c;
    let sX' = Seq.upd sX (fidx n k c i) xi in
    introduce forall (c':nat) (i':nat). c' < c /\ i' < n ==>
                Seq.index sX' (fidx n k c' i') == ftrsv_at n na sA (bcol n k sB c') i'
    with introduce _ ==> _
    with _. (ML.lemma_mult_le_right n (c' + 1) c;
             ML.lemma_mult_le_right n (c' + 1) k;
             Seq.lemma_index_upd2 sX (c * n + i) xi (c' * n + i'));
    introduce forall (j:nat). j < i + 1 ==>
                Seq.index sX' (fidx n k c j) == ftrsv_at n na sA (bcol n k sB c) j
    with introduce _ ==> _
    with _. (col_bound n k c;
             if j < i
             then Seq.lemma_index_upd2 sX (c * n + i) xi (c * n + j)
             else Seq.lemma_index_upd1 sX (c * n + i) xi)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 100"
inline_for_extraction noextract
fn trsm_col
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#k : szp { SZ.fits (SZ.v n * SZ.v k) })
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
            == ftrsv_at (SZ.v n) (SZ.v (n *^ n)) sA (bcol (SZ.v n) (SZ.v k) sB c') i')
  ensures
    exists* (sX : lseq et (SZ.v (n *^ k))). gX |-> sX **
      pure (forall (c':nat) (i':nat). c' <= SZ.v c /\ i' < SZ.v n ==>
              Seq.index sX (fidx (SZ.v n) (SZ.v k) c' i')
              == ftrsv_at (SZ.v n) (SZ.v (n *^ n)) sA (bcol (SZ.v n) (SZ.v k) sB c') i')
{
  let mut i : szle n = 0sz;

  while (!i <^ n)
    invariant live i
    invariant exists* (sX : lseq et (SZ.v (n *^ k))). gX |-> sX **
      pure (SZ.v !i <= SZ.v n /\
            (forall (c':nat) (i':nat). c' < SZ.v c /\ i' < SZ.v n ==>
               Seq.index sX (fidx (SZ.v n) (SZ.v k) c' i')
               == ftrsv_at (SZ.v n) (SZ.v (n *^ n)) sA (bcol (SZ.v n) (SZ.v k) sB c') i') /\
            (forall (j:nat). j < SZ.v !i ==>
               Seq.index sX (fidx (SZ.v n) (SZ.v k) (SZ.v c) j)
               == ftrsv_at (SZ.v n) (SZ.v (n *^ n)) sA (bcol (SZ.v n) (SZ.v k) sB (SZ.v c)) j))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let xi = trsm_row gA gB gX c vi;
    with sxpre. assert (gX |-> sxpre);
    Array1.(gX.((c *^ n) +^ vi) <- xi);
    col_bound (SZ.v n) (SZ.v k) (SZ.v c);
    upd_preserves_2d (SZ.v n) (SZ.v (n *^ n)) sA (SZ.v k) sB sxpre (SZ.v c) (SZ.v vi) xi;
    i := !i +^ 1sz;
  };
}
#pop-options

#push-options "--fuel 2 --ifuel 1 --z3rlimit 100"
inline_for_extraction noextract
fn trsm_kf
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
  requires
    gpu ** gA |-> Frac fa sA ** gB |-> Frac fb sB ** gX |-> sX0
  ensures
    gpu ** gA |-> Frac fa sA ** gB |-> Frac fb sB **
    gX |-> ftrsm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) sA sB
{
  let mut c : szle k = 0sz;

  while (!c <^ k)
    invariant live c
    invariant exists* (sX : lseq et (SZ.v (n *^ k))). gX |-> sX **
      pure (SZ.v !c <= SZ.v k /\
            (forall (c':nat) (i':nat). c' < SZ.v !c /\ i' < SZ.v n ==>
               Seq.index sX (fidx (SZ.v n) (SZ.v k) c' i')
               == ftrsv_at (SZ.v n) (SZ.v (n *^ n)) sA (bcol (SZ.v n) (SZ.v k) sB c') i'))
    decreases (SZ.v k - SZ.v !c)
  {
    let vc = !c;
    trsm_col gA gB gX vc;
    c := !c +^ 1sz;
  };

  with sXf. assert (gX |-> sXf);
  ftrsm_ext (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) sA sB sXf;
  ()
}
#pop-options


(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper (single-thread kernel_desc_n, like Klas.Trsv).           *)
(* Clean output gX |-> ftrsm; the per-column guarantee (column c of X is the  *)
(* trsv solution of column c of B) is exposed as a pure fact, and its real    *)
(* approximation follows from Klas.Trsv.ftrsv_at_approx applied per column.   *)
(* ----------------------------------------------------------------------- *)

(* Every solved cell equals the per-column trsv solution. *)
let trsm_post_all (#et:Type0) {| floating et |}
  (n:pos) (na:nat{n * n <= na}) (k:nat) (nk:nat{nk == n * k})
  (sA : lseq et na) (sB : lseq et nk)
  : Lemma (ensures forall (c:nat) (i:nat). c < k /\ i < n ==>
             Seq.index (ftrsm n na k nk sA sB) (fidx n k c i)
             == ftrsv_at n na sA (bcol n k (sB <: lseq et (n * k)) c) i)
  = ()

ghost
fn trsm_setup
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
fn trsm_teardown
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#k : szp { SZ.fits (SZ.v n * SZ.v k) })
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
       gX |-> ftrsm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) sA sB) ** emp
  ensures
    gA |-> Frac fa sA ** gB |-> Frac fb sB **
    gX |-> ftrsm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) sA sB
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gB |-> Frac fb sB **
       gX |-> ftrsm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) sA sB);
}

inline_for_extraction noextract
let kamtrsm
  (#et:Type0) {| floating et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (k : szp { SZ.fits (SZ.v n * SZ.v k) })
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
                gX |-> ftrsm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) sA sB)
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> trsm_kf gA gB gX #sA #sB #sX0 #fa #fb);

    frame    = emp;
    teardown = trsm_teardown gA gB gX;
    setup    = trsm_setup gA gB gX;
    kpre  = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gB |-> Frac fb sB ** gX |-> sX0);
    kpost = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gB |-> Frac fb sB **
               gX |-> ftrsm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) sA sB);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn trsm_gen
  (#et:Type0) {| floating et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (k : szp { SZ.fits (SZ.v n * SZ.v k) })
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
    on gpu_loc (gX |-> ftrsm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) sA sB) **
    pure (forall (c:nat) (i:nat). c < SZ.v k /\ i < SZ.v n ==>
            Seq.index (ftrsm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) sA sB) (fidx (SZ.v n) (SZ.v k) c i)
            == ftrsv_at (SZ.v n) (SZ.v (n *^ n)) sA (bcol (SZ.v n) (SZ.v k) sB c) i)
{
  on_star_eq gpu_loc (gB |-> Frac fb sB) (gX |-> sX0);
  rewrite (on gpu_loc (gB |-> Frac fb sB) ** on gpu_loc (gX |-> sX0))
       as (on gpu_loc ((gB |-> Frac fb sB) ** (gX |-> sX0)));
  on_star_eq gpu_loc (gA |-> Frac fa sA) ((gB |-> Frac fb sB) ** (gX |-> sX0));
  rewrite (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gB |-> Frac fb sB) ** (gX |-> sX0)))
       as (on gpu_loc ((gA |-> Frac fa sA) ** ((gB |-> Frac fb sB) ** (gX |-> sX0))));

  launch_sync (kamtrsm n k gA gB gX #() #sA #sB #sX0 #fa #fb);

  on_star_eq gpu_loc (gA |-> Frac fa sA)
             ((gB |-> Frac fb sB) ** (gX |-> ftrsm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) sA sB));
  rewrite (on gpu_loc ((gA |-> Frac fa sA) ** ((gB |-> Frac fb sB) ** (gX |-> ftrsm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) sA sB))))
       as (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gB |-> Frac fb sB) ** (gX |-> ftrsm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) sA sB)));
  on_star_eq gpu_loc (gB |-> Frac fb sB) (gX |-> ftrsm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) sA sB);
  rewrite (on gpu_loc ((gB |-> Frac fb sB) ** (gX |-> ftrsm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) sA sB)))
       as (on gpu_loc (gB |-> Frac fb sB) ** on gpu_loc (gX |-> ftrsm (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) sA sB));

  trsm_post_all (SZ.v n) (SZ.v (n *^ n)) (SZ.v k) (SZ.v (n *^ k)) sA sB;
  ()
}

let trsm_f32 = trsm_gen #f32
let trsm_f64 = trsm_gen #f64
