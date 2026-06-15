module Klas.Syrk

(* cuBLAS syrk (uplo = lower, trans = N) : symmetric rank-k update

       X := alpha * A * A^T + beta * C,

   with A an n x k matrix (row-major, length n*k), C and X both n x n
   (row-major / equivalently column-major: A*A^T is symmetric). Out-of-place
   output X (cuBLAS is in-place and updates one triangle; the separate full
   output is functionally equivalent and simpler to verify). Each cell is a dot
   product of two rows of A -- both reads come from the SAME array A, so no
   transpose view / aliasing is needed:

       X[i*n+j] = alpha * (sum_l A[i*k+l] * A[j*k+l]) + beta * C[i*n+j].

   Reuses the flat index helpers fidx / col_bound / div_lt from Klas.Trsm. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Klas.Trsm { fidx, col_bound, div_lt }
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* ----------------------------------------------------------------------- *)
(* Spec: row i . row j of A, the cell, and the whole matrix.                 *)
(* ----------------------------------------------------------------------- *)

(* Flat index into an n x k row-major A: row r, col l. *)
let arow_bound (n k r l : nat)
  : Lemma (requires r < n /\ l < k) (ensures r * k + l < n * k)
  = ML.lemma_mult_le_right k (r + 1) n

let rec arow_dot (#et:Type0) {| floating et |}
  (n:nat) (k:nat) (sA : lseq et (n * k)) (i:nat{i < n}) (j:nat{j < n}) (m:nat{m <= k})
  : Tot et (decreases m)
  = if m = 0 then zero
    else (arow_bound n k i (m - 1); arow_bound n k j (m - 1);
          (arow_dot n k sA i j (m - 1))
          `add` ((Seq.index sA (i * k + (m - 1))) `mul` (Seq.index sA (j * k + (m - 1)))))

let fsyrk_at (#et:Type0) {| floating et |}
  (n:pos) (k:nat) (nn:nat{nn == n * n}) (alpha beta : et)
  (sA : lseq et (n * k)) (sC : lseq et nn) (i:nat{i < n}) (j:nat{j < n}) : et
  = (alpha `mul` (arow_dot n k sA i j k)) `add` (beta `mul` (Seq.index sC (fidx n n i j)))

noextract
let fsyrk (#et:Type0) {| floating et |}
  (n:pos) (k:nat) (nn:nat{nn == n * n}) (alpha beta : et)
  (sA : lseq et (n * k)) (sC : lseq et nn) : lseq et nn
  = Seq.init nn (fun (idx:nat{idx < nn}) ->
      div_lt idx n n; fsyrk_at n k nn alpha beta sA sC (idx / n) (idx % n))

let fsyrk_index (#et:Type0) {| floating et |}
  (n:pos) (k:nat) (nn:nat{nn == n * n}) (alpha beta : et)
  (sA : lseq et (n * k)) (sC : lseq et nn) (i:nat{i < n}) (j:nat{j < n})
  : Lemma (ensures Seq.index (fsyrk n k nn alpha beta sA sC) (fidx n n i j)
                   == fsyrk_at n k nn alpha beta sA sC i j)
          [SMTPat (Seq.index (fsyrk n k nn alpha beta sA sC) (fidx n n i j))]
  = ML.lemma_div_plus j i n; ML.small_div j n; ML.lemma_mod_plus j i n; ML.small_mod j n

(* ----------------------------------------------------------------------- *)
(* Per-cell dot of rows i and j of A.                                        *)
(* ----------------------------------------------------------------------- *)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 100"
inline_for_extraction noextract
fn syrk_dot
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (gA : array1 et (l1_forward (n *^ k)))
  (i : szlt n)
  (j : szlt n)
  (#sA : erased (lseq et (SZ.v (n *^ k))))
  (#fa : perm)
  preserves
    gpu ** gA |-> Frac fa sA
  returns
    s : et
  ensures
    pure (s == arow_dot (SZ.v n) (SZ.v k) sA (SZ.v i) (SZ.v j) (SZ.v k))
{
  let mut s : et = zero;
  let mut l : szle k = 0sz;

  while (!l <^ k)
    invariant live s
    invariant live l
    invariant pure (SZ.v !l <= SZ.v k /\
                    !s == arow_dot (SZ.v n) (SZ.v k) sA (SZ.v i) (SZ.v j) (SZ.v !l))
    decreases (SZ.v k - SZ.v !l)
  {
    let vl = !l;
    arow_bound (SZ.v n) (SZ.v k) (SZ.v i) (SZ.v vl);
    arow_bound (SZ.v n) (SZ.v k) (SZ.v j) (SZ.v vl);
    let ail = Array1.(gA.((i *^ k) +^ vl));
    let ajl = Array1.(gA.((j *^ k) +^ vl));
    s := !s `add` (ail `mul` ajl);
    l := !l +^ 1sz;
  };
  !s
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: one independent cell per (i,j), fresh output X.     *)
(* ----------------------------------------------------------------------- *)

let syrk_upd_2d (#et:Type0) {| floating et |}
  (n:pos) (k:nat) (nn:nat{nn == n * n}) (alpha beta : et)
  (sA : lseq et (n * k)) (sC : lseq et nn) (sX : lseq et nn)
  (i:nat{i < n}) (j:nat{j < n}) (xij : et)
  : Lemma
      (requires
        (forall (i':nat) (j':nat). i' < i /\ j' < n ==>
           Seq.index sX (fidx n n i' j') == fsyrk_at n k nn alpha beta sA sC i' j') /\
        (forall (j':nat). j' < j ==>
           Seq.index sX (fidx n n i j') == fsyrk_at n k nn alpha beta sA sC i j') /\
        xij == fsyrk_at n k nn alpha beta sA sC i j)
      (ensures
        (let sX' = Seq.upd sX (fidx n n i j) xij in
         (forall (i':nat) (j':nat). i' < i /\ j' < n ==>
            Seq.index sX' (fidx n n i' j') == fsyrk_at n k nn alpha beta sA sC i' j') /\
         (forall (j':nat). j' < j + 1 ==>
            Seq.index sX' (fidx n n i j') == fsyrk_at n k nn alpha beta sA sC i j')))
  = let sX' = Seq.upd sX (fidx n n i j) xij in
    introduce forall (i':nat) (j':nat). i' < i /\ j' < n ==>
                Seq.index sX' (fidx n n i' j') == fsyrk_at n k nn alpha beta sA sC i' j'
    with introduce _ ==> _
    with _. (ML.lemma_mult_le_right n (i' + 1) i;
             Seq.lemma_index_upd2 sX (fidx n n i j) xij (fidx n n i' j'));
    introduce forall (j':nat). j' < j + 1 ==>
                Seq.index sX' (fidx n n i j') == fsyrk_at n k nn alpha beta sA sC i j'
    with introduce _ ==> _
    with _. (if j' < j
             then Seq.lemma_index_upd2 sX (fidx n n i j) xij (fidx n n i j')
             else Seq.lemma_index_upd1 sX (fidx n n i j) xij)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 120"
inline_for_extraction noextract
fn syrk_kf
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (alpha beta : et)
  (gA : array1 et (l1_forward (n *^ k)))
  (gC : array1 et (l1_forward (n *^ n)))
  (gX : array1 et (l1_forward (n *^ n)))
  (#sA : erased (lseq et (SZ.v (n *^ k))))
  (#sC : erased (lseq et (SZ.v (n *^ n))))
  (#sX0 : erased (lseq et (SZ.v (n *^ n))))
  (#fa #fc : perm)
  ()
  norewrite
  requires
    gpu ** gA |-> Frac fa sA ** gC |-> Frac fc sC ** gX |-> sX0
  ensures
    gpu ** gA |-> Frac fa sA ** gC |-> Frac fc sC **
    gX |-> fsyrk (SZ.v n) (SZ.v k) (SZ.v (n *^ n)) alpha beta sA sC
{
  let mut i : szle n = 0sz;

  while (!i <^ n)
    invariant live i
    invariant exists* (sX : lseq et (SZ.v (n *^ n))). gX |-> sX **
      pure (SZ.v !i <= SZ.v n /\
            (forall (i':nat) (j':nat). i' < SZ.v !i /\ j' < SZ.v n ==>
               Seq.index sX (fidx (SZ.v n) (SZ.v n) i' j')
               == fsyrk_at (SZ.v n) (SZ.v k) (SZ.v (n *^ n)) alpha beta sA sC i' j'))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let mut j : szle n = 0sz;

    while (!j <^ n)
      invariant live j
      invariant exists* (sX : lseq et (SZ.v (n *^ n))). gX |-> sX **
        pure (SZ.v !j <= SZ.v n /\
              (forall (i':nat) (j':nat). i' < SZ.v vi /\ j' < SZ.v n ==>
                 Seq.index sX (fidx (SZ.v n) (SZ.v n) i' j')
                 == fsyrk_at (SZ.v n) (SZ.v k) (SZ.v (n *^ n)) alpha beta sA sC i' j') /\
              (forall (j':nat). j' < SZ.v !j ==>
                 Seq.index sX (fidx (SZ.v n) (SZ.v n) (SZ.v vi) j')
                 == fsyrk_at (SZ.v n) (SZ.v k) (SZ.v (n *^ n)) alpha beta sA sC (SZ.v vi) j'))
      decreases (SZ.v n - SZ.v !j)
    {
      let vj = !j;
      let s = syrk_dot gA vi vj;
      let cij = Array1.(gC.((vi *^ n) +^ vj));
      let xij = (alpha `mul` s) `add` (beta `mul` cij);
      with sxpre. assert (gX |-> sxpre);
      Array1.(gX.((vi *^ n) +^ vj) <- xij);
      syrk_upd_2d (SZ.v n) (SZ.v k) (SZ.v (n *^ n)) alpha beta sA sC sxpre (SZ.v vi) (SZ.v vj) xij;
      j := !j +^ 1sz;
    };
    i := !i +^ 1sz;
  };

  with sXf. assert (gX |-> sXf);
  Seq.lemma_eq_intro sXf (fsyrk (SZ.v n) (SZ.v k) (SZ.v (n *^ n)) alpha beta sA sC);
  ()
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper (single-thread kernel_desc_n, like Klas.Symm).           *)
(* ----------------------------------------------------------------------- *)

ghost
fn syrk_setup
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (gA : array1 et (l1_forward (n *^ k)))
  (gC : array1 et (l1_forward (n *^ n)))
  (gX : array1 et (l1_forward (n *^ n)))
  (#sA : erased (lseq et (SZ.v (n *^ k))))
  (#sC : erased (lseq et (SZ.v (n *^ n))))
  (#sX0 : erased (lseq et (SZ.v (n *^ n))))
  (#fa #fc : perm)
  ()
  norewrite
  requires gA |-> Frac fa sA ** gC |-> Frac fc sC ** gX |-> sX0
  ensures (forall+ (tid : natlt 1sz).
             gA |-> Frac fa sA ** gC |-> Frac fc sC ** gX |-> sX0) ** emp
{
  forevery_singleton_intro #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gC |-> Frac fc sC ** gX |-> sX0);
}

ghost
fn syrk_teardown
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (alpha beta : et)
  (gA : array1 et (l1_forward (n *^ k)))
  (gC : array1 et (l1_forward (n *^ n)))
  (gX : array1 et (l1_forward (n *^ n)))
  (#sA : erased (lseq et (SZ.v (n *^ k))))
  (#sC : erased (lseq et (SZ.v (n *^ n))))
  (#fa #fc : perm)
  ()
  norewrite
  requires
    (forall+ (tid : natlt 1sz).
       gA |-> Frac fa sA ** gC |-> Frac fc sC **
       gX |-> fsyrk (SZ.v n) (SZ.v k) (SZ.v (n *^ n)) alpha beta sA sC) ** emp
  ensures
    gA |-> Frac fa sA ** gC |-> Frac fc sC **
    gX |-> fsyrk (SZ.v n) (SZ.v k) (SZ.v (n *^ n)) alpha beta sA sC
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gC |-> Frac fc sC **
       gX |-> fsyrk (SZ.v n) (SZ.v k) (SZ.v (n *^ n)) alpha beta sA sC);
}

inline_for_extraction noextract
let kamsyrk
  (#et:Type0) {| floating et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (alpha beta : et)
  (gA : array1 et (l1_forward (n *^ k)))
  (gC : array1 et (l1_forward (n *^ n)))
  (gX : array1 et (l1_forward (n *^ n)))
  (#_ : squash (Array1.is_global gA /\ Array1.is_global gC /\ Array1.is_global gX))
  (#sA : erased (lseq et (SZ.v (n *^ k))))
  (#sC : erased (lseq et (SZ.v (n *^ n))))
  (#sX0 : erased (lseq et (SZ.v (n *^ n))))
  (#fa #fc : perm)
  : kernel_desc
      (requires gA |-> Frac fa sA ** gC |-> Frac fc sC ** gX |-> sX0)
      (ensures  gA |-> Frac fa sA ** gC |-> Frac fc sC **
                gX |-> fsyrk (SZ.v n) (SZ.v k) (SZ.v (n *^ n)) alpha beta sA sC)
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> syrk_kf alpha beta gA gC gX #sA #sC #sX0 #fa #fc);
    frame    = emp;
    teardown = syrk_teardown alpha beta gA gC gX;
    setup    = syrk_setup gA gC gX;
    kpre  = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gC |-> Frac fc sC ** gX |-> sX0);
    kpost = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gC |-> Frac fc sC **
                                     gX |-> fsyrk (SZ.v n) (SZ.v k) (SZ.v (n *^ n)) alpha beta sA sC);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn syrk_gen
  (#et:Type0) {| floating et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (k : szp { SZ.fits (SZ.v n * SZ.v k) })
  (alpha beta : et)
  (gA : array1 et (l1_forward (n *^ k)) { Array1.is_global gA })
  (gC : array1 et (l1_forward (n *^ n)) { Array1.is_global gC })
  (gX : array1 et (l1_forward (n *^ n)) { Array1.is_global gX })
  (#sA : erased (lseq et (SZ.v (n *^ k))))
  (#sC : erased (lseq et (SZ.v (n *^ n))))
  (#sX0 : erased (lseq et (SZ.v (n *^ n))))
  (#fa #fc : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fa sA) **
    on gpu_loc (gC |-> Frac fc sC)
  requires
    on gpu_loc (gX |-> sX0)
  ensures
    on gpu_loc (gX |-> fsyrk (SZ.v n) (SZ.v k) (SZ.v (n *^ n)) alpha beta sA sC)
{
  on_star_eq gpu_loc (gC |-> Frac fc sC) (gX |-> sX0);
  rewrite (on gpu_loc (gC |-> Frac fc sC) ** on gpu_loc (gX |-> sX0))
       as (on gpu_loc ((gC |-> Frac fc sC) ** (gX |-> sX0)));
  on_star_eq gpu_loc (gA |-> Frac fa sA) ((gC |-> Frac fc sC) ** (gX |-> sX0));
  rewrite (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gC |-> Frac fc sC) ** (gX |-> sX0)))
       as (on gpu_loc ((gA |-> Frac fa sA) ** ((gC |-> Frac fc sC) ** (gX |-> sX0))));

  launch_sync (kamsyrk n k alpha beta gA gC gX #() #sA #sC #sX0 #fa #fc);

  on_star_eq gpu_loc (gA |-> Frac fa sA)
             ((gC |-> Frac fc sC) ** (gX |-> fsyrk (SZ.v n) (SZ.v k) (SZ.v (n *^ n)) alpha beta sA sC));
  rewrite (on gpu_loc ((gA |-> Frac fa sA) ** ((gC |-> Frac fc sC) ** (gX |-> fsyrk (SZ.v n) (SZ.v k) (SZ.v (n *^ n)) alpha beta sA sC))))
       as (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gC |-> Frac fc sC) ** (gX |-> fsyrk (SZ.v n) (SZ.v k) (SZ.v (n *^ n)) alpha beta sA sC)));
  on_star_eq gpu_loc (gC |-> Frac fc sC) (gX |-> fsyrk (SZ.v n) (SZ.v k) (SZ.v (n *^ n)) alpha beta sA sC);
  rewrite (on gpu_loc ((gC |-> Frac fc sC) ** (gX |-> fsyrk (SZ.v n) (SZ.v k) (SZ.v (n *^ n)) alpha beta sA sC)))
       as (on gpu_loc (gC |-> Frac fc sC) ** on gpu_loc (gX |-> fsyrk (SZ.v n) (SZ.v k) (SZ.v (n *^ n)) alpha beta sA sC));
  ()
}

let syrk_f32 = syrk_gen #f32
let syrk_f64 = syrk_gen #f64
