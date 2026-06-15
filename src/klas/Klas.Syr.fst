module Klas.Syr

(* cuBLAS syr / syr2 building block: symmetric rank-1 update

       C := A + alpha * x * x^T,   i.e.  C[i][j] = A[i][j] + alpha*x[i]*x[j].

   A and C are n x n row-major (length n*n); x is length n. cuBLAS syr updates
   only one triangle in place; we compute the whole (symmetric) matrix into a
   separate output C, which is functionally equivalent and simpler to verify.
   Each cell is independent, so a single thread with a double loop suffices.

   syr2 ( A + alpha*(x*y^T + y*x^T) ) is two such updates; that and the rank-1
   ger case (separate x,y) are the natural generalizations. Reuses the flat
   index helpers fidx / div_lt from Klas.Trsm. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Klas.Trsm { fidx, div_lt }
open Klas.Trsv { idx_bound }
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* ----------------------------------------------------------------------- *)
(* Spec: the rank-1 updated cell and the whole matrix.                       *)
(* ----------------------------------------------------------------------- *)

let fsyr_at (#et:Type0) {| floating et |}
  (n:nat) (nn:nat{nn == n * n}) (alpha : et) (sA : lseq et nn) (sx : lseq et n)
  (i:nat{i < n}) (j:nat{j < n}) : et
  = (Seq.index sA (fidx n n i j)) `add` (alpha `mul` ((Seq.index sx i) `mul` (Seq.index sx j)))

let rsyr_at
  (n:nat) (nn:nat{nn == n * n}) (ralpha : real) (rA : lseq real nn) (rx : lseq real n)
  (i:nat{i < n}) (j:nat{j < n}) : real
  = (Seq.index rA (fidx n n i j)) +. (ralpha *. ((Seq.index rx i) *. (Seq.index rx j)))

noextract
let fsyr (#et:Type0) {| floating et |}
  (n:pos) (nn:nat{nn == n * n}) (alpha : et) (sA : lseq et nn) (sx : lseq et n) : lseq et nn
  = Seq.init nn (fun (idx:nat{idx < nn}) ->
      div_lt idx n n; fsyr_at n nn alpha sA sx (idx / n) (idx % n))

let fsyr_index (#et:Type0) {| floating et |}
  (n:pos) (nn:nat{nn == n * n}) (alpha : et) (sA : lseq et nn) (sx : lseq et n)
  (i:nat{i < n}) (j:nat{j < n})
  : Lemma (ensures Seq.index (fsyr n nn alpha sA sx) (fidx n n i j) == fsyr_at n nn alpha sA sx i j)
          [SMTPat (Seq.index (fsyr n nn alpha sA sx) (fidx n n i j))]
  = (* fidx n n i j == i*n+j; then (i*n+j)/n = i and (i*n+j)%n = j since j<n *)
    FStar.Math.Lemmas.lemma_div_plus j i n;
    FStar.Math.Lemmas.small_div j n;
    FStar.Math.Lemmas.lemma_mod_plus j i n;
    FStar.Math.Lemmas.small_mod j n

let fsyr_at_approx
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (n:nat) (nn:nat{nn == n * n}) (alpha : et) (ralpha : real)
  (sA : lseq et nn) (sx : lseq et n) (rA : lseq real nn) (rx : lseq real n)
  (i:nat{i < n}) (j:nat{j < n})
  : Lemma (requires sA %~ rA /\ sx %~ rx /\ alpha %~ ralpha)
          (ensures fsyr_at n nn alpha sA sx i j %~ rsyr_at n nn ralpha rA rx i j)
  = assert ((sA @! (fidx n n i j)) %~ (rA @! (fidx n n i j)));
    assert ((sx @! i) %~ (rx @! i));
    assert ((sx @! j) %~ (rx @! j))

let syr_post
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (n:pos) (nn:nat{nn == n * n}) (alpha : et) (ralpha : real)
  (sA : lseq et nn) (sx : lseq et n) (rA : lseq real nn) (rx : lseq real n) : prop
  = forall (i:nat) (j:nat). i < n /\ j < n ==>
      (Seq.index (fsyr n nn alpha sA sx) (fidx n n i j)) %~ rsyr_at n nn ralpha rA rx i j

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: one independent cell per (i,j).                     *)
(* ----------------------------------------------------------------------- *)

(* Writing cell (i,j) preserves the already-done rows (< i) and the done prefix
   of row i (cols < j). *)
let syr_upd_2d (#et:Type0) {| floating et |}
  (n:pos) (nn:nat{nn == n * n}) (alpha : et) (sA : lseq et nn) (sx : lseq et n)
  (sC : lseq et nn) (i:nat{i < n}) (j:nat{j < n}) (cij : et)
  : Lemma
      (requires
        (forall (i':nat) (j':nat). i' < i /\ j' < n ==>
           Seq.index sC (fidx n n i' j') == fsyr_at n nn alpha sA sx i' j') /\
        (forall (j':nat). j' < j ==>
           Seq.index sC (fidx n n i j') == fsyr_at n nn alpha sA sx i j') /\
        cij == fsyr_at n nn alpha sA sx i j)
      (ensures
        (let sC' = Seq.upd sC (fidx n n i j) cij in
         (forall (i':nat) (j':nat). i' < i /\ j' < n ==>
            Seq.index sC' (fidx n n i' j') == fsyr_at n nn alpha sA sx i' j') /\
         (forall (j':nat). j' < j + 1 ==>
            Seq.index sC' (fidx n n i j') == fsyr_at n nn alpha sA sx i j')))
  = let sC' = Seq.upd sC (fidx n n i j) cij in
    introduce forall (i':nat) (j':nat). i' < i /\ j' < n ==>
                Seq.index sC' (fidx n n i' j') == fsyr_at n nn alpha sA sx i' j'
    with introduce _ ==> _
    with _. (ML.lemma_mult_le_right n (i' + 1) i;
             Seq.lemma_index_upd2 sC (fidx n n i j) cij (fidx n n i' j'));
    introduce forall (j':nat). j' < j + 1 ==>
                Seq.index sC' (fidx n n i j') == fsyr_at n nn alpha sA sx i j'
    with introduce _ ==> _
    with _. (if j' < j
             then Seq.lemma_index_upd2 sC (fidx n n i j) cij (fidx n n i j')
             else Seq.lemma_index_upd1 sC (fidx n n i j) cij)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 100"
inline_for_extraction noextract
fn syr_kf
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (alpha : et)
  (gA : array1 et (l1_forward (n *^ n)))
  (gx : array1 et (l1_forward n))
  (gC : array1 et (l1_forward (n *^ n)))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sx : erased (lseq et (SZ.v n)))
  (#sC0 : erased (lseq et (SZ.v (n *^ n))))
  (#fa #fx : perm)
  ()
  norewrite
  requires
    gpu ** gA |-> Frac fa sA ** gx |-> Frac fx sx ** gC |-> sC0
  ensures
    gpu ** gA |-> Frac fa sA ** gx |-> Frac fx sx **
    gC |-> fsyr (SZ.v n) (SZ.v (n *^ n)) alpha sA sx
{
  let mut i : szle n = 0sz;

  while (!i <^ n)
    invariant live i
    invariant exists* (sC : lseq et (SZ.v (n *^ n))). gC |-> sC **
      pure (SZ.v !i <= SZ.v n /\
            (forall (i':nat) (j':nat). i' < SZ.v !i /\ j' < SZ.v n ==>
               Seq.index sC (fidx (SZ.v n) (SZ.v n) i' j')
               == fsyr_at (SZ.v n) (SZ.v (n *^ n)) alpha sA sx i' j'))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let mut j : szle n = 0sz;

    while (!j <^ n)
      invariant live j
      invariant exists* (sC : lseq et (SZ.v (n *^ n))). gC |-> sC **
        pure (SZ.v !j <= SZ.v n /\
              (forall (i':nat) (j':nat). i' < SZ.v vi /\ j' < SZ.v n ==>
                 Seq.index sC (fidx (SZ.v n) (SZ.v n) i' j')
                 == fsyr_at (SZ.v n) (SZ.v (n *^ n)) alpha sA sx i' j') /\
              (forall (j':nat). j' < SZ.v !j ==>
                 Seq.index sC (fidx (SZ.v n) (SZ.v n) (SZ.v vi) j')
                 == fsyr_at (SZ.v n) (SZ.v (n *^ n)) alpha sA sx (SZ.v vi) j'))
      decreases (SZ.v n - SZ.v !j)
    {
      let vj = !j;
      idx_bound (SZ.v n) (SZ.v vi) (SZ.v vj);
      let aij = Array1.(gA.((vi *^ n) +^ vj));
      let xi = Array1.(gx.(vi));
      let xj = Array1.(gx.(vj));
      let cij = aij `add` (alpha `mul` (xi `mul` xj));
      with sCp. assert (gC |-> sCp);
      Array1.(gC.((vi *^ n) +^ vj) <- cij);
      syr_upd_2d (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sCp (SZ.v vi) (SZ.v vj) cij;
      j := !j +^ 1sz;
    };
    i := !i +^ 1sz;
  };

  with sCf. assert (gC |-> sCf);
  Seq.lemma_eq_intro sCf (fsyr (SZ.v n) (SZ.v (n *^ n)) alpha sA sx);
  ()
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper (single-thread kernel_desc_n, like Klas.Trsv).           *)
(* ----------------------------------------------------------------------- *)

let fsyr_approx_all
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (n:pos) (nn:nat{nn == n * n}) (alpha : et) (ralpha : real)
  (sA : lseq et nn) (sx : lseq et n) (rA : lseq real nn) (rx : lseq real n)
  : Lemma (requires sA %~ rA /\ sx %~ rx /\ alpha %~ ralpha)
          (ensures syr_post n nn alpha ralpha sA sx rA rx)
  = introduce forall (i:nat) (j:nat). i < n /\ j < n ==>
                (Seq.index (fsyr n nn alpha sA sx) (fidx n n i j)) %~ rsyr_at n nn ralpha rA rx i j
    with introduce _ ==> _
    with _. (fsyr_at_approx n nn alpha ralpha sA sx rA rx i j; fsyr_index n nn alpha sA sx i j)

ghost
fn syr_setup
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (gA : array1 et (l1_forward (n *^ n)))
  (gx : array1 et (l1_forward n))
  (gC : array1 et (l1_forward (n *^ n)))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sx : erased (lseq et (SZ.v n)))
  (#sC0 : erased (lseq et (SZ.v (n *^ n))))
  (#fa #fx : perm)
  ()
  norewrite
  requires gA |-> Frac fa sA ** gx |-> Frac fx sx ** gC |-> sC0
  ensures (forall+ (tid : natlt 1sz).
             gA |-> Frac fa sA ** gx |-> Frac fx sx ** gC |-> sC0) ** emp
{
  forevery_singleton_intro #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx ** gC |-> sC0);
}

ghost
fn syr_teardown
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (alpha : et)
  (gA : array1 et (l1_forward (n *^ n)))
  (gx : array1 et (l1_forward n))
  (gC : array1 et (l1_forward (n *^ n)))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sx : erased (lseq et (SZ.v n)))
  (#fa #fx : perm)
  ()
  norewrite
  requires
    (forall+ (tid : natlt 1sz).
       gA |-> Frac fa sA ** gx |-> Frac fx sx **
       gC |-> fsyr (SZ.v n) (SZ.v (n *^ n)) alpha sA sx) ** emp
  ensures
    gA |-> Frac fa sA ** gx |-> Frac fx sx **
    gC |-> fsyr (SZ.v n) (SZ.v (n *^ n)) alpha sA sx
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx **
       gC |-> fsyr (SZ.v n) (SZ.v (n *^ n)) alpha sA sx);
}

inline_for_extraction noextract
let kamsyr
  (#et:Type0) {| floating et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (alpha : et)
  (gA : array1 et (l1_forward (n *^ n)))
  (gx : array1 et (l1_forward n))
  (gC : array1 et (l1_forward (n *^ n)))
  (#_ : squash (Array1.is_global gA /\ Array1.is_global gx /\ Array1.is_global gC))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sx : erased (lseq et (SZ.v n)))
  (#sC0 : erased (lseq et (SZ.v (n *^ n))))
  (#fa #fx : perm)
  : kernel_desc
      (requires gA |-> Frac fa sA ** gx |-> Frac fx sx ** gC |-> sC0)
      (ensures  gA |-> Frac fa sA ** gx |-> Frac fx sx **
                gC |-> fsyr (SZ.v n) (SZ.v (n *^ n)) alpha sA sx)
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> syr_kf alpha gA gx gC #sA #sx #sC0 #fa #fx);
    frame    = emp;
    teardown = syr_teardown alpha gA gx gC;
    setup    = syr_setup gA gx gC;
    kpre  = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx ** gC |-> sC0);
    kpost = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx **
                                     gC |-> fsyr (SZ.v n) (SZ.v (n *^ n)) alpha sA sx);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn syr_gen
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (alpha : et)
  (gA : array1 et (l1_forward (n *^ n)) { Array1.is_global gA })
  (gx : array1 et (l1_forward n) { Array1.is_global gx })
  (gC : array1 et (l1_forward (n *^ n)) { Array1.is_global gC })
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sx : erased (lseq et (SZ.v n)))
  (#sC0 : erased (lseq et (SZ.v (n *^ n))))
  (ralpha : erased real)
  (rA : erased (lseq real (SZ.v (n *^ n))))
  (rx : erased (lseq real (SZ.v n)))
  (#fa #fx : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fa sA) **
    on gpu_loc (gx |-> Frac fx sx)
  requires
    on gpu_loc (gC |-> sC0) **
    pure (sA %~ rA /\ sx %~ rx /\ alpha %~ ralpha)
  ensures
    on gpu_loc (gC |-> fsyr (SZ.v n) (SZ.v (n *^ n)) alpha sA sx) **
    pure (syr_post (SZ.v n) (SZ.v (n *^ n)) alpha (Ghost.reveal ralpha) sA sx rA rx)
{
  on_star_eq gpu_loc (gx |-> Frac fx sx) (gC |-> sC0);
  rewrite (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc (gC |-> sC0))
       as (on gpu_loc ((gx |-> Frac fx sx) ** (gC |-> sC0)));
  on_star_eq gpu_loc (gA |-> Frac fa sA) ((gx |-> Frac fx sx) ** (gC |-> sC0));
  rewrite (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gx |-> Frac fx sx) ** (gC |-> sC0)))
       as (on gpu_loc ((gA |-> Frac fa sA) ** ((gx |-> Frac fx sx) ** (gC |-> sC0))));

  launch_sync (kamsyr n alpha gA gx gC #() #sA #sx #sC0 #fa #fx);

  on_star_eq gpu_loc (gA |-> Frac fa sA)
             ((gx |-> Frac fx sx) ** (gC |-> fsyr (SZ.v n) (SZ.v (n *^ n)) alpha sA sx));
  rewrite (on gpu_loc ((gA |-> Frac fa sA) ** ((gx |-> Frac fx sx) ** (gC |-> fsyr (SZ.v n) (SZ.v (n *^ n)) alpha sA sx))))
       as (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gx |-> Frac fx sx) ** (gC |-> fsyr (SZ.v n) (SZ.v (n *^ n)) alpha sA sx)));
  on_star_eq gpu_loc (gx |-> Frac fx sx) (gC |-> fsyr (SZ.v n) (SZ.v (n *^ n)) alpha sA sx);
  rewrite (on gpu_loc ((gx |-> Frac fx sx) ** (gC |-> fsyr (SZ.v n) (SZ.v (n *^ n)) alpha sA sx)))
       as (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc (gC |-> fsyr (SZ.v n) (SZ.v (n *^ n)) alpha sA sx));

  fsyr_approx_all (SZ.v n) (SZ.v (n *^ n)) alpha (Ghost.reveal ralpha) sA sx rA rx;
  ()
}

let syr_f32 = syr_gen #f32
let syr_f64 = syr_gen #f64
