module Klas.Syr2

(* cuBLAS syr2 : symmetric rank-2 update

       C := A + alpha * (x*y^T + y*x^T),
       i.e.  C[i][j] = A[i][j] + alpha*(x[i]*y[j] + y[i]*x[j]).

   A and C are n x n row-major (length n*n); x, y are length n. cuBLAS syr2
   updates only one triangle in place; we compute the whole (symmetric) matrix
   into a separate output C, which is functionally equivalent and simpler to
   verify. Each cell is independent, so a single thread with a double loop
   suffices. This mirrors Klas.Syr with a second vector. Reuses the flat index
   helpers fidx / div_lt from Klas.Trsm and idx_bound from Klas.Trsv. *)

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
(* Spec: the rank-2 updated cell and the whole matrix.                       *)
(* ----------------------------------------------------------------------- *)

let fsyr2_at (#et:Type0) {| floating et |}
  (n:nat) (nn:nat{nn == n * n}) (alpha : et) (sA : lseq et nn) (sx sy : lseq et n)
  (i:nat{i < n}) (j:nat{j < n}) : et
  = (Seq.index sA (fidx n n i j)) `add`
    (alpha `mul` (((Seq.index sx i) `mul` (Seq.index sy j)) `add`
                  ((Seq.index sy i) `mul` (Seq.index sx j))))

let rsyr2_at
  (n:nat) (nn:nat{nn == n * n}) (ralpha : real) (rA : lseq real nn) (rx ry : lseq real n)
  (i:nat{i < n}) (j:nat{j < n}) : real
  = (Seq.index rA (fidx n n i j)) +.
    (ralpha *. (((Seq.index rx i) *. (Seq.index ry j)) +.
                ((Seq.index ry i) *. (Seq.index rx j))))

noextract
let fsyr2 (#et:Type0) {| floating et |}
  (n:pos) (nn:nat{nn == n * n}) (alpha : et) (sA : lseq et nn) (sx sy : lseq et n)
  : lseq et nn
  = Seq.init nn (fun (idx:nat{idx < nn}) ->
      div_lt idx n n; fsyr2_at n nn alpha sA sx sy (idx / n) (idx % n))

let fsyr2_index (#et:Type0) {| floating et |}
  (n:pos) (nn:nat{nn == n * n}) (alpha : et) (sA : lseq et nn) (sx sy : lseq et n)
  (i:nat{i < n}) (j:nat{j < n})
  : Lemma (ensures Seq.index (fsyr2 n nn alpha sA sx sy) (fidx n n i j)
                   == fsyr2_at n nn alpha sA sx sy i j)
          [SMTPat (Seq.index (fsyr2 n nn alpha sA sx sy) (fidx n n i j))]
  = ML.lemma_div_plus j i n; ML.small_div j n; ML.lemma_mod_plus j i n; ML.small_mod j n

let fsyr2_at_approx
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (n:nat) (nn:nat{nn == n * n}) (alpha : et) (ralpha : real)
  (sA : lseq et nn) (sx sy : lseq et n) (rA : lseq real nn) (rx ry : lseq real n)
  (i:nat{i < n}) (j:nat{j < n})
  : Lemma (requires sA %~ rA /\ sx %~ rx /\ sy %~ ry /\ alpha %~ ralpha)
          (ensures fsyr2_at n nn alpha sA sx sy i j %~ rsyr2_at n nn ralpha rA rx ry i j)
  = assert ((sA @! (fidx n n i j)) %~ (rA @! (fidx n n i j)));
    assert ((sx @! i) %~ (rx @! i));
    assert ((sx @! j) %~ (rx @! j));
    assert ((sy @! i) %~ (ry @! i));
    assert ((sy @! j) %~ (ry @! j))

let syr2_post
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (n:pos) (nn:nat{nn == n * n}) (alpha : et) (ralpha : real)
  (sA : lseq et nn) (sx sy : lseq et n) (rA : lseq real nn) (rx ry : lseq real n) : prop
  = forall (i:nat) (j:nat). i < n /\ j < n ==>
      (Seq.index (fsyr2 n nn alpha sA sx sy) (fidx n n i j))
      %~ rsyr2_at n nn ralpha rA rx ry i j

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: one independent cell per (i,j).                     *)
(* ----------------------------------------------------------------------- *)

let syr2_upd_2d (#et:Type0) {| floating et |}
  (n:pos) (nn:nat{nn == n * n}) (alpha : et) (sA : lseq et nn) (sx sy : lseq et n)
  (sC : lseq et nn) (i:nat{i < n}) (j:nat{j < n}) (cij : et)
  : Lemma
      (requires
        (forall (i':nat) (j':nat). i' < i /\ j' < n ==>
           Seq.index sC (fidx n n i' j') == fsyr2_at n nn alpha sA sx sy i' j') /\
        (forall (j':nat). j' < j ==>
           Seq.index sC (fidx n n i j') == fsyr2_at n nn alpha sA sx sy i j') /\
        cij == fsyr2_at n nn alpha sA sx sy i j)
      (ensures
        (let sC' = Seq.upd sC (fidx n n i j) cij in
         (forall (i':nat) (j':nat). i' < i /\ j' < n ==>
            Seq.index sC' (fidx n n i' j') == fsyr2_at n nn alpha sA sx sy i' j') /\
         (forall (j':nat). j' < j + 1 ==>
            Seq.index sC' (fidx n n i j') == fsyr2_at n nn alpha sA sx sy i j')))
  = let sC' = Seq.upd sC (fidx n n i j) cij in
    introduce forall (i':nat) (j':nat). i' < i /\ j' < n ==>
                Seq.index sC' (fidx n n i' j') == fsyr2_at n nn alpha sA sx sy i' j'
    with introduce _ ==> _
    with _. (ML.lemma_mult_le_right n (i' + 1) i;
             Seq.lemma_index_upd2 sC (fidx n n i j) cij (fidx n n i' j'));
    introduce forall (j':nat). j' < j + 1 ==>
                Seq.index sC' (fidx n n i j') == fsyr2_at n nn alpha sA sx sy i j'
    with introduce _ ==> _
    with _. (if j' < j
             then Seq.lemma_index_upd2 sC (fidx n n i j) cij (fidx n n i j')
             else Seq.lemma_index_upd1 sC (fidx n n i j) cij)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 100"
inline_for_extraction noextract
fn syr2_kf
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (alpha : et)
  (gA : array1 et (l1_forward (n *^ n)))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (gC : array1 et (l1_forward (n *^ n)))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy : erased (lseq et (SZ.v n)))
  (#sC0 : erased (lseq et (SZ.v (n *^ n))))
  (#fa #fx #fy : perm)
  ()
  norewrite
  requires
    gpu ** gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> Frac fy sy ** gC |-> sC0
  ensures
    gpu ** gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> Frac fy sy **
    gC |-> fsyr2 (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy
{
  let mut i : szle n = 0sz;

  while (!i <^ n)
    invariant live i
    invariant exists* (sC : lseq et (SZ.v (n *^ n))). gC |-> sC **
      pure (SZ.v !i <= SZ.v n /\
            (forall (i':nat) (j':nat). i' < SZ.v !i /\ j' < SZ.v n ==>
               Seq.index sC (fidx (SZ.v n) (SZ.v n) i' j')
               == fsyr2_at (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy i' j'))
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
                 == fsyr2_at (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy i' j') /\
              (forall (j':nat). j' < SZ.v !j ==>
                 Seq.index sC (fidx (SZ.v n) (SZ.v n) (SZ.v vi) j')
                 == fsyr2_at (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy (SZ.v vi) j'))
      decreases (SZ.v n - SZ.v !j)
    {
      let vj = !j;
      idx_bound (SZ.v n) (SZ.v vi) (SZ.v vj);
      let aij = Array1.(gA.((vi *^ n) +^ vj));
      let xi = Array1.(gx.(vi));
      let xj = Array1.(gx.(vj));
      let yi = Array1.(gy.(vi));
      let yj = Array1.(gy.(vj));
      let cij = aij `add` (alpha `mul` ((xi `mul` yj) `add` (yi `mul` xj)));
      with sCp. assert (gC |-> sCp);
      Array1.(gC.((vi *^ n) +^ vj) <- cij);
      syr2_upd_2d (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy sCp (SZ.v vi) (SZ.v vj) cij;
      j := !j +^ 1sz;
    };
    i := !i +^ 1sz;
  };

  with sCf. assert (gC |-> sCf);
  Seq.lemma_eq_intro sCf (fsyr2 (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy);
  ()
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper (single-thread kernel_desc_n, like Klas.Syr).            *)
(* ----------------------------------------------------------------------- *)

let fsyr2_approx_all
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (n:pos) (nn:nat{nn == n * n}) (alpha : et) (ralpha : real)
  (sA : lseq et nn) (sx sy : lseq et n) (rA : lseq real nn) (rx ry : lseq real n)
  : Lemma (requires sA %~ rA /\ sx %~ rx /\ sy %~ ry /\ alpha %~ ralpha)
          (ensures syr2_post n nn alpha ralpha sA sx sy rA rx ry)
  = introduce forall (i:nat) (j:nat). i < n /\ j < n ==>
                (Seq.index (fsyr2 n nn alpha sA sx sy) (fidx n n i j))
                %~ rsyr2_at n nn ralpha rA rx ry i j
    with introduce _ ==> _
    with _. (fsyr2_at_approx n nn alpha ralpha sA sx sy rA rx ry i j;
             fsyr2_index n nn alpha sA sx sy i j)

ghost
fn syr2_setup
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (gA : array1 et (l1_forward (n *^ n)))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (gC : array1 et (l1_forward (n *^ n)))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy : erased (lseq et (SZ.v n)))
  (#sC0 : erased (lseq et (SZ.v (n *^ n))))
  (#fa #fx #fy : perm)
  ()
  norewrite
  requires gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> Frac fy sy ** gC |-> sC0
  ensures (forall+ (tid : natlt 1sz).
             gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> Frac fy sy ** gC |-> sC0) ** emp
{
  forevery_singleton_intro #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> Frac fy sy ** gC |-> sC0);
}

ghost
fn syr2_teardown
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (alpha : et)
  (gA : array1 et (l1_forward (n *^ n)))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (gC : array1 et (l1_forward (n *^ n)))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy : erased (lseq et (SZ.v n)))
  (#fa #fx #fy : perm)
  ()
  norewrite
  requires
    (forall+ (tid : natlt 1sz).
       gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> Frac fy sy **
       gC |-> fsyr2 (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy) ** emp
  ensures
    gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> Frac fy sy **
    gC |-> fsyr2 (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> Frac fy sy **
       gC |-> fsyr2 (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy);
}

inline_for_extraction noextract
let kamsyr2
  (#et:Type0) {| floating et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (alpha : et)
  (gA : array1 et (l1_forward (n *^ n)))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (gC : array1 et (l1_forward (n *^ n)))
  (#_ : squash (Array1.is_global gA /\ Array1.is_global gx /\ Array1.is_global gy /\ Array1.is_global gC))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy : erased (lseq et (SZ.v n)))
  (#sC0 : erased (lseq et (SZ.v (n *^ n))))
  (#fa #fx #fy : perm)
  : kernel_desc
      (requires gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> Frac fy sy ** gC |-> sC0)
      (ensures  gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> Frac fy sy **
                gC |-> fsyr2 (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy)
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> syr2_kf alpha gA gx gy gC #sA #sx #sy #sC0 #fa #fx #fy);
    frame    = emp;
    teardown = syr2_teardown alpha gA gx gy gC;
    setup    = syr2_setup gA gx gy gC;
    kpre  = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> Frac fy sy ** gC |-> sC0);
    kpost = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> Frac fy sy **
                                     gC |-> fsyr2 (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn syr2_gen
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (alpha : et)
  (gA : array1 et (l1_forward (n *^ n)) { Array1.is_global gA })
  (gx : array1 et (l1_forward n) { Array1.is_global gx })
  (gy : array1 et (l1_forward n) { Array1.is_global gy })
  (gC : array1 et (l1_forward (n *^ n)) { Array1.is_global gC })
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy : erased (lseq et (SZ.v n)))
  (#sC0 : erased (lseq et (SZ.v (n *^ n))))
  (ralpha : erased real)
  (rA : erased (lseq real (SZ.v (n *^ n))))
  (rx : erased (lseq real (SZ.v n)))
  (ry : erased (lseq real (SZ.v n)))
  (#fa #fx #fy : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fa sA) **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gy |-> Frac fy sy)
  requires
    on gpu_loc (gC |-> sC0) **
    pure (sA %~ rA /\ sx %~ rx /\ sy %~ ry /\ alpha %~ ralpha)
  ensures
    on gpu_loc (gC |-> fsyr2 (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy) **
    pure (syr2_post (SZ.v n) (SZ.v (n *^ n)) alpha (Ghost.reveal ralpha) sA sx sy rA rx ry)
{
  on_star_eq gpu_loc (gy |-> Frac fy sy) (gC |-> sC0);
  rewrite (on gpu_loc (gy |-> Frac fy sy) ** on gpu_loc (gC |-> sC0))
       as (on gpu_loc ((gy |-> Frac fy sy) ** (gC |-> sC0)));
  on_star_eq gpu_loc (gx |-> Frac fx sx) ((gy |-> Frac fy sy) ** (gC |-> sC0));
  rewrite (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc ((gy |-> Frac fy sy) ** (gC |-> sC0)))
       as (on gpu_loc ((gx |-> Frac fx sx) ** ((gy |-> Frac fy sy) ** (gC |-> sC0))));
  on_star_eq gpu_loc (gA |-> Frac fa sA) ((gx |-> Frac fx sx) ** ((gy |-> Frac fy sy) ** (gC |-> sC0)));
  rewrite (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gx |-> Frac fx sx) ** ((gy |-> Frac fy sy) ** (gC |-> sC0))))
       as (on gpu_loc ((gA |-> Frac fa sA) ** ((gx |-> Frac fx sx) ** ((gy |-> Frac fy sy) ** (gC |-> sC0)))));

  launch_sync (kamsyr2 n alpha gA gx gy gC #() #sA #sx #sy #sC0 #fa #fx #fy);

  on_star_eq gpu_loc (gA |-> Frac fa sA)
             ((gx |-> Frac fx sx) ** ((gy |-> Frac fy sy) ** (gC |-> fsyr2 (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy)));
  rewrite (on gpu_loc ((gA |-> Frac fa sA) ** ((gx |-> Frac fx sx) ** ((gy |-> Frac fy sy) ** (gC |-> fsyr2 (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy)))))
       as (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gx |-> Frac fx sx) ** ((gy |-> Frac fy sy) ** (gC |-> fsyr2 (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy))));
  on_star_eq gpu_loc (gx |-> Frac fx sx) ((gy |-> Frac fy sy) ** (gC |-> fsyr2 (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy));
  rewrite (on gpu_loc ((gx |-> Frac fx sx) ** ((gy |-> Frac fy sy) ** (gC |-> fsyr2 (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy))))
       as (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc ((gy |-> Frac fy sy) ** (gC |-> fsyr2 (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy)));
  on_star_eq gpu_loc (gy |-> Frac fy sy) (gC |-> fsyr2 (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy);
  rewrite (on gpu_loc ((gy |-> Frac fy sy) ** (gC |-> fsyr2 (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy)))
       as (on gpu_loc (gy |-> Frac fy sy) ** on gpu_loc (gC |-> fsyr2 (SZ.v n) (SZ.v (n *^ n)) alpha sA sx sy));

  fsyr2_approx_all (SZ.v n) (SZ.v (n *^ n)) alpha (Ghost.reveal ralpha) sA sx sy rA rx ry;
  ()
}

let syr2_f32 = syr2_gen #f32
let syr2_f64 = syr2_gen #f64
