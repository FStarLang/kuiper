module Klas.Tpttr

(* cuBLAS tpttr (lower) : triangular packed -> full conversion

       A_full := unpack(AP),

   copying the lower triangle stored in PACKED row-major form (entry (i,j),
   j<=i, at offset off(i)+j; AP length np = n*(n+1)/2) into a full n x n
   row-major matrix, with the strict upper triangle set to zero:

       A_full[i*n+j] = (j<=i) ? AP[off(i)+j] : 0.

   Reads packed (forward index, no inverse needed); the full output is indexed
   directly by (i,j). Reuses the recursive offset off / poff_bound from
   Klas.Tpmv and the flat index helpers fidx / div_lt from Klas.Trsm. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Klas.Tpmv { off, off_mono, poff_bound }
open Klas.Trsm { fidx, col_bound, div_lt }
open Klas.Trsv { idx_bound }
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* ----------------------------------------------------------------------- *)
(* Spec: the unpacked cell and the whole full matrix.                        *)
(* ----------------------------------------------------------------------- *)

(* Guarded read bound: only the lower triangle reads the packed array. *)
let tpttr_bound (n i j : nat)
  : Lemma (requires i < n /\ j < n) (ensures j <= i ==> off i + j < off n)
  = if j <= i then poff_bound n i j

let ftpttr_at (#et:Type0) {| floating et |}
  (n:pos) (np:nat{np == off n}) (sA : lseq et np) (i:nat{i < n}) (j:nat{j < n}) : et
  = if j <= i then (poff_bound n i j; Seq.index sA (off i + j)) else zero

noextract
let ftpttr (#et:Type0) {| floating et |}
  (n:pos) (np:nat{np == off n}) (nn:nat{nn == n * n}) (sA : lseq et np) : lseq et nn
  = Seq.init nn (fun (idx:nat{idx < nn}) ->
      div_lt idx n n; ftpttr_at n np sA (idx / n) (idx % n))

let ftpttr_index (#et:Type0) {| floating et |}
  (n:pos) (np:nat{np == off n}) (nn:nat{nn == n * n}) (sA : lseq et np)
  (i:nat{i < n}) (j:nat{j < n})
  : Lemma (ensures Seq.index (ftpttr n np nn sA) (fidx n n i j) == ftpttr_at n np sA i j)
          [SMTPat (Seq.index (ftpttr n np nn sA) (fidx n n i j))]
  = ML.lemma_div_plus j i n; ML.small_div j n; ML.lemma_mod_plus j i n; ML.small_mod j n

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: one cell per (i,j), row offset oi carried.          *)
(* ----------------------------------------------------------------------- *)

let tpttr_upd (#et:Type0) {| floating et |}
  (n:pos) (np:nat{np == off n}) (nn:nat{nn == n * n}) (sA : lseq et np)
  (sX : lseq et nn) (i:nat{i < n}) (j:nat{j < n}) (xij : et)
  : Lemma
      (requires
        (forall (i':nat) (j':nat). i' < i /\ j' < n ==>
           Seq.index sX (fidx n n i' j') == ftpttr_at n np sA i' j') /\
        (forall (j':nat). j' < j ==>
           Seq.index sX (fidx n n i j') == ftpttr_at n np sA i j') /\
        xij == ftpttr_at n np sA i j)
      (ensures
        (let sX' = Seq.upd sX (fidx n n i j) xij in
         (forall (i':nat) (j':nat). i' < i /\ j' < n ==>
            Seq.index sX' (fidx n n i' j') == ftpttr_at n np sA i' j') /\
         (forall (j':nat). j' < j + 1 ==>
            Seq.index sX' (fidx n n i j') == ftpttr_at n np sA i j')))
  = let sX' = Seq.upd sX (fidx n n i j) xij in
    introduce forall (i':nat) (j':nat). i' < i /\ j' < n ==>
                Seq.index sX' (fidx n n i' j') == ftpttr_at n np sA i' j'
    with introduce _ ==> _
    with _. (ML.lemma_mult_le_right n (i' + 1) i;
             Seq.lemma_index_upd2 sX (fidx n n i j) xij (fidx n n i' j'));
    introduce forall (j':nat). j' < j + 1 ==>
                Seq.index sX' (fidx n n i j') == ftpttr_at n np sA i j'
    with introduce _ ==> _
    with _. (if j' < j
             then Seq.lemma_index_upd2 sX (fidx n n i j) xij (fidx n n i j')
             else Seq.lemma_index_upd1 sX (fidx n n i j) xij)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 120"
inline_for_extraction noextract
fn tpttr_kf
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#np : szp { SZ.v np == off (SZ.v n) })
  (gA : array1 et (l1_forward np))
  (gX : array1 et (l1_forward (n *^ n)))
  (#sA : erased (lseq et (SZ.v np)))
  (#sX0 : erased (lseq et (SZ.v (n *^ n))))
  (#fa : perm)
  ()
  norewrite
  requires
    gpu ** gA |-> Frac fa sA ** gX |-> sX0
  ensures
    gpu ** gA |-> Frac fa sA **
    gX |-> ftpttr (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sA
{
  let mut i : szle n = 0sz;
  let mut o : sz = 0sz;

  while (!i <^ n)
    invariant live i
    invariant live o
    invariant exists* (sX : lseq et (SZ.v (n *^ n))). gX |-> sX **
      pure (SZ.v !i <= SZ.v n /\ SZ.v !o == off (SZ.v !i) /\
            (forall (i':nat) (j':nat). i' < SZ.v !i /\ j' < SZ.v n ==>
               Seq.index sX (fidx (SZ.v n) (SZ.v n) i' j')
               == ftpttr_at (SZ.v n) (SZ.v np) sA i' j'))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let vo = !o;
    let mut j : szle n = 0sz;

    while (!j <^ n)
      invariant live j
      invariant exists* (sX : lseq et (SZ.v (n *^ n))). gX |-> sX **
        pure (SZ.v !j <= SZ.v n /\
              (forall (i':nat) (j':nat). i' < SZ.v vi /\ j' < SZ.v n ==>
                 Seq.index sX (fidx (SZ.v n) (SZ.v n) i' j')
                 == ftpttr_at (SZ.v n) (SZ.v np) sA i' j') /\
              (forall (j':nat). j' < SZ.v !j ==>
                 Seq.index sX (fidx (SZ.v n) (SZ.v n) (SZ.v vi) j')
                 == ftpttr_at (SZ.v n) (SZ.v np) sA (SZ.v vi) j'))
      decreases (SZ.v n - SZ.v !j)
    {
      let vj = !j;
      idx_bound (SZ.v n) (SZ.v vi) (SZ.v vj);
      tpttr_bound (SZ.v n) (SZ.v vi) (SZ.v vj);
      let z : et = zero;
      let xij =
        if (vj <=^ vi) {
          Array1.(gA.(vo +^ vj))
        } else {
          z
        };
      with sxpre. assert (gX |-> sxpre);
      Array1.(gX.((vi *^ n) +^ vj) <- xij);
      tpttr_upd (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sA sxpre (SZ.v vi) (SZ.v vj) xij;
      j := !j +^ 1sz;
    };
    off_mono (SZ.v vi + 1) (SZ.v n);
    o := !o +^ (vi +^ 1sz);
    i := !i +^ 1sz;
  };

  with sXf. assert (gX |-> sXf);
  Seq.lemma_eq_intro sXf (ftpttr (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sA);
  ()
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper (single-thread kernel_desc_n).                           *)
(* ----------------------------------------------------------------------- *)

ghost
fn tpttr_setup
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#np : szp { SZ.v np == off (SZ.v n) })
  (gA : array1 et (l1_forward np))
  (gX : array1 et (l1_forward (n *^ n)))
  (#sA : erased (lseq et (SZ.v np)))
  (#sX0 : erased (lseq et (SZ.v (n *^ n))))
  (#fa : perm)
  ()
  norewrite
  requires gA |-> Frac fa sA ** gX |-> sX0
  ensures (forall+ (tid : natlt 1sz). gA |-> Frac fa sA ** gX |-> sX0) ** emp
{
  forevery_singleton_intro #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gX |-> sX0);
}

ghost
fn tpttr_teardown
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#np : szp { SZ.v np == off (SZ.v n) })
  (gA : array1 et (l1_forward np))
  (gX : array1 et (l1_forward (n *^ n)))
  (#sA : erased (lseq et (SZ.v np)))
  (#fa : perm)
  ()
  norewrite
  requires
    (forall+ (tid : natlt 1sz).
       gA |-> Frac fa sA ** gX |-> ftpttr (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sA) ** emp
  ensures
    gA |-> Frac fa sA ** gX |-> ftpttr (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sA
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA **
       gX |-> ftpttr (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sA);
}

inline_for_extraction noextract
let kamtpttr
  (#et:Type0) {| floating et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (np : szp { SZ.v np == off (SZ.v n) })
  (gA : array1 et (l1_forward np))
  (gX : array1 et (l1_forward (n *^ n)))
  (#_ : squash (Array1.is_global gA /\ Array1.is_global gX))
  (#sA : erased (lseq et (SZ.v np)))
  (#sX0 : erased (lseq et (SZ.v (n *^ n))))
  (#fa : perm)
  : kernel_desc
      (requires gA |-> Frac fa sA ** gX |-> sX0)
      (ensures  gA |-> Frac fa sA ** gX |-> ftpttr (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sA)
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> tpttr_kf gA gX #sA #sX0 #fa);
    frame    = emp;
    teardown = tpttr_teardown gA gX;
    setup    = tpttr_setup gA gX;
    kpre  = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gX |-> sX0);
    kpost = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gX |-> ftpttr (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sA);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn tpttr_gen
  (#et:Type0) {| floating et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (np : szp { SZ.v np == off (SZ.v n) })
  (gA : array1 et (l1_forward np) { Array1.is_global gA })
  (gX : array1 et (l1_forward (n *^ n)) { Array1.is_global gX })
  (#sA : erased (lseq et (SZ.v np)))
  (#sX0 : erased (lseq et (SZ.v (n *^ n))))
  (#fa : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fa sA)
  requires
    on gpu_loc (gX |-> sX0)
  ensures
    on gpu_loc (gX |-> ftpttr (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sA)
{
  on_star_eq gpu_loc (gA |-> Frac fa sA) (gX |-> sX0);
  rewrite (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc (gX |-> sX0))
       as (on gpu_loc ((gA |-> Frac fa sA) ** (gX |-> sX0)));

  launch_sync (kamtpttr n np gA gX #() #sA #sX0 #fa);

  on_star_eq gpu_loc (gA |-> Frac fa sA) (gX |-> ftpttr (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sA);
  rewrite (on gpu_loc ((gA |-> Frac fa sA) ** (gX |-> ftpttr (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sA)))
       as (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc (gX |-> ftpttr (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sA));
  ()
}

let tpttr_f32 = tpttr_gen #f32
let tpttr_f64 = tpttr_gen #f64
