module Klas.Trttp

(* cuBLAS trttp (lower) : full -> triangular packed conversion

       AP := pack(A_full),

   copying the lower triangle of a full n x n row-major matrix into PACKED
   row-major form (entry (i,j), j<=i, stored at off(i)+j; AP length
   np = n*(n+1)/2):

       AP[off(i)+j] = A_full[i*n+j]   for j<=i.

   This is the inverse of Klas.Tpttr. The output is packed, so its clean
   Seq.init spec needs the inverse packed->(i,j) map (prow / pcol), defined here
   and reusable by other "write packed" ops. The kernel iterates the lower
   triangle (forward), carrying a running row offset; the invariant is stated
   over the packed index p (always < np) via the inverse. Reuses off / off_mono
   from Klas.Tpmv and the flat index helpers from Klas.Trsm / Klas.Trsv. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Klas.Tpmv { off, off_mono, poff_bound }
open Klas.Trsv { idx_bound }
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* ----------------------------------------------------------------------- *)
(* Inverse packed index: prow p / pcol p with off(prow p)+pcol p == p.       *)
(* ----------------------------------------------------------------------- *)

noextract
let rec off_ge (i:nat) : Lemma (ensures off i >= i) (decreases i)
  = if i = 0 then () else off_ge (i - 1)

noextract
let rec scan_row (p:nat) (i:nat{off i <= p})
  : Tot (r:nat{off r <= p /\ p < off (r + 1)}) (decreases (p - i))
  = off_ge (i + 1);
    if off (i + 1) <= p then scan_row p (i + 1) else i

noextract
let prow (p:nat) : (r:nat{off r <= p /\ p < off (r + 1)}) = scan_row p 0
noextract
let pcol (p:nat) : nat = p - off (prow p)

(* p decomposes as off(prow p) + pcol p, with pcol p <= prow p. *)
let pcol_le (p:nat) : Lemma (pcol p <= prow p /\ off (prow p) + pcol p == p) = ()

(* off i + j with j<=i inverts to (i,j). *)
let prow_off (i:nat) (j:nat{j <= i}) : Lemma (prow (off i + j) == i /\ pcol (off i + j) == j)
  = let p = off i + j in
    let r = prow p in
    if r < i then off_mono (r + 1) i
    else if r > i then off_mono (i + 1) r
    else ()

let pidx_bound (n:nat) (p:nat{p < off n}) : Lemma (prow p < n /\ pcol p <= prow p)
  = pcol_le p;
    if prow p >= n then off_mono n (prow p)

(* ----------------------------------------------------------------------- *)
(* Spec: the packed cell (via the inverse) and the whole packed array.       *)
(* ----------------------------------------------------------------------- *)

let ftrttp_at (#et:Type0) {| floating et |}
  (n:pos) (nn:nat{nn == n * n}) (sX : lseq et nn) (p:nat{p < off n}) : et
  = pidx_bound n p;
    idx_bound n (prow p) (pcol p);
    Seq.index sX (prow p * n + pcol p)

noextract
let ftrttp (#et:Type0) {| floating et |}
  (n:pos) (np:nat{np == off n}) (nn:nat{nn == n * n}) (sX : lseq et nn) : lseq et np
  = Seq.init np (fun (p:nat{p < np}) -> ftrttp_at n nn sX p)

let ftrttp_ext (#et:Type0) {| floating et |}
  (n:pos) (np:nat{np == off n}) (nn:nat{nn == n * n}) (sX : lseq et nn) (sAP : lseq et np)
  : Lemma (requires (forall (p:nat). p < np /\ prow p < n ==> Seq.index sAP p == ftrttp_at n nn sX p))
          (ensures sAP == ftrttp n np nn sX)
  = introduce forall (p:nat). p < np ==> Seq.index sAP p == ftrttp_at n nn sX p
    with introduce _ ==> _
    with _. pidx_bound n p;
    Seq.lemma_eq_intro sAP (ftrttp n np nn sX)

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: write the lower triangle into packed form.          *)
(* ----------------------------------------------------------------------- *)

(* Writing packed position pij = off(i)+j (the unique p with (prow,pcol)=(i,j))
   extends the "done" set from (rows<i, plus row i cols<j) to cols<j+1. Stated
   over the packed index p, so every index is < np. *)
#push-options "--fuel 2 --ifuel 2 --z3rlimit 200"
let trttp_upd (#et:Type0) {| floating et |}
  (n:pos) (np:nat{np == off n}) (nn:nat{nn == n * n}) (sX : lseq et nn) (sAP : lseq et np)
  (i:nat{i < n}) (j:nat{j <= i}) (pij:nat{pij < np /\ prow pij == i /\ pcol pij == j}) (v : et)
  : Lemma
      (requires v == ftrttp_at n nn sX pij /\
        (forall (p:nat). p < np /\ (prow p < i \/ (prow p == i /\ pcol p < j)) ==>
           Seq.index sAP p == ftrttp_at n nn sX p))
      (ensures
        (let sAP' = Seq.upd sAP pij v in
         (forall (p:nat). p < np /\ (prow p < i \/ (prow p == i /\ pcol p < j + 1)) ==>
            Seq.index sAP' p == ftrttp_at n nn sX p)))
  = let sAP' = Seq.upd sAP pij v in
    introduce forall (p:nat). p < np /\ (prow p < i \/ (prow p == i /\ pcol p < j + 1)) ==>
                Seq.index sAP' p == ftrttp_at n nn sX p
    with introduce _ ==> _
    with _. (if p = pij then Seq.lemma_index_upd1 sAP pij v
             else (Seq.lemma_index_upd2 sAP pij v p;
                   pcol_le p; pcol_le pij))
#pop-options

#push-options "--fuel 2 --ifuel 1 --z3rlimit 200"
inline_for_extraction noextract
fn trttp_kf
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#np : szp { SZ.v np == off (SZ.v n) })
  (gX : array1 et (l1_forward (n *^ n)))
  (gAP : array1 et (l1_forward np))
  (#sX : erased (lseq et (SZ.v (n *^ n))))
  (#sAP0 : erased (lseq et (SZ.v np)))
  (#fx : perm)
  ()
  norewrite
  requires
    gpu ** gX |-> Frac fx sX ** gAP |-> sAP0
  ensures
    gpu ** gX |-> Frac fx sX **
    gAP |-> ftrttp (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sX
{
  let mut i : szle n = 0sz;
  let mut o : sz = 0sz;

  while (!i <^ n)
    invariant live i
    invariant live o
    invariant exists* (sAP : lseq et (SZ.v np)). gAP |-> sAP **
      pure (SZ.v !i <= SZ.v n /\ SZ.v !o == off (SZ.v !i) /\
            (forall (p:nat). p < SZ.v np /\ prow p < SZ.v !i ==>
               Seq.index sAP p == ftrttp_at (SZ.v n) (SZ.v (n *^ n)) sX p))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let vo = !o;
    let mut j : sz = 0sz;

    while (!j <=^ vi)
      invariant live j
      invariant exists* (sAP : lseq et (SZ.v np)). gAP |-> sAP **
        pure (SZ.v !j <= SZ.v vi + 1 /\
              (forall (p:nat). p < SZ.v np /\
                 (prow p < SZ.v vi \/ (prow p == SZ.v vi /\ pcol p < SZ.v !j)) ==>
                 Seq.index sAP p == ftrttp_at (SZ.v n) (SZ.v (n *^ n)) sX p))
      decreases (SZ.v vi + 1 - SZ.v !j)
    {
      let vj = !j;
      idx_bound (SZ.v n) (SZ.v vi) (SZ.v vj);
      poff_bound (SZ.v n) (SZ.v vi) (SZ.v vj);
      prow_off (SZ.v vi) (SZ.v vj);
      let v = Array1.(gX.((vi *^ n) +^ vj));
      with sappre. assert (gAP |-> sappre);
      Array1.(gAP.(vo +^ vj) <- v);
      trttp_upd (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sX sappre (SZ.v vi) (SZ.v vj) (SZ.v (vo +^ vj)) v;
      j := !j +^ 1sz;
    };
    off_mono (SZ.v vi + 1) (SZ.v n);
    o := !o +^ (vi +^ 1sz);
    i := !i +^ 1sz;
  };

  with sapf. assert (gAP |-> sapf);
  ftrttp_ext (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sX sapf;
  ()
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper (single-thread kernel_desc_n).                           *)
(* ----------------------------------------------------------------------- *)

ghost
fn trttp_setup
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#np : szp { SZ.v np == off (SZ.v n) })
  (gX : array1 et (l1_forward (n *^ n)))
  (gAP : array1 et (l1_forward np))
  (#sX : erased (lseq et (SZ.v (n *^ n))))
  (#sAP0 : erased (lseq et (SZ.v np)))
  (#fx : perm)
  ()
  norewrite
  requires gX |-> Frac fx sX ** gAP |-> sAP0
  ensures (forall+ (tid : natlt 1sz). gX |-> Frac fx sX ** gAP |-> sAP0) ** emp
{
  forevery_singleton_intro #(natlt 1sz)
    (fun (_:natlt 1sz) -> gX |-> Frac fx sX ** gAP |-> sAP0);
}

ghost
fn trttp_teardown
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (#np : szp { SZ.v np == off (SZ.v n) })
  (gX : array1 et (l1_forward (n *^ n)))
  (gAP : array1 et (l1_forward np))
  (#sX : erased (lseq et (SZ.v (n *^ n))))
  (#fx : perm)
  ()
  norewrite
  requires
    (forall+ (tid : natlt 1sz).
       gX |-> Frac fx sX ** gAP |-> ftrttp (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sX) ** emp
  ensures
    gX |-> Frac fx sX ** gAP |-> ftrttp (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sX
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> gX |-> Frac fx sX **
       gAP |-> ftrttp (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sX);
}

inline_for_extraction noextract
let kamtrttp
  (#et:Type0) {| floating et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (np : szp { SZ.v np == off (SZ.v n) })
  (gX : array1 et (l1_forward (n *^ n)))
  (gAP : array1 et (l1_forward np))
  (#_ : squash (Array1.is_global gX /\ Array1.is_global gAP))
  (#sX : erased (lseq et (SZ.v (n *^ n))))
  (#sAP0 : erased (lseq et (SZ.v np)))
  (#fx : perm)
  : kernel_desc
      (requires gX |-> Frac fx sX ** gAP |-> sAP0)
      (ensures  gX |-> Frac fx sX ** gAP |-> ftrttp (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sX)
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> trttp_kf gX gAP #sX #sAP0 #fx);
    frame    = emp;
    teardown = trttp_teardown gX gAP;
    setup    = trttp_setup gX gAP;
    kpre  = (fun (_i : natlt 1sz) -> gX |-> Frac fx sX ** gAP |-> sAP0);
    kpost = (fun (_i : natlt 1sz) -> gX |-> Frac fx sX ** gAP |-> ftrttp (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sX);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn trttp_gen
  (#et:Type0) {| floating et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (np : szp { SZ.v np == off (SZ.v n) })
  (gX : array1 et (l1_forward (n *^ n)) { Array1.is_global gX })
  (gAP : array1 et (l1_forward np) { Array1.is_global gAP })
  (#sX : erased (lseq et (SZ.v (n *^ n))))
  (#sAP0 : erased (lseq et (SZ.v np)))
  (#fx : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gX |-> Frac fx sX)
  requires
    on gpu_loc (gAP |-> sAP0)
  ensures
    on gpu_loc (gAP |-> ftrttp (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sX)
{
  on_star_eq gpu_loc (gX |-> Frac fx sX) (gAP |-> sAP0);
  rewrite (on gpu_loc (gX |-> Frac fx sX) ** on gpu_loc (gAP |-> sAP0))
       as (on gpu_loc ((gX |-> Frac fx sX) ** (gAP |-> sAP0)));

  launch_sync (kamtrttp n np gX gAP #() #sX #sAP0 #fx);

  on_star_eq gpu_loc (gX |-> Frac fx sX) (gAP |-> ftrttp (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sX);
  rewrite (on gpu_loc ((gX |-> Frac fx sX) ** (gAP |-> ftrttp (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sX)))
       as (on gpu_loc (gX |-> Frac fx sX) ** on gpu_loc (gAP |-> ftrttp (SZ.v n) (SZ.v np) (SZ.v (n *^ n)) sX));
  ()
}

let trttp_f32 = trttp_gen #f32
let trttp_f64 = trttp_gen #f64
