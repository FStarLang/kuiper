module Kuiper.EMatrix3
#lang-pulse

open Kuiper
open Kuiper.Container
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality

[@@erasable]
noeq
type ematrix3 (et:Type) (d0 d1 d2 : nat) =
  | M : f:(natlt d0 & natlt d1 & natlt d2 ^->> et)
     -> ematrix3 et d0 d1 d2

unfold let t = ematrix3

let mkM (#et:Type) (#d0 #d1 #d2 : nat)
  (f : natlt d0 -> natlt d1 -> natlt d2 -> GTot et)
  : ematrix3 et d0 d1 d2
  = M <| F.on_g _ <| fun (i, j, k) -> f i j k

let const_matrix (#et:Type) (#d0 #d1 #d2 : nat)
  (v:et)
  : ematrix3 et d0 d1 d2
  = mkM fun _ _ _ -> v

let macc (#et:Type) (#d0 #d1 #d2 : nat)
  (m : ematrix3 et d0 d1 d2)
  (i : natlt d0)
  (j : natlt d1)
  (k : natlt d2)
  : GTot et
  = m.f (i, j, k)

let mupd (#et:Type) (#d0 #d1 #d2 : nat)
  (m : ematrix3 et d0 d1 d2)
  (i : natlt d0)
  (j : natlt d1)
  (k : natlt d2)
  (v : et)
  : ematrix3 et d0 d1 d2
  = mkM fun i' j' k' ->
      if i' = i && j' = j && k' = k
      then v
      else m.f (i', j', k')

val macc_pat (#et :Type) (#d0 #d1 #d2 : nat)
  (m : ematrix3 et d0 d1 d2)
  (i : natlt d0)
  (j : natlt d1)
  (k : natlt d2)
  : Lemma (macc m i j k == m.f (i, j, k))
          [SMTPat (m.f (i, j, k))]

let matrix_comb (#et:Type) (#d0 #d1 #d2 : nat)
  (f : binop et)
  (m1 m2 : ematrix3 et d0 d1 d2)
  : ematrix3 et d0 d1 d2
  = mkM fun i j k -> f (macc m1 i j k) (macc m2 i j k)

val equal (#et #d0 #d1 #d2 : _) (m1 m2 : ematrix3 et d0 d1 d2) : prop

val lemma_equal_intro (#et #d0 #d1 #d2 : _)
  (m1 m2 : ematrix3 et d0 d1 d2)
  : Lemma (requires forall (i:natlt d0) (j:natlt d1) (k:natlt d2). macc m1 i j k == macc m2 i j k)
          (ensures equal m1 m2)
          [SMTPat (equal m1 m2)]

val ext #et #d0 #d1 #d2
  (m1 m2 : ematrix3 et d0 d1 d2)
  : Lemma (requires equal m1 m2)
          (ensures m1 == m2)
          [SMTPat (equal m1 m2)]

let ematrix_approximates #et
  {| scalar et, Kuiper.Approximates.real_like et |}
  #d0 #d1 #d2
  (m1 : ematrix3 et d0 d1 d2)
  (m2 : ematrix3 real d0 d1 d2)
  : prop
  = forall (i:natlt d0) (j:natlt d1) (k:natlt d2).
      macc m1 i j k %~ macc m2 i j k

instance ematrix_can_approximate
  (#et : Type0) {| scalar et, real_like et |}
  (#d0 #d1 #d2 : nat)
  : can_approximate (ematrix3 et d0 d1 d2) (ematrix3 real d0 d1 d2) =
{
  approximates = ematrix_approximates;
}

let to_real_matrix (#et : Type0)
  {| scalar et, real_like et |}
  (#d0 #d1 #d2 : nat)
  (m : ematrix3 et d0 d1 d2)
  : GTot (ematrix3 real d0 d1 d2)
  = mkM fun i j k -> to_real (macc m i j k)

val lemma_to_real_matrix_approximates (#et : Type0)
  {| scalar et, d : real_like et |}
  (#d0 #d1 #d2 : nat)
  (m : ematrix3 et d0 d1 d2)
  : Lemma (ensures m %~ to_real_matrix m)
          [SMTPat (to_real_matrix m)]

instance ematrix_is_container
  (et:Type) (#d0 #d1 #d2 : nat)
  : container (ematrix3 et d0 d1 d2) (natlt d0 & natlt d1 & natlt d2) et
= {
    acc = (fun m (r,c,d) -> macc m r c d);
    upd = (fun m (i, j, k) x -> mupd m i j k x);
    l1 = ez;
    l2 = ez;
    ext = (fun c1 c2 _ -> assert (equal c1 c2));
    from_fun = (fun f -> mkM fun i j k -> f (i, j, k));
    from_fun_ok = ez;
  }
