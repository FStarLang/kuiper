module Kuiper.EMatrix4
#lang-pulse

open Kuiper
open Kuiper.Container
open Kuiper.Approximates
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality

[@@erasable]
noeq
type ematrix4 (et:Type) (d0 d1 d2 d3 : nat) =
  | M : f:(natlt d0 & natlt d1 & natlt d2 & natlt d3 ^->> et)
     -> ematrix4 et d0 d1 d2 d3

unfold let t = ematrix4

let mkM (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (f : natlt d0 -> natlt d1 -> natlt d2 -> natlt d3 -> GTot et)
  : ematrix4 et d0 d1 d2 d3
  = M <| F.on_g _ <| fun (i, j, k, l) -> f i j k l

let const_matrix (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (v:et)
  : ematrix4 et d0 d1 d2 d3
  = mkM fun _ _ _ _ -> v

let macc (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (m : ematrix4 et d0 d1 d2 d3)
  (i : natlt d0)
  (j : natlt d1)
  (k : natlt d2)
  (l : natlt d3)
  : GTot et
  = m.f (i, j, k, l)

let mupd (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (m : ematrix4 et d0 d1 d2 d3)
  (i : natlt d0)
  (j : natlt d1)
  (k : natlt d2)
  (l : natlt d3)
  (v : et)
  : ematrix4 et d0 d1 d2 d3
  = mkM fun i' j' k' l' ->
      if i' = i && j' = j && k' = k && l' = l
      then v
      else m.f (i', j', k', l')

val macc_pat (#et :Type) (#d0 #d1 #d2 #d3 : nat)
  (m : ematrix4 et d0 d1 d2 d3)
  (i : natlt d0)
  (j : natlt d1)
  (k : natlt d2)
  (l : natlt d3)
  : Lemma (macc m i j k l == m.f (i, j, k, l))
          [SMTPat (m.f (i, j, k, l))]

let matrix_comb (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (f : binop et)
  (m1 m2 : ematrix4 et d0 d1 d2 d3)
  : ematrix4 et d0 d1 d2 d3
  = mkM fun i j k l -> f (macc m1 i j k l) (macc m2 i j k l)

val equal (#et #d0 #d1 #d2 #d3 : _) (m1 m2 : ematrix4 et d0 d1 d2 d3) : prop

val lemma_equal_intro (#et #d0 #d1 #d2 #d3 : _)
  (m1 m2 : ematrix4 et d0 d1 d2 d3)
  : Lemma (requires forall (i:natlt d0) (j:natlt d1) (k:natlt d2) (l:natlt d3). macc m1 i j k l == macc m2 i j k l)
          (ensures equal m1 m2)
          [SMTPat (equal m1 m2)]

val ext #et #d0 #d1 #d2 #d3
  (m1 m2 : ematrix4 et d0 d1 d2 d3)
  : Lemma (requires equal m1 m2)
          (ensures m1 == m2)
          [SMTPat (equal m1 m2)]

let ematrix_approximates #et
  {| scalar et, Kuiper.Approximates.real_like et |}
  #d0 #d1 #d2 #d3
  (m1 : ematrix4 et d0 d1 d2 d3)
  (m2 : ematrix4 real d0 d1 d2 d3)
  : prop
  = forall (i:natlt d0) (j:natlt d1) (k:natlt d2) (l:natlt d3).
      macc m1 i j k l %~ macc m2 i j k l

instance ematrix_can_approximate
  (#et : Type0) {| scalar et, real_like et |}
  (#d0 #d1 #d2 #d3 : nat)
  : can_approximate (ematrix4 et d0 d1 d2 d3) (ematrix4 real d0 d1 d2 d3) =
{
  approximates = ematrix_approximates;
}

let to_real_matrix (#et : Type0)
  {| scalar et, real_like et |}
  (#d0 #d1 #d2 #d3 : nat)
  (m : ematrix4 et d0 d1 d2 d3)
  : GTot (ematrix4 real d0 d1 d2 d3)
  = mkM fun i j k l -> to_real (macc m i j k l)

val lemma_to_real_matrix_approximates (#et : Type0)
  {| scalar et, d : real_like et |}
  (#d0 #d1 #d2 #d3 : nat)
  (m : ematrix4 et d0 d1 d2 d3)
  : Lemma (ensures m %~ to_real_matrix m)
          [SMTPat (to_real_matrix m)]

instance ematrix_is_container
  (et:Type) (#d0 #d1 #d2 #d3 : nat)
  : container (ematrix4 et d0 d1 d2 d3) (natlt d0 & natlt d1 & natlt d2 & natlt d3) et
= {
    acc = (fun m (i, j, k, l) -> macc m i j k l);
    upd = (fun m (i, j, k, l) x -> mupd m i j k l x);
    l1 = ez;
    l2 = ez;
    ext = (fun c1 c2 _ -> assert (equal c1 c2));
    from_fun = (fun f -> mkM fun i j k l -> f (i, j, k, l));
    from_fun_ok = ez;
  }
