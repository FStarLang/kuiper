module Kuiper.EMatrix
#lang-pulse

(* An "erased" matrix, for specification purposes only *)

open Kuiper
open Kuiper.Container
open Kuiper.Approximates
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality

[@@erasable]
noeq
type ematrix (et:Type) (rows cols : nat) =
  | M : f:(natlt rows & natlt cols ^->> et)
     -> ematrix et rows cols

let mkM (#et:Type) (#rows #cols : nat)
  (f : natlt rows -> natlt cols -> GTot et)
  : ematrix et rows cols
  = M <| F.on_g _ <| fun (i, j) -> f i j

let const_matrix (#et:Type) (#rows #cols : nat)
  (v:et)
  : ematrix et rows cols
  = mkM fun _ _ -> v

let macc (#et:Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  (i : natlt rows)
  (j : natlt cols)
  : GTot et
  = m.f (i, j)

let mupd (#et:Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  (i : nat{ i < rows })
  (j : nat{ j < cols })
  (v : et)
  : ematrix et rows cols
  = mkM fun i' j' ->
      if i' = i && j' = j
      then v
      else m.f (i', j')

val macc_pat (#et :Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  (i : natlt rows)
  (j : natlt cols)
  : Lemma (macc m i j == m.f (i, j))
          [SMTPat (m.f (i, j))]

let matrix_comb (#et:Type) (#rows #cols : nat)
  (f : binop et)
  (m1 m2 : ematrix et rows cols)
  : ematrix et rows cols
  = mkM fun i j -> f (macc m1 i j) (macc m2 i j)

let mtranspose (#et:Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  : ematrix et cols rows
  = mkM fun i j -> m.f (j, i)

val equal (#et #rows #cols : _) (m1 m2 : ematrix et rows cols) : prop

val lemma_equal_intro (#et #rows #cols : _)
  (m1 m2 : ematrix et rows cols)
  : Lemma (requires forall (i:natlt rows) (j:natlt cols). macc m1 i j == macc m2 i j)
          (ensures equal m1 m2)
          [SMTPat (equal m1 m2)]

val ematrix_ext #et #rows #cols
  (m1 m2 : ematrix et rows cols)
  : Lemma (requires equal m1 m2)
          (ensures m1 == m2)
          [SMTPat (equal m1 m2)]

let ematrix_approximates #et
  {| scalar et, Kuiper.Approximates.real_like et |}
  #rows #cols
  (m1 : ematrix et rows cols)
  (m2 : ematrix real rows cols)
  : prop
  = forall (i:natlt rows) (j:natlt cols).
      macc m1 i j %~ macc m2 i j

instance ematrix_can_approximate
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat)
  : can_approximate (ematrix et rows cols) (ematrix real rows cols) =
{
  approximates = ematrix_approximates;
}

let to_real_matrix (#et : Type0)
  {| scalar et, real_like et |}
  (#rows #cols : nat)
  (m : ematrix et rows cols)
  : GTot (ematrix real rows cols)
  = mkM fun i j -> to_real (macc m i j)

val lemma_to_real_matrix_approximates (#et : Type0)
  {| scalar et, d : real_like et |}
  (#rows #cols : nat)
  (m : ematrix et rows cols)
  : Lemma (ensures m %~ to_real_matrix m)
          [SMTPat (to_real_matrix m)]

instance ematrix_is_container
  (et:Type) (#rows #cols : nat)
  : container (ematrix et rows cols) (natlt rows & natlt cols) et
= {
    acc = (fun m (r,c) -> macc m r c);
    upd = (fun m (i, j) x -> mupd m i j x);
    l1 = ez;
    l2 = ez;
    ext = (fun c1 c2 _ -> assert (equal c1 c2));
    from_fun = (fun f -> mkM fun i j -> f (i, j));
    from_fun_ok = ez;
  }

let ematrix_row
  (#et : Type0)
  (#rows #cols : erased nat)
  (em : ematrix et rows cols)
  (i : natlt rows)
  : GTot (lseq et cols)
  = Seq.init_ghost cols (fun j -> macc em i j)

let ematrix_col
  (#et : Type0)
  (#rows #cols : erased nat)
  (em : ematrix et rows cols)
  (j : natlt cols)
  : GTot (lseq et rows)
  = Seq.init_ghost rows (fun i -> macc em i j)

let ematrix_upd_row
  (#et : Type0)
  (#rows #cols : erased nat)
  (em : ematrix et rows cols)
  (i : natlt rows)
  (new_row : lseq et cols)
  : ematrix et rows cols
  = mkM fun i' j ->
      if i' = i
      then Seq.index new_row j
      else macc em i' j

let ematrix_upd_col
  (#et : Type0)
  (#rows #cols : erased nat)
  (em : ematrix et rows cols)
  (j : natlt cols)
  (new_col : lseq et rows)
  : ematrix et rows cols
  = mkM fun i j' ->
      if j' = j
      then Seq.index new_col i
      else macc em i j'
