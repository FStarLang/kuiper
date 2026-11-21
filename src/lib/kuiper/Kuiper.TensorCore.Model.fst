module Kuiper.TensorCore.Model

friend Kuiper.Float16.Base
friend Kuiper.BFloat16.Base
friend Kuiper.Float32.Base
friend Kuiper.Float64.Base
friend Kuiper.Float16
friend Kuiper.BFloat16
friend Kuiper.Float32
friend Kuiper.Float64

open Kuiper
open Kuiper.Chest
open Kuiper.Spec.GEMM
open Kuiper.Float.Model
module M = Kuiper.Float.Model
module R = Kuiper.TensorCore.Raw

private let ideals #rank #dims #raw (c:chest #rank dims (M.value raw)) =
  chest_map (fun x -> x.ideal) c

private let decorated_mma
  (#raw0 #raw1:Type0)
  (#rows #shared #columns:nat)
  (mc:chest2 (M.value raw1) rows columns)
  (ma:chest2 (M.value raw0) rows shared)
  (mb:chest2 (M.value raw0) shared columns)
  : chest2 (M.value raw1) rows columns
  =
  let repr = R.emma
    (chest_map (fun x -> x.repr) mc)
    (chest_map (fun x -> x.repr) ma)
    (chest_map (fun x -> x.repr) mb) in
  let ideal = matplus (ideals mc) (matmul (ideals ma) (ideals mb)) in
  chest_comb (fun repr ideal -> { repr; ideal }) repr ideal

let emma
  (#et0 #et1:Type0)
  (#rows #shared #columns:nat)
  (mc:chest2 et1 rows columns)
  (ma:chest2 et0 rows shared)
  (mb:chest2 et0 shared columns)
  : chest2 et1 rows columns
  =
  if t2b (et0 == half /\ et1 == half) then
    decorated_mma #Kuiper.Float16.Raw.t #Kuiper.Float16.Raw.t mc ma mb
  else if t2b (et0 == half /\ et1 == float) then
    decorated_mma #Kuiper.Float16.Raw.t #Kuiper.Float32.Raw.t mc ma mb
  else if t2b (et0 == bf16 /\ et1 == float) then
    decorated_mma #Kuiper.BFloat16.Raw.t #Kuiper.Float32.Raw.t mc ma mb
  else if t2b (et0 == double /\ et1 == double) then
    decorated_mma #Kuiper.Float64.Raw.t #Kuiper.Float64.Raw.t mc ma mb
  else
    R.emma mc ma mb

private let ideals_eq #rank #dims #raw
  (c:chest #rank dims (M.value raw)) (r:chest dims real)
  : Lemma (requires forall i. (acc c i).ideal == acc r i)
          (ensures ideals c == r)
  = lemma_equal_intro (ideals c) r;
    ext (ideals c) r

private let decorated_mma_ideals
  (#raw0 #raw1:Type0)
  (#rows #shared #columns:nat)
  (mc:chest2 (M.value raw1) rows columns)
  (ma:chest2 (M.value raw0) rows shared)
  (mb:chest2 (M.value raw0) shared columns)
  : Lemma (ideals (decorated_mma mc ma mb) ==
      matplus (ideals mc) (matmul (ideals ma) (ideals mb)))
  = ideals_eq (decorated_mma mc ma mb)
      (matplus (ideals mc) (matmul (ideals ma) (ideals mb)))

instance mma_half_half : mma_approximation half half = {
  mma_approx = (fun #rows #shared #columns mc ma mb rc ra rb ->
    ideals_eq mc rc; ideals_eq ma ra; ideals_eq mb rb;
    decorated_mma_ideals mc ma mb);
}

instance mma_half_float : mma_approximation half float = {
  mma_approx = (fun #rows #shared #columns mc ma mb rc ra rb ->
    ideals_eq mc rc; ideals_eq ma ra; ideals_eq mb rb;
    decorated_mma_ideals mc ma mb);
}

instance mma_bf16_float : mma_approximation bf16 float = {
  mma_approx = (fun #rows #shared #columns mc ma mb rc ra rb ->
    ideals_eq mc rc; ideals_eq ma ra; ideals_eq mb rb;
    decorated_mma_ideals mc ma mb);
}

instance mma_double_double : mma_approximation double double = {
  mma_approx = (fun #rows #shared #columns mc ma mb rc ra rb ->
    ideals_eq mc rc; ideals_eq ma ra; ideals_eq mb rb;
    decorated_mma_ideals mc ma mb);
}
