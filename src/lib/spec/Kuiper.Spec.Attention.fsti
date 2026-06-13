// TEMPORARY - MERGE INTO APPROPRIATE PLACES IN Kuiper.Tensor, Kuiper.Chest, etc.
// Or rewrite EMatrix2, EMatrix3, etc. to be defined with CH.t

module Kuiper.Spec.Attention
#lang-pulse
open Kuiper
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor
open Kuiper.Index
open Kuiper.Bijection

module SMX = Kuiper.Spec.Softmax
module MS = Kuiper.Spec.GEMM
module CH = Kuiper.Chest
module SZ = Kuiper.SizeT

let tup_of_idesc4
  (#d0 #d1 #d2 #d3 : nat) 
  (idx : abs (d0 @| d1 @| d2 @| d3 @| INil)) : (natlt d0 & natlt d1 & natlt d2 & natlt d3) =
  let (i,(j,(k,(l,())))) = idx in
  (i,j,k,l)

let tup_of_idesc3
  (#d0 #d1 #d2 : nat) 
  (idx : abs (d0 @| d1 @| d2 @| INil)) : (natlt d0 & natlt d1 & natlt d2) =
  let (i,(j,(k,()))) = idx in
  (i,j,k)

let tup_of_idesc2
  (#d0 #d1 : nat) 
  (idx : abs (d0 @| d1 @| INil)) : (natlt d0 & natlt d1) =
  let (i,(j,())) = idx in
  (i,j)

let tensor2_row 
  (#et : Type0) {| scalar et |}
  (#m #n : nat)
  (e: CH.t (m @| n @| INil) et)
  (i : natlt m)
  : GTot (lseq et n)
  = Seq.init_ghost n (fun l -> CH.acc e (i,(l,())))

let chest_mk4 (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (f : natlt d0 -> natlt d1 -> natlt d2 -> natlt d3 -> GTot et)
  : CH.t (d0 @| d1 @| d2 @| d3 @| INil) et
  = CH.mk (d0 @| d1 @| d2 @| d3 @| INil) (fun idx ->
      let i,j,k,l = tup_of_idesc4 idx in
      f i j k l)

let chest_mk3 (#et:Type) (#d0 #d1 #d2 : nat)
  (f : natlt d0 -> natlt d1 -> natlt d2 -> GTot et)
  : CH.t (d0 @| d1 @| d2 @| INil) et
  = CH.mk (d0 @| d1 @| d2 @| INil) (fun idx ->
      let i,j,k = tup_of_idesc3 idx in
      f i j k)

let chest_mk2 (#et:Type) (#d0 #d1 : nat)
  (f : natlt d0 -> natlt d1 -> GTot et)
  : CH.t (d0 @| d1 @| INil) et
  = CH.mk (d0 @| d1 @| INil) (fun idx ->
      let i,j = tup_of_idesc2 idx in
      f i j)
    
let chest_mk1 (#et:Type) (#d0 : nat)
  (f : natlt d0 -> GTot et)
  : CH.t (d0 @| INil) et
  = CH.mk (d0 @| INil) (fun (i,()) -> f i)

let tensor_matmul
  (#et : Type0) {| scalar et |}
  (#m #n #k : nat)
  (eA: CH.t (m @| k @| INil) et)
  (eB: CH.t (k @| n @| INil) et)
  : GTot (CH.t (m @| n @| INil) et)
  = chest_mk2 (fun i j ->
      let rowA = fun l -> CH.acc eA (i,(l,())) in
      let colB = fun l -> CH.acc eB (l,(j,())) in
      let products = Seq.init_ghost k (fun l -> rowA l `mul` colB l) in
      seq_fold_left (fun acc x -> acc `add` x) zero products)

let tensor_row_softmax_real
  (#m #n : nat)
  (eA: CH.t (m @| n @| INil) real)
  : GTot (CH.t (m @| n @| INil) real)
  = chest_mk2 (fun i j ->
      let row = fun l -> CH.acc eA (i,(l,())) in
      Seq.index (SMX.softmax_real (Seq.init_ghost n row)) j)
