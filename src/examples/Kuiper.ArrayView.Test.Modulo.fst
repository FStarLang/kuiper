module Kuiper.ArrayView.Test.Modulo

(* Striding an array. *)
#lang-pulse

open Kuiper
open Kuiper.View
open Kuiper.VArray
open Kuiper.Bijection
open Kuiper.Injection
module IView = Kuiper.IView
module SZ    = FStar.SizeT

// Can we use divup here? It seems much harder on Z3.
noextract
let strided_view et (len : nat) (stride : nat) (offset : natlt stride) :
  aview et len (lseq et ((len + stride - 1 - offset) / stride))
= {
  iview = {
    ait = natlt ((len + stride - 1 - offset) / stride);
    ait_enum = solve;
    imap = {
      f = (fun (i : natlt ((len + stride - 1 - offset) / stride)) -> i * stride + offset <: natlt len);
      is_inj = ez;
    }
  };
  igm = solve;
}

let even_view et len : aview et len _ = strided_view et len 2 0
let odd_view  et len : aview et len _ = strided_view et len 2 1

inline_for_extraction noextract
instance _cview_strided
   (#et : Type) (#len : erased nat{SZ.fits len})
   (stride : sz) (offset : szlt stride)
: IView.cview (strided_view et len stride offset).iview
= {
  fits = ();
  cit  = szlt ((len + stride - 1 - offset) / stride);
  bij  = fin_size_t_bij _;
  imap = {
    f = (fun (i : szlt ((len + stride - 1 - offset) / stride)) -> i `SZ.mul` stride `SZ.add` offset <: szlt len);
    is_inj = ez;
  };
  compat = ez;
}

(* force extraction *)
let x = 1ul