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
  aview et (lseq et ((len + stride - 1 - offset) / stride))
= {
  iview = {
    len = len;
    ait = natlt ((len + stride - 1 - offset) / stride);
    step = {
      imap = {
        f = (fun (i : natlt ((len + stride - 1 - offset) / stride)) -> i * stride + offset <: natlt len);
        is_inj = ez;
      };
    };
  };
  ctn = solve;
}

let even_view et len : aview et _ = strided_view et len 2 0
let odd_view  et len : aview et _ = strided_view et len 2 1

inline_for_extraction noextract
instance _cview_strided
   (#et : Type) (#len : erased int{SZ.fits len})
   (sz_len : concrete_sz len)
   (stride : sz) (offset : szlt stride)
: IView.ciview (strided_view et len stride offset).iview
= {
  len_fits = ();
  sch = {
    cit  = szlt ((len + stride - 1 - offset) / stride);
    bij  = natural;
  };
  step = {
    cimap = mk_cinj (fun (i : szlt ((len + stride - 1 - offset) / stride)) -> i `SZ.mul` stride `SZ.add` offset <: szlt len);
    compat = ez;
  };
}

(* force extraction *)
let x = 1ul
