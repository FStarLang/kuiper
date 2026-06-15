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
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.Bijection

module SMX = Kuiper.Spec.Softmax
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module EM3 = Kuiper.EMatrix3
module EM4 = Kuiper.EMatrix4

module RSMX = Kuiper.Kernel.RowSoftmax

// TODO: consistent usage of ematrix and chest forms? review 

(* Pre-softmax attention scores. *)
let attn_scores
  (#l #s #e: pos)
  (eQ : chest (l @| e @| INil) real)
  (eK : chest (e @| s @| INil) real)
  (bias : chest (l @| s @| INil) real)
  (scale : real)
  : chest (l @| s @| INil) real
  = chest_comb (fun bias_qk score -> (bias_qk +. score) *. scale) bias (MS.matmul eQ eK)

(* row-wise log sum exp of scores *)
let attn_lse
  (#l #s : pos)
  (scores : chest (l @| s @| INil) real)
  : GTot (lseq real l)
  = Seq.init_ghost l (fun i -> log (rsum (seq_map exp (ematrix_row scores i))))

(* Top-level real-valued spec: (output, log-sum-exp) given real inputs. *)
let attention_real
  (#l #s #e #ev : pos)
  (eQ : ematrix real l e)
  (eK : ematrix real e s)
  (eV : ematrix real s ev)
  (bias : ematrix real l s)
  (scale : real)
  : GTot (ematrix real l ev & lseq real l)
  = let scores = attn_scores eQ eK bias scale in
    let probs  = RSMX.row_softmax_real scores in
    let out    = MS.matmul probs eV in
    let lse    = attn_lse scores in
    (out, lse)

let attention_real_batched
  (#n #h #l #s #e #ev : pos)
  (rQ : chest (n @| h @| l @| e @| INil) real)
  (rKT : chest (n @| h @| e @| s @| INil) real)
  (rV : chest (n @| h @| s @| ev @| INil) real)
  (rbias : chest (n @| h @| l @| s @| INil) real)
  (scale : real)
  : GTot (chest (n @| h @| l @| ev @| INil) real & chest (n @| h @| l @| INil) real)
  = let attn_tile = fun i j -> attention_real
            (EM4.slice_page rQ i j)
            (EM4.slice_page rKT i j)
            (EM4.slice_page rV i j)
            (EM4.slice_page rbias i j)
            scale in
    let out_spec = EM4.mkM fun i j -> macc (fst (attn_tile i j)) in 
    let lse_spec = EM3.mkM fun i j -> Seq.index (snd (attn_tile i j)) in 
    (out_spec, lse_spec)