module Kuiper.Spec.Attention
#lang-pulse
open Kuiper
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Bijection

module MS = Kuiper.Spec.GEMM

module RSMX = Kuiper.Kernel.RowSoftmax

(* Pre-softmax attention scores. *)
let attn_scores
  (#l #s #e: pos)
  (eQ : chest (l @| e @| INil) real)
  (eK : chest (e @| s @| INil) real)
  (bias : chest (l @| s @| INil) real)
  (scale : real)
  : chest (l @| s @| INil) real
  = chest_comb (fun bias_qk score -> (bias_qk +. (score *. scale))) bias (MS.matmul eQ eK)

(* Top-level real-valued spec *)
let attention_real
  (#l #s #e #ev : pos)
  (eQ : chest2 real l e)
  (eK : chest2 real e s)
  (eV : chest2 real s ev)
  (bias : chest2 real l s)
  (scale : real)
  : GTot (chest2 real l ev)
  = let scores = attn_scores eQ eK bias scale in
    let probs  = RSMX.row_softmax_real scores in
    MS.matmul probs eV

let attention_real_batched
  (#n #h #l #s #e #ev : pos)
  (rQ:    chest4 real n h l e)
  (rKT:   chest4 real n h e s)
  (rV:    chest4 real n h s ev)
  (rbias: chest4 real n h l s)
  (scale : real)  
  : GTot (chest4 real n h l ev)
  = mk4 (fun i j -> 
      acc2 (attention_real
        (slice_page4 rQ i j)
        (slice_page4 rKT i j)
        (slice_page4 rV i j)
        (slice_page4 rbias i j)
        scale))

(* Specs for attention with log-sum-exp.
 LATER: just separate the LSE stuff out so we aren't dealing with tuples & etc. *)

(* row-wise log sum exp of scores *)
let attn_lse
  (#l #s : pos)
  (scores : chest (l @| s @| INil) real)
  : GTot (chest1 real l)
  = mk1 (fun i -> log (chest1_rsum (chest_map exp (chest2_row scores i))))

(* Top-level real-valued spec: (output, log-sum-exp) given real inputs. *)
let attention_real_lse
  (#l #s #e #ev : pos)
  (eQ : chest2 real l e)
  (eK : chest2 real e s)
  (eV : chest2 real s ev)
  (bias : chest2 real l s)
  (scale : real)
  : GTot (chest2 real l ev & chest1 real l)
  = let scores = attn_scores eQ eK bias scale in
    let probs  = RSMX.row_softmax_real scores in
    let out    = MS.matmul probs eV in
    let lse    = attn_lse scores in
    (out, lse)

let attention_real_batched_lse
  (#n #h #l #s #e #ev : pos)
  (rQ : chest (n @| h @| l @| e @| INil) real)
  (rKT : chest (n @| h @| e @| s @| INil) real)
  (rV : chest (n @| h @| s @| ev @| INil) real)
  (rbias : chest (n @| h @| l @| s @| INil) real)
  (scale : real)
  : GTot (chest (n @| h @| l @| ev @| INil) real & chest (n @| h @| l @| INil) real)
  = let attn_tile = fun i j -> attention_real_lse
            (slice_page4 rQ i j)
            (slice_page4 rKT i j)
            (slice_page4 rV i j)
            (slice_page4 rbias i j)
            scale in
    let out_spec = mk4 fun i j -> acc2 (fst (attn_tile i j)) in
    let lse_spec = mk3 fun i j -> acc1 (snd (attn_tile i j)) in
    (out_spec, lse_spec)