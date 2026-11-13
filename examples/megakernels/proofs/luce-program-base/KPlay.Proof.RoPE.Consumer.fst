module KPlay.Proof.RoPE.Consumer

open FStar.Real
module Source = KPlay.Proof.RoPE.Source
module Constants = Luce.RoPE.Constants
module Attention = KPlay.Proof.Shared.Attention
module A = Kuiper.Approximates.Base
module F = Kuiper.Floating
module FP = Kuiper.Float32
module BF = Kuiper.BFloat16
module Casts = Kuiper.Float.Casts
module I64 = FStar.Int64
module Trig = Kuiper.Real.Trigonometry
module S = FStar.Seq.Base

let identical_source_frequency (j:nat{j < 32})
  : Lemma (Source.inverse_frequency j == Constants.inverse_frequency j) =
  match j with
  | 0 -> ()
  | 1 -> ()
  | 2 -> ()
  | 3 -> ()
  | 4 -> ()
  | 5 -> ()
  | 6 -> ()
  | 7 -> ()
  | 8 -> ()
  | 9 -> ()
  | 10 -> ()
  | 11 -> ()
  | 12 -> ()
  | 13 -> ()
  | 14 -> ()
  | 15 -> ()
  | 16 -> ()
  | 17 -> ()
  | 18 -> ()
  | 19 -> ()
  | 20 -> ()
  | 21 -> ()
  | 22 -> ()
  | 23 -> ()
  | 24 -> ()
  | 25 -> ()
  | 26 -> ()
  | 27 -> ()
  | 28 -> ()
  | 29 -> ()
  | 30 -> ()
  | _ -> ()

let actual_rotary_components (p:Source.position) (j:nat{j < 32})
  : Lemma (
      A.v_approximates (Source.rotary_component p j) (Attention.rope_cos (I64.v p) j) /\
      A.v_approximates (Source.rotary_component p (32 + j)) (Attention.rope_sin (I64.v p) j)) =
  identical_source_frequency j;
  Constants.inverse_frequency_approx j;
  A.of_int_approx #FP.t p;
  let inverse = Source.inverse_frequency j in
  let position = F.of_int #FP.t p in
  let angle = Source.rotary_angle p j in
  let xr = Attention.rope_angle (I64.v p) j in
  assert (A.v_approximates inverse (Attention.inverse_frequency j));
  A.a_mul inverse position (Attention.inverse_frequency j) (of_int (I64.v p));
  FP.cos_approx angle xr;
  FP.sin_approx angle xr;
  Casts.fcast_approx #FP.t #BF.t (F.cos angle) (Trig.cos xr);
  Casts.fcast_approx #FP.t #BF.t (F.sin angle) (Trig.sin xr)

let producer_consumer (p:Source.position) (cells:Kuiper.lseq BF.t 64) (j:nat{j < 32})
  : Lemma
      (requires Source.rotary_post p cells)
      (ensures
        A.v_approximates (S.index cells j) (Attention.rope_cos (I64.v p) j) /\
        A.v_approximates (S.index cells (32 + j)) (Attention.rope_sin (I64.v p) j)) =
  actual_rotary_components p j
