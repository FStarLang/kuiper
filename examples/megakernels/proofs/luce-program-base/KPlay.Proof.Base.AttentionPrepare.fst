module KPlay.Proof.Base.AttentionPrepare

#lang-pulse

open Kuiper
open FStar.Ghost

module Array = Pulse.Lib.Array
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module A = KPlay.Proof.Shared.Attention
module RMS = KPlay.Proof.Base.RMS
module Config = KPlay.Proof.Base.RMSConfigs
module Gates = KPlay.Proof.Base.AttentionGateCopy
module Rotation = KPlay.Proof.Base.AttentionRotation
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
let normalized (weights:Transfer.packed) (before:Transfer.scratch) (layer:A.full_attention_layer)
  : GTot Transfer.scratch =
  RMS.after (Config.key_norm layer)
    (RMS.after (Config.query_norm layer) (Gates.writes before 2048) weights 8) weights 2

inline_for_extraction noextract
let prepared (weights:Transfer.packed) (before:Transfer.scratch) (layer:A.full_attention_layer)
  : GTot Transfer.scratch =
  Rotation.after (normalized weights before layer) 10

inline_for_extraction noextract
fn apply
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000) (partial:larray FP.t 256)
  (layer:A.full_attention_layer)
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch) (#vp:erased (lseq FP.t 256)) (#fraction:perm)
  preserves gpu ** Array.pts_to weights #fraction vw
  requires Array.pts_to hidden vh ** Array.pts_to partial (reveal vp)
  ensures exists* (temporary:lseq FP.t 256).
    Array.pts_to hidden (prepared vw vh layer) ** Array.pts_to partial temporary
{
  Gates.apply hidden;
  let query = Config.query_norm layer;
  RMS.apply query weights hidden partial;
  let key = Config.key_norm layer;
  RMS.apply key weights hidden partial;
  Rotation.apply hidden;
}

inline_for_extraction noextract
fn source_apply
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000) (layer:A.full_attention_layer)
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch) (#fraction:perm)
  preserves gpu ** Array.pts_to weights #fraction vw
  requires Array.pts_to hidden vh
  ensures Array.pts_to hidden (prepared vw vh layer)
{
  Gates.apply hidden;
  RMS.source_apply (Config.query_norm layer) weights hidden;
  RMS.source_apply (Config.key_norm layer) weights hidden;
  Rotation.apply hidden;
}
