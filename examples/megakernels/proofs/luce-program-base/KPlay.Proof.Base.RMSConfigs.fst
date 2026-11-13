module KPlay.Proof.Base.RMSConfigs

module SZ = Kuiper.SizeT
module Model = KPlay.Proof.Shared.Model
module A = KPlay.Proof.Shared.Attention
module Inputs = KPlay.Proof.Base.Inputs
module RMS = KPlay.Proof.Base.RMS

inline_for_extraction noextract
let model_norm (layer:Model.cursor) (post:bool{layer = 24 ==> not post}) : RMS.layout =
  let offset = if layer = 24 then 254279680
    else (Inputs.layer_extent layer; Inputs.layer_base layer + (if post then 1024 else 0)) in
  {
    RMS.width = 1024sz; RMS.rows = 1sz; RMS.source = 0sz; RMS.destination = 1024sz;
    RMS.stride = 1024sz; RMS.weight_offset = SZ.uint_to_t offset; RMS.bounds = ();
  }

inline_for_extraction noextract
let query_norm (layer:A.full_attention_layer) : RMS.layout =
  Inputs.layer_extent layer;
  {
    RMS.width = 256sz; RMS.rows = 8sz; RMS.source = 2048sz; RMS.destination = 2048sz;
    RMS.stride = 512sz; RMS.weight_offset = SZ.uint_to_t (Inputs.layer_base layer + 16254976);
    RMS.bounds = ();
  }

inline_for_extraction noextract
let key_norm (layer:A.full_attention_layer) : RMS.layout =
  Inputs.layer_extent layer;
  {
    RMS.width = 256sz; RMS.rows = 2sz; RMS.source = 6144sz; RMS.destination = 6144sz;
    RMS.stride = 256sz; RMS.weight_offset = SZ.uint_to_t (Inputs.layer_base layer + 16255232);
    RMS.bounds = ();
  }
