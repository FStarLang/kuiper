module KPlay.Functional

#lang-pulse

open Kuiper

include KPlay.Proof.Optimized.Cache

module Layout = KPlay.Proof.Optimized.Layout
module Block = KPlay.Proof.Optimized.Block
module Head = KPlay.Proof.Optimized.Head
module Convolution = KPlay.Proof.Optimized.Convolution
module Decay = KPlay.Proof.Optimized.Decay
module Delta = KPlay.Proof.Optimized.Delta
module Final = KPlay.Proof.Optimized.Final
module Recurrent = KPlay.Proof.Optimized.Recurrent
module FamilyRecurrent = KPlay.Proof.Optimized.FamilyRecurrent
module FamilyAttention = KPlay.Proof.Optimized.FamilyAttention
module FamilyProjection = KPlay.Proof.Optimized.FamilyProjection
module FamilyEmbedding = KPlay.Proof.Optimized.FamilyEmbedding
module FamilySelection = KPlay.Proof.Optimized.FamilySelection
module AttentionOps = KPlay.Proof.Optimized.AttentionOps
module Online = KPlay.Proof.Optimized.Online
module Attention = KPlay.Proof.Optimized.Attention
module Selector = KPlay.Proof.Optimized.Selector
module Scores = KPlay.Proof.Optimized.Scores
module PackedOwnership = KPlay.Proof.Optimized.PackedOwnership
module Trace = KPlay.Proof.Optimized.Trace
module State = KPlay.Proof.Optimized.State
module Program = KPlay.Proof.Optimized.Program
module Stream = Kuiper.Kernel.Stream
module Epoch = Kuiper.Epoch
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module NativeInput = KPlay.Proof.NativeInput
module I64 = FStar.Int64

noextract
fn program_refines
  (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (steps:I64.t{I64.v steps == call.Model.decode_steps}) (stream:Stream.stream_t)
  (#epoch:Epoch.epoch_t) (#weights:erased Layout.packed)
  (#physical_prompt:erased (lseq I64.t call.Model.prompt_length))
  (#prompt:erased (D.vec (D.token 248320) call.Model.prompt_length)) (#permission:perm)
  preserves cpu
  requires Program.pre call inputs stream epoch weights physical_prompt prompt permission
  returns result:Program.outcome
  ensures Program.post call inputs stream epoch weights physical_prompt prompt permission result
