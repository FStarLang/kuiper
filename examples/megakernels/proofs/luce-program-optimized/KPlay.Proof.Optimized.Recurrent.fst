module KPlay.Proof.Optimized.Recurrent

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
module S = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module G = FStar.Ghost
module Norm = KPlay.Proof.Optimized.Norm
module Matrix = KPlay.Proof.Optimized.Matrix
module Dot = KPlay.Proof.Optimized.Dot
module Head = KPlay.Proof.Optimized.Head
module Conv = KPlay.Proof.Optimized.Convolution
module Decay = KPlay.Proof.Optimized.Decay
module Scalar = KPlay.Proof.Optimized.Scalar
module Strided = KPlay.Proof.Optimized.Strided
module Delta = KPlay.Proof.Optimized.Delta
module Memory = KPlay.Proof.Optimized.Memory
module Block = KPlay.Proof.Optimized.Block

noextract
noeq type observation = {
  projected: lseq f32 8192;
  hidden: lseq bf16 1024;
  recurrent: lseq f32 262144;
  convolution: lseq f32 24576;
}

noextract
fn source_block
  (weights:larray bf16 21555360) (hidden normalized:larray bf16 1024)
  (projected:larray f32 8192) (activation:larray f32 2048)
  (states:larray f32 262144) (convolution:larray f32 24576)
  (#wv:erased (lseq bf16 21555360)) (#hv:erased (lseq bf16 1024))
  (#sv:erased (lseq f32 262144)) (#cv:erased (lseq f32 24576))
  (#nv:erased (lseq (option bf16) 1024)) (#pv:erased (lseq (option f32) 8192))
  (#av:erased (lseq (option f32) 2048)) (#fw:perm)
  preserves gpu ** weights |-> Frac fw wv
  requires hidden |-> hv ** states |-> sv ** convolution |-> cv **
    A.pts_to_mask normalized nv Memory.all_cells **
    A.pts_to_mask projected pv Memory.all_cells **
    A.pts_to_mask activation av Memory.all_cells
  returns observed:erased observation
  ensures hidden |-> observed.hidden ** states |-> observed.recurrent **
    convolution |-> observed.convolution
  ensures exists* (next_norm:lseq (option bf16) 1024)
                  (next_projected:lseq (option f32) 8192)
                  (next_activation:lseq (option f32) 2048).
    A.pts_to_mask normalized next_norm Memory.all_cells **
    A.pts_to_mask projected next_projected Memory.all_cells **
    A.pts_to_mask activation next_activation Memory.all_cells **
    pure (Memory.ready next_norm 0 1024 /\ Memory.ready next_projected 0 8192 /\
      Memory.ready next_activation 0 2048)
  ensures pure (observed.convolution == Conv.heads_prefix cv observed.projected 16)
{
  let mut beta:A.array f32 = [| 16sz |];
  let mut alpha:A.array f32 = [| 16sz |];
  let mut query:A.array f32 = [| 128sz |];
  let mut key:A.array f32 = [| 128sz |];
  let mut value:A.array f32 = [| 128sz |];
  Norm.normalize_hidden weights hidden normalized 0sz;
  Matrix.project_bf16 weights normalized projected 6144sz
    (SZ.uint32_to_sizet 11012096ul) 0sz;
  Matrix.project_bf16 weights normalized projected 2048sz
    (SZ.uint32_to_sizet 17303552ul) 6144sz;
  A.from_mask projected;
  with raw_values. assert projected |-> raw_values;
  let raw:erased (lseq f32 8192) = G.hide (G.reveal #(S.seq f32) raw_values);
  let mut head = 0sz;
  while (!head <^ 16sz)
    invariant exists* (h:sz{h <= 16}) (bv av:lseq (option f32) 16).
      head |-> h ** A.pts_to_mask beta bv Memory.all_cells **
      A.pts_to_mask alpha av Memory.all_cells **
      pure (Memory.ready bv 0 (SZ.v h) /\ Memory.ready av 0 (SZ.v h))
    decreases (16 - !head)
  {
    let h = !head;
    let b = Dot.dot_bf16 weights normalized (SZ.uint32_to_sizet 19400704ul +^ h *^ 1024sz);
    A.mask_write beta h b;
    let a = Dot.dot_bf16 weights normalized (SZ.uint32_to_sizet 19417088ul +^ h *^ 1024sz);
    A.mask_write alpha h a;
    head := h +^ 1sz;
  };
  A.from_mask beta;
  A.from_mask alpha;
  head := 0sz;
  Conv.heads_zero cv raw;
  while (!head <^ 16sz)
    invariant exists* (h:sz{h <= 16}) (current_states:lseq f32 262144)
                     (current_activation:lseq (option f32) 2048).
      head |-> h ** states |-> current_states **
      convolution |-> Conv.heads_prefix cv raw (SZ.v h) **
      A.pts_to_mask activation current_activation Memory.all_cells **
      pure (Memory.ready current_activation 0 (SZ.v h * 128))
    invariant exists* (qv kv vv:lseq (option f32) 128).
      A.pts_to_mask query qv Memory.all_cells **
      A.pts_to_mask key kv Memory.all_cells **
      A.pts_to_mask value vv Memory.all_cells
    decreases (16 - !head)
  {
    let h = !head;
    Conv.region weights convolution projected query h 0sz;
    Conv.region weights convolution projected key h 1sz;
    Conv.region weights convolution projected value h 2sz;
    Conv.heads_successor cv raw (SZ.v h);
    let b = Scalar.sigmoid32 (beta.(h));
    let decay = Decay.decay weights h (alpha.(h));
    Head.normalize_qk query key;
    let kq = Strided.strided_dot key query 128sz;
    let mut row = 0sz;
    while (!row <^ 128sz)
      invariant exists* (j:sz{j <= 128}) (current_states:lseq f32 262144)
                       (current_activation:lseq (option f32) 2048).
        row |-> j ** states |-> current_states **
        A.pts_to_mask activation current_activation Memory.all_cells **
        pure (Memory.ready current_activation 0 (SZ.v h * 128 + SZ.v j))
      decreases (128 - !row)
    {
      let j = !row;
      with before. assert states |-> before;
      let offset = h *^ 16384sz +^ j *^ 128sz;
      let state_row = Memory.take states offset 128sz;
      let v = value.(j);
      let _ = Delta.update_row state_row key query activation
        (h *^ 128sz +^ j) v b decay kq;
      with next_row. assert state_row |-> next_row;
      Memory.give states (SZ.v offset) 128 #before #next_row #1.0R;
      row := j +^ 1sz;
    };
    with readout_values. assert A.pts_to_mask activation readout_values Memory.all_cells;
    let initialized : squash (Memory.ready readout_values (SZ.v h * 128) 128) = ();
    let readout = Memory.borrow activation (h *^ 128sz) 128sz
      #readout_values #1.0R initialized;
    let norm_offset = SZ.uint32_to_sizet 19458080ul;
    let norm_weights = Memory.take weights norm_offset 128sz;
    let gates = Memory.take projected (6144sz +^ h *^ 128sz) 128sz;
    Head.gate_readout norm_weights readout gates;
    with gated. assert readout |-> gated;
    Memory.restore activation (SZ.v h * 128) 128 #readout_values #gated #1.0R;
    Memory.give projected (6144 + SZ.v h * 128) 128 #raw
      #(S.slice raw (6144 + SZ.v h * 128) (6272 + SZ.v h * 128)) #1.0R;
    Memory.overwrite_readonly #f32 #8192 raw (6144 + SZ.v h * 128) 128;
    Memory.give weights (SZ.v norm_offset) 128 #wv
      #(S.slice wv (SZ.v norm_offset) (SZ.v norm_offset + 128)) #fw;
    Memory.overwrite_readonly wv (SZ.v norm_offset) 128;
    A.to_mask query;
    A.to_mask key;
    A.to_mask value;
    head := h +^ 1sz;
  };
  A.from_mask activation;
  A.to_mask normalized;
  A.to_mask projected;
  Block.finish weights activation hidden normalized projected
    (SZ.uint32_to_sizet 19458208ul) 1024sz 2048sz
    (SZ.uint32_to_sizet 3672064ul) (SZ.uint32_to_sizet 7342080ul);
  with after_hidden. assert hidden |-> after_hidden;
  with after_states. assert states |-> after_states;
  with after_conv. assert convolution |-> after_conv;
  let observed:erased observation = G.hide ({
    projected = raw;
    hidden = after_hidden;
    recurrent = after_states;
    convolution = after_conv;
  } <: observation);
  A.to_mask normalized;
  A.to_mask activation;
  A.to_mask beta;
  A.to_mask alpha;
  observed
}
