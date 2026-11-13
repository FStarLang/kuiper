module KPlay.Proof.Base.RecurrentModel

#set-options "--fuel 1 --ifuel 1 --z3rlimit 40"

module P = KPlay.Proof.Shared.Projection
module R = KPlay.Proof.Shared.Recurrent
module H = KPlay.Proof.Base.RecurrentHeadModel
module Heads = KPlay.Proof.Base.RecurrentHeads

[@@"opaque_to_smt"]
noextract
let head (w:R.weights) (qkv:R.vector 6144) (z:R.vector 2048) (b a:R.vector 16)
  (old:R.recurrent_state) (h:P.index 16) : H.inputs = {
  H.raw_query = (fun i -> qkv (h * 128 + i));
  H.raw_key = (fun i -> qkv (2048 + h * 128 + i));
  H.raw_value = (fun i -> qkv (4096 + h * 128 + i));
  H.gate = (fun i -> z (h * 128 + i));
  H.norm_weight = w.R.gated_norm;
  H.beta_input = b h;
  H.decay_input = a h;
  H.decay_bias = w.R.dt_bias h;
  H.decay_log = w.R.a_log h;
  H.state = old h;
}

[@@"opaque_to_smt"]
noextract
let convolution (w:R.weights) (normalized:R.vector 1024) (old:R.convolution_state) : GTot R.convolution_state =
  R.shift_convolution old (R.project #6144 #1024 w.R.in_qkv normalized)

[@@"opaque_to_smt"]
noextract
let convolved (w:R.weights) (normalized:R.vector 1024) (old:R.convolution_state) : GTot (R.vector 6144) =
  R.convolve w.R.conv_kernel (convolution w normalized old)

[@@"opaque_to_smt"]
noextract
let heads (w:R.weights) (normalized:R.vector 1024) (old_conv:R.convolution_state) (old_state:R.recurrent_state)
  : GTot Heads.models =
  fun h -> head w (convolved w normalized old_conv)
    (R.project #2048 #1024 w.R.in_z normalized)
    (R.project #16 #1024 w.R.in_beta normalized)
    (R.project #16 #1024 w.R.in_a normalized) old_state h

[@@"opaque_to_smt"]
noextract
let gated (w:R.weights) (normalized:R.vector 1024) (old_conv:R.convolution_state) (old_state:R.recurrent_state)
  : GTot (R.vector 2048) =
  fun i -> H.gated (heads w normalized old_conv old_state (i / 128)) (i % 128)

[@@"opaque_to_smt"]
noextract
let mixed (w:R.weights) (normalized:R.vector 1024) (old_conv:R.convolution_state) (old_state:R.recurrent_state)
  : GTot (R.vector 1024) =
  R.project #1024 #2048 w.R.out_projection (gated w normalized old_conv old_state)
