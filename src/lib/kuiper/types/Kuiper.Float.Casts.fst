module Kuiper.Float.Casts

#lang-pulse
open Kuiper
open Kuiper.Approximates.Base
open Kuiper.Float.Casts.Base
module F16 = Kuiper.Float16
module BF16 = Kuiper.BFloat16
module F32 = Kuiper.Float32
module F64 = Kuiper.Float64

inline_for_extraction noextract
instance c_16_16 : float_cast F16.t F16.t = { fcast = id }
inline_for_extraction noextract
instance c_16_32 : float_cast F16.t F32.t = { fcast = cast_f16_to_f32 }
inline_for_extraction noextract
instance c_16_64 : float_cast F16.t F64.t = { fcast = cast_f16_to_f64 }
inline_for_extraction noextract
instance c_32_16 : float_cast F32.t F16.t = { fcast = cast_f32_to_f16 }
inline_for_extraction noextract
instance c_32_32 : float_cast F32.t F32.t = { fcast = id }
inline_for_extraction noextract
instance c_32_64 : float_cast F32.t F64.t = { fcast = cast_f32_to_f64 }
inline_for_extraction noextract
instance c_64_16 : float_cast F64.t F16.t = { fcast = cast_f64_to_f16 }
inline_for_extraction noextract
instance c_64_32 : float_cast F64.t F32.t = { fcast = cast_f64_to_f32 }
inline_for_extraction noextract
instance c_64_64 : float_cast F64.t F64.t = { fcast = id }
inline_for_extraction noextract
instance c_bf16_bf16 : float_cast BF16.t BF16.t = { fcast = id }
inline_for_extraction noextract
instance c_bf16_32 : float_cast BF16.t F32.t = { fcast = cast_bf16_to_f32 }
inline_for_extraction noextract
instance c_32_bf16 : float_cast F32.t BF16.t = { fcast = cast_f32_to_bf16 }
inline_for_extraction noextract
instance c_16_bf16 : float_cast F16.t BF16.t = { fcast = cast_f16_to_bf16 }
inline_for_extraction noextract
instance c_bf16_16 : float_cast BF16.t F16.t = { fcast = cast_bf16_to_f16 }
inline_for_extraction noextract
instance c_bf16_64 : float_cast BF16.t F64.t = { fcast = cast_bf16_to_f64 }
inline_for_extraction noextract
instance c_64_bf16 : float_cast F64.t BF16.t = { fcast = cast_f64_to_bf16 }
