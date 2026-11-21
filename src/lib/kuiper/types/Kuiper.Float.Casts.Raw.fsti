module Kuiper.Float.Casts.Raw

(* Representation conversions only. Semantic casts preserve the ideal field. *)
module F16 = Kuiper.Float16.Raw
module BF16 = Kuiper.BFloat16.Raw
module F32 = Kuiper.Float32.Raw
module F64 = Kuiper.Float64.Raw

val cast_f16_to_f32 : F16.t -> F32.t
val cast_f16_to_f64 : F16.t -> F64.t
val cast_f32_to_f16 : F32.t -> F16.t
val cast_f32_to_f64 : F32.t -> F64.t
val cast_bf16_to_f32 : BF16.t -> F32.t
val cast_f32_to_bf16 : F32.t -> BF16.t
val cast_f16_to_bf16 : F16.t -> BF16.t
val cast_bf16_to_f16 : BF16.t -> F16.t
val cast_bf16_to_f64 : BF16.t -> F64.t
val cast_f64_to_bf16 : F64.t -> BF16.t
val cast_f64_to_f16 : F64.t -> F16.t
val cast_f64_to_f32 : F64.t -> F32.t
