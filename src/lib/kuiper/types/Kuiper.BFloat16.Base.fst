module Kuiper.BFloat16.Base

open Kuiper.Real
open Kuiper.Float.Model
module B = Kuiper.BFloat16.Raw
module M = Kuiper.Float.Model
module R = Kuiper.Float.Realops

let t = M.value B.t
let zero : t = { repr = B.zero; ideal = 0.0R }
let one : t = { repr = B.one; ideal = 1.0R }
let add (x y:t) : t = M.binary B.add R.r_add x y
let mul (x y:t) : t = M.binary B.mul R.r_mul x y
let lt (x y:t) = B.lt x.repr y.repr
let lte (x y:t) = B.lte x.repr y.repr
let eq (x y:t) = B.eq x.repr y.repr
let bit_eq (x y:t) = B.bit_eq x.repr y.repr
let neg (x:t) : t = M.unary B.neg R.r_neg x
let sub (x y:t) : t = M.binary B.sub R.r_sub x y
let div (x y:t) : t = M.binary B.div M.div x y
let of_int (x:Int64.t) : t = { repr = B.of_int x; ideal = R.r_of_int x }
let of_int_zero = let _ = B.of_int_zero in ()
let of_int_one = let _ = B.of_int_one in ()
let of_literal (x:string) : t = { repr = B.of_literal x; ideal = R.r_of_literal x }
let kind (x:t) = B.kind x.repr
let is_zero (x:t) = B.is_zero x.repr
let bits (x:t) = B.bits x.repr
let largest : t = { repr = B.largest; ideal = 0.0R }
let infinity : t = { repr = B.infinity; ideal = 0.0R }
let kind_one = B.kind_one
let kind_zero = B.kind_zero
let kind_largest = B.kind_largest
let kind_infinity = B.kind_infinity
let zero_is_zero = B.zero_is_zero
let one_is_nonzero = B.one_is_nonzero
let bit_eq_spec (x y:t) = B.bit_eq_spec x.repr y.repr
let bits_spec (x y:t) = B.bits_spec x.repr y.repr
let is_zero_spec (x:t) = B.is_zero_spec x.repr
let eq_spec (x y:t) = B.eq_spec x.repr y.repr
let lte_is_lt_or_eq (x y:t) = B.lte_is_lt_or_eq x.repr y.repr
let neg_kind (x:t) = B.neg_kind x.repr
let neg_is_zero (x:t) = B.neg_is_zero x.repr
let zero_sub_kind (x:t) = B.zero_sub_kind x.repr
let zero_sub_involutive (x:t) = B.zero_sub_involutive x.repr
let lt_zero_sub_flip (x y:t) = B.lt_zero_sub_flip x.repr y.repr
let zero_sub_is_neg (x:t) = B.zero_sub_is_neg x.repr
let neg_neg (x:t) = B.neg_neg x.repr
let lt_neg_flip (x y:t) = B.lt_neg_flip x.repr y.repr
let negate_lt_is_lte (x y:t) = B.negate_lt_is_lte x.repr y.repr
let add_comm (x y:t) = B.add_comm x.repr y.repr
let mul_comm (x y:t) = B.mul_comm x.repr y.repr
let add_zero (x:t) = B.add_zero x.repr
let mul_zero (x:t) = B.mul_zero x.repr
let mul_one (x:t) = B.mul_one x.repr
let sub_is_add_neg (x y:t) = B.sub_is_add_neg x.repr y.repr
let sub_is_add_zero_sub (x y:t) = B.sub_is_add_zero_sub x.repr y.repr
let largest_val_spec (x:t) = B.largest_val_spec x.repr
let infinity_val_spec (x:t) = B.infinity_val_spec x.repr
let fmax (x y:t) : t = M.binary B.fmax R.r_fmax x y
let fmax_spec (x y:t) = B.fmax_spec x.repr y.repr
let fexp (x:t) : t = M.unary B.fexp R.r_exp x
let exp_special_values (x:t) = B.exp_special_values x.repr
let flog (x:t) : t = M.unary B.flog M.log x
let fexpm1 (x:t) : t = M.unary B.fexpm1 R.r_expm1 x
let flog1p (x:t) : t = M.unary B.flog1p M.log1p x
let sqrt (x:t) : t = M.unary B.sqrt M.sqrt x
let rsqrt (x:t) : t = M.unary B.rsqrt M.rsqrt x
let sin (x:t) : t = M.unary B.sin R.r_sin x
let cos (x:t) : t = M.unary B.cos R.r_cos x
let tan (x:t) : t = M.unary B.tan M.tan x
let asin (x:t) : t = M.unary B.asin M.asin x
let acos (x:t) : t = M.unary B.acos M.acos x
let atan (x:t) : t = M.unary B.atan R.r_atan x
let sinh (x:t) : t = M.unary B.sinh R.r_sinh x
let cosh (x:t) : t = M.unary B.cosh R.r_cosh x
let tanh (x:t) : t = M.unary B.tanh R.r_tanh x
let ceil (x:t) : t = M.unary B.ceil R.r_ceil x
let floor (x:t) : t = M.unary B.floor R.r_floor x
let round (x:t) : t = M.unary B.round R.r_round x
let fabs (x:t) : t = M.unary B.fabs R.r_fabs x
let erf (x:t) : t = M.unary B.erf R.r_erf x
let log2 (x:t) : t = M.unary B.log2 M.log2 x
let log10 (x:t) : t = M.unary B.log10 M.log10 x
let exp2 (x:t) : t = M.unary B.exp2 R.r_exp2 x
let pow (x y:t) : t = M.binary B.pow M.pow x y
let atan2 (x y:t) : t = M.binary B.atan2 R.r_atan2 x y
let fmin (x y:t) : t = M.binary B.fmin R.r_fmin x y
let fmod (x y:t) : t = M.binary B.fmod M.fmod x y
let copysign (x y:t) : t = M.binary B.copysign R.r_copysign x y
let fma (x y z:t) : t = M.ternary B.fma R.r_fma x y z
