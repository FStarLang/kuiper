module Kuiper.Example.FloatTests

open Kuiper.Floating

// Exactly one of these functions is true
let test_kind_0 #t {| floating t |} (x : t) =
  let c (f : t -> GTot bool) = if f x then 1 else 0 in
  assert c is_finite + c is_inf + c is_nan == 1

let test_mul_zero_fin #t {| floating t |} (x : t{is_finite x}) =
  assert eq (zero `mul` x) zero;
  assert zero `mul` x == zero;
  assert eq (x `mul` zero) zero;
  assert x `mul` zero == zero;
  ()

(* 0 times infinity is NOT zero. *)
[@@expect_failure [19]]
let test_mul_zero_inf_0 #t {| floating t |} (x : t{is_inf x}) =
  assert zero `mul` x == x

[@@expect_failure [19]]
let test_mul_zero_inf_1 #t {| floating t |} (x : t{is_inf x}) =
  assert eq (zero `mul` x) x

[@@expect_failure [19]]
let test_mul_zero_inf_2 #t {| floating t |} (x : t{is_inf x}) =
  assert x == zero `mul` x

[@@expect_failure [19]]
let test_mul_zero_inf_3 #t {| floating t |} (x : t{is_inf x}) =
  assert eq x (zero `mul` x)

(* zero times NaN is NOT zero *)
[@@expect_failure [19]]
let test_mul_zero_nan_0 #t {| floating t |} (x : t{is_nan x}) =
  assert zero `mul` x == zero

[@@expect_failure [19]]
let test_mul_zero_nan_1 #t {| floating t |} (x : t{is_nan x}) =
  assert eq (zero `mul` x) zero

[@@expect_failure [19]]
let test_mul_zero_nan_2 #t {| floating t |} (x : t{is_nan x}) =
  assert zero == zero `mul` x

[@@expect_failure [19]]
let test_mul_zero_nan_3 #t {| floating t |} (x : t{is_nan x}) =
  assert eq zero (zero `mul` x)



let test_mul_one_fin #t {| floating t |} (x : t{is_finite x \/ is_inf x}) =
  assert eq (one `mul` x) x;
  assert one `mul` x == x;
  assert eq (x `mul` one) x;
  assert x `mul` one == x;
  ()

(* one times NaN may not be the same NaN *)
[@@expect_failure [19]]
let test_mul_one_nan_0 #t {| floating t |} (x : t{is_nan x}) =
  assert one `mul` x == x

[@@expect_failure [19]]
let test_mul_one_nan_1 #t {| floating t |} (x : t{is_nan x}) =
  assert eq (one `mul` x) x

[@@expect_failure [19]]
let test_mul_one_nan_2 #t {| floating t |} (x : t{is_nan x}) =
  assert x == one `mul` x

[@@expect_failure [19]]
let test_mul_one_nan_3 #t {| floating t |} (x : t{is_nan x}) =
  assert eq x (one `mul` x)





(* x+0 is x for finite and infinite x, but not for NaN. *)
let test_add_zero_fin #t {| floating t |} (x : t{is_finite x \/ is_inf x}) =
  assert eq (zero `add` x) x;
  assert zero `add` x == x;
  assert eq (x `add` zero) x;
  assert x `add` zero == x;
  ()

// Adding zero to a NaN may not be the same NaN
[@@expect_failure [19]]
let test_add_zero_nan_0 #t {| floating t |} (x : t{is_nan x}) =
  assert zero `add` x == x

[@@expect_failure [19]]
let test_add_zero_nan_1 #t {| floating t |} (x : t{is_nan x}) =
  assert eq (zero `add` x) x

[@@expect_failure [19]]
let test_add_zero_nan_2 #t {| floating t |} (x : t{is_nan x}) =
  assert x == zero `add` x

[@@expect_failure [19]]
let test_add_zero_nan_3 #t {| floating t |} (x : t{is_nan x}) =
  assert eq x (zero `add` x)
