module Kuiper.Floating

include Kuiper.Floating.Base

(* Derived methods *)

inline_for_extraction noextract
let neg (#t:Type) {| floating t |} (x : t) : t =
  zero `sub` x

inline_for_extraction noextract
let gt (#t:Type) {| floating t |} (x y : t) : bool =
  lt y x

inline_for_extraction noextract
let gte (#t:Type) {| floating t |} (x y : t) : bool =
  lte y x

inline_for_extraction noextract
let neq (#t:Type) {| floating t |} (x y : t) : bool =
  not (eq x y)

inline_for_extraction noextract
let abs (#t:Type) {| floating t |} (x : t) : t =
  if x `gte` zero then x else sub zero x

inline_for_extraction noextract
let relu (#t:Type) {| floating t |} (x : t) : t =
  fmax x zero

(* We could provide executable versions for these if needed. *)
let is_nan (#t:Type) {| floating t |} (x : t) : GTot bool =
  NaN? (kind x)

let not_nan (#t:Type) {| floating t |} (x : t) : GTot bool =
  ~(is_nan x)

let is_inf (#t:Type) {| floating t |} (x : t) : GTot bool =
  Infinite? (kind x)

let is_finite (#t:Type) {| floating t |} (x : t) : GTot bool =
  Finite? (kind x)
