module Kuiper.Float32.Base

(* All assumptions about Float32. *)

open Kuiper.Floating.Base

new
[@@FStar.Attributes.custard_float 32]
val t : Type0

val of_int       : Int64.t -> t

val zero : t
val one : t

val add : t -> t -> t
val mul : t -> t -> t

val lt : t -> t -> bool
val lte : t -> t -> bool
val ieee_eq : t -> t -> bool
unfold let eq (x y : t) : bool = ieee_eq x y

val sub : t -> t -> t
val div : t -> t -> t

val of_int_zero  : squash (of_int 0L == zero)
val of_int_one   : squash (of_int 1L == one)

(* Float literal from a string. Must be called with a concrete string;
   replaced during extraction with the corresponding C constant. *)
val of_literal : string -> t

val kind : t -> fkind

[@@FStar.Attributes.custard_extern "FLT_MAX";
   FStar.Attributes.custard_c_header "float.h"]
val largest : t

[@@FStar.Attributes.custard_extern "INFINITY";
   FStar.Attributes.custard_c_header "math.h"]
val infinity : t

val kind_one      : squash (kind one == Finite)
val kind_zero     : squash (kind zero == Finite)
val kind_largest  : squash (kind largest  == Finite)
val kind_infinity : squash (kind infinity == Infinite)

val eq_spec : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures eq x y <==> x == y)
          [SMTPat (eq x y)]

val lte_is_lt_or_eq : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures lte x y <==> lt x y \/ x == y)
          [SMTPat (lte x y)]

val neg_kind : (x : t) ->
    Lemma (ensures kind (zero `sub` x) == kind x)
          [SMTPat (zero `sub` x)]

val neg_neg : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures zero `sub` (zero `sub` x) == x)
          [SMTPat (zero `sub` (zero `sub` x))]

val lt_neg_flip : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures lt x y <==> lt (zero `sub` y) (zero `sub` x))
          [SMTPat (lt x y)]

val negate_lt_is_lte : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures lt x y <==> not (lte y x))
          [SMTPat (lt x y)]

val add_comm : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures add x y == add y x)
          [SMTPat (add x y)]

val mul_comm : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures mul x y == mul y x)
          [SMTPat (mul x y)]

val add_zero : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures add x zero == x)
          [SMTPat (add x zero)]

val  mul_zero : (x : t) ->
    Lemma (requires Finite? (kind x))
          (ensures mul x zero == zero)
          [SMTPat (mul x zero)]

val  mul_one : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures mul x one == x)
          [SMTPat (mul x one)]

val sub_is_add_neg : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures sub x y == add x (zero `sub` y))
          [SMTPat (sub x y)]

val largest_val_spec : (x : t) ->
    Lemma (requires Finite? (kind x))
          (ensures lte x largest)
          [SMTPat (lte x largest)]

val infinity_val_spec : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures lte x infinity)
          [SMTPat (lte x infinity)]

[@@FStar.Attributes.custard_extern "fmaxf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val fmax : t -> t -> t

val fmax_spec : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures fmax x y == (if lt x y then y else x))
          [SMTPat (fmax x y)]

[@@FStar.Attributes.custard_extern "expf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val fexp : t -> t
[@@FStar.Attributes.custard_extern "logf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val flog : t -> t
[@@FStar.Attributes.custard_extern "sqrtf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val sqrt : t -> t
[@@FStar.Attributes.custard_extern "rsqrtf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val rsqrt : t -> t
[@@FStar.Attributes.custard_extern "sinf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val sin : t -> t
[@@FStar.Attributes.custard_extern "cosf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val cos : t -> t
[@@FStar.Attributes.custard_extern "tanf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val tan : t -> t
[@@FStar.Attributes.custard_extern "asinf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val asin : t -> t
[@@FStar.Attributes.custard_extern "acosf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val acos : t -> t
[@@FStar.Attributes.custard_extern "atanf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val atan : t -> t
[@@FStar.Attributes.custard_extern "sinhf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val sinh : t -> t
[@@FStar.Attributes.custard_extern "coshf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val cosh : t -> t
[@@FStar.Attributes.custard_extern "tanhf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val tanh : t -> t
[@@FStar.Attributes.custard_extern "ceilf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val ceil : t -> t
[@@FStar.Attributes.custard_extern "floorf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val floor : t -> t
[@@FStar.Attributes.custard_extern "roundf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val round : t -> t
[@@FStar.Attributes.custard_extern "fabsf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val fabs : t -> t
[@@FStar.Attributes.custard_extern "erff";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val erf : t -> t
[@@FStar.Attributes.custard_extern "log2f";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val log2 : t -> t
[@@FStar.Attributes.custard_extern "log10f";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val log10 : t -> t
[@@FStar.Attributes.custard_extern "exp2f";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val exp2 : t -> t
[@@FStar.Attributes.custard_extern "powf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val pow : t -> t -> t
[@@FStar.Attributes.custard_extern "atan2f";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val atan2 : t -> t -> t
[@@FStar.Attributes.custard_extern "fminf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val fmin : t -> t -> t
[@@FStar.Attributes.custard_extern "fmodf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val fmod : t -> t -> t
[@@FStar.Attributes.custard_extern "copysignf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val copysign : t -> t -> t
[@@FStar.Attributes.custard_extern "fmaf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val fma : t -> t -> t -> t
