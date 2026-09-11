module Kuiper.Float16.Base

(* All assumptions about Float16. *)

open Kuiper.Floating.Base

new
[@@FStar.Attributes.custard_float 16]
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

[@@FStar.Attributes.custard_extern "HLF_MAX";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val largest : t
[@@FStar.Attributes.custard_extern "HLF_INFINITY";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
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

[@@FStar.Attributes.custard_extern "kpr_hfmax";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val fmax : t -> t -> t

val fmax_spec : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures fmax x y == (if lt x y then y else x))
          [SMTPat (fmax x y)]

[@@FStar.Attributes.custard_extern "hexp";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val fexp : t -> t
[@@FStar.Attributes.custard_extern "hlog";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val flog : t -> t
[@@FStar.Attributes.custard_extern "kpr_hsqrt";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val sqrt : t -> t
[@@FStar.Attributes.custard_extern "kpr_hrsqrt";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val rsqrt : t -> t
[@@FStar.Attributes.custard_extern "kpr_hsin";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val sin : t -> t
[@@FStar.Attributes.custard_extern "kpr_hcos";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val cos : t -> t
[@@FStar.Attributes.custard_extern "kpr_htan";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val tan : t -> t
[@@FStar.Attributes.custard_extern "kpr_hasin";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val asin : t -> t
[@@FStar.Attributes.custard_extern "kpr_hacos";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val acos : t -> t
[@@FStar.Attributes.custard_extern "kpr_hatan";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val atan : t -> t
[@@FStar.Attributes.custard_extern "kpr_hsinh";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val sinh : t -> t
[@@FStar.Attributes.custard_extern "kpr_hcosh";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val cosh : t -> t
[@@FStar.Attributes.custard_extern "kpr_htanh";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val tanh : t -> t
[@@FStar.Attributes.custard_extern "kpr_hceil";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val ceil : t -> t
[@@FStar.Attributes.custard_extern "kpr_hfloor";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val floor : t -> t
[@@FStar.Attributes.custard_extern "kpr_hround";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val round : t -> t
[@@FStar.Attributes.custard_extern "kpr_hfabs";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val fabs : t -> t
[@@FStar.Attributes.custard_extern "kpr_herf";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val erf : t -> t
[@@FStar.Attributes.custard_extern "kpr_hlog2";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val log2 : t -> t
[@@FStar.Attributes.custard_extern "kpr_hlog10";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val log10 : t -> t
[@@FStar.Attributes.custard_extern "kpr_hexp2";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val exp2 : t -> t
[@@FStar.Attributes.custard_extern "kpr_hpow";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val pow : t -> t -> t
[@@FStar.Attributes.custard_extern "kpr_hatan2";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val atan2 : t -> t -> t
[@@FStar.Attributes.custard_extern "kpr_hfmin";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val fmin : t -> t -> t
[@@FStar.Attributes.custard_extern "kpr_hfmod";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val fmod : t -> t -> t
[@@FStar.Attributes.custard_extern "kpr_hcopysign";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val copysign : t -> t -> t
[@@FStar.Attributes.custard_extern "kpr_hfma";
   FStar.Attributes.custard_c_header "kuiper/math.h"]
val fma : t -> t -> t -> t
