module Kuiper.Float32.Exact

(* Bit-preserving binary32, deliberately independent of Float32.Base.t:
   the generic floating model identifies signed zeros. Here propositional
   equality must imply identical bits, even for zeros and NaNs.

   Trusted boundary: these declarations describe CUDA operations, not an
   implementation of IEEE arithmetic in F*. Fix the CUDA toolkit, target,
   and flags when interpreting the *_bits functions. In particular exp_bits
   denotes expf (NOT __expf), including its actual rounding and NaN result.
   Use matching compilation flags for both reference and extracted code. *)

new val t : Type0

(* Specification-only observation; erased from extracted code. *)
val to_bits : t -> GTot FStar.UInt32.t
val of_bits (b : FStar.UInt32.t) : Tot (x : t{to_bits x == b})

val one : x:t{to_bits x == 0x3f800000ul}

(* CUDA unary negation, not 0.0f - x. Keep this abstract as well:
   PTX permits an unspecified NaN result, and FTZ affects subnormals. *)
val neg_bits : FStar.UInt32.t -> GTot FStar.UInt32.t

val exp_bits : FStar.UInt32.t -> GTot FStar.UInt32.t
val add_rn_bits : FStar.UInt32.t -> FStar.UInt32.t -> GTot FStar.UInt32.t
val div_rn_bits : FStar.UInt32.t -> FStar.UInt32.t -> GTot FStar.UInt32.t

val neg (x : t) : Tot (r:t{to_bits r == neg_bits (to_bits x)})
val exp (x : t) : Tot (r:t{to_bits r == exp_bits (to_bits x)})
val add_rn (x y : t) : Tot (r:t{to_bits r == add_rn_bits (to_bits x) (to_bits y)})
val div_rn (x y : t) : Tot (r:t{to_bits r == div_rn_bits (to_bits x) (to_bits y)})

inline_for_extraction let () = ()

(* Qualitative contracts, also part of the trusted primitive boundary.
   On nonnegative binary32 encodings, unsigned bit order is numeric order:
   0 .. 0x3f800000 describes [0,1], and 0x7f800000 is +infinity.
   The add/div laws require a positive normal left operand (in sigmoid it
   is one), so these laws hold with gradual underflow and with FTZ. They
   do not assume correctly rounded expf or a quantitative error bound. *)
let is_nan_bits (x : FStar.UInt32.t) : prop =
  FStar.UInt32.v (FStar.UInt32.logand x 0x7ffffffful) > 0x7f800000

val neg_nan (x : FStar.UInt32.t) : Lemma
  (ensures (is_nan_bits (neg_bits x) <==> is_nan_bits x))

val exp_nonnegative (x : FStar.UInt32.t) : Lemma
  (requires not (is_nan_bits x))
  (ensures FStar.UInt32.v (exp_bits x) <= 0x7f800000)

val add_rn_nonnegative (x y : FStar.UInt32.t) : Lemma
  (requires 0x00800000 <= FStar.UInt32.v x /\ FStar.UInt32.v x < 0x7f800000 /\
            FStar.UInt32.v y <= 0x7f800000)
  (ensures FStar.UInt32.v x <= FStar.UInt32.v (add_rn_bits x y) /\
           FStar.UInt32.v (add_rn_bits x y) <= 0x7f800000)

val div_rn_unit_interval (x y : FStar.UInt32.t) : Lemma
  (requires 0x00800000 <= FStar.UInt32.v x /\ FStar.UInt32.v x < 0x7f800000 /\
            FStar.UInt32.v x <= FStar.UInt32.v y /\ FStar.UInt32.v y <= 0x7f800000)
  (ensures FStar.UInt32.v (div_rn_bits x y) <= 0x3f800000)

(* Runtime bit observation, with the same bit-preserving meaning as the
   ghost observation above. CUDA __float_as_uint is a bit cast. *)
val bits (x : t) : Tot (r:FStar.UInt32.t{r == to_bits x})

(* Keep fast CUDA intrinsics separate from their standard/RN counterparts.
   All models retain the fixed-toolkit/target/flags interpretation above. *)
val mul_rn_bits : FStar.UInt32.t -> FStar.UInt32.t -> GTot FStar.UInt32.t
val fma_bits : FStar.UInt32.t -> FStar.UInt32.t -> FStar.UInt32.t -> GTot FStar.UInt32.t
val exp_fast_bits : FStar.UInt32.t -> GTot FStar.UInt32.t
val div_fast_bits : FStar.UInt32.t -> FStar.UInt32.t -> GTot FStar.UInt32.t
val le_bits : FStar.UInt32.t -> FStar.UInt32.t -> GTot bool

val mul_rn (x y : t) : Tot (r:t{to_bits r == mul_rn_bits (to_bits x) (to_bits y)})
val fma (x y z : t) : Tot (r:t{to_bits r == fma_bits (to_bits x) (to_bits y) (to_bits z)})
val exp_fast (x : t) : Tot (r:t{to_bits r == exp_fast_bits (to_bits x)})
val div_fast (x y : t) : Tot (r:t{to_bits r == div_fast_bits (to_bits x) (to_bits y)})
val le (x y : t) : Tot (r:bool{r == le_bits (to_bits x) (to_bits y)})
val ge (x y : t) : Tot (r:bool{r == le_bits (to_bits y) (to_bits x)})
