module Kuiper.Complex32.Base

(* All assumptions about single-precision complex (cuFloatComplex / float2).

   Mirrors the abstract treatment of Kuiper.Float32.Base: [t] is opaque, mapped
   by the extraction plugin to CUDA's [cuFloatComplex]; the arithmetic maps to
   the cuComplex intrinsics. The verification model is "a complex value is a pair
   of (public) Float32 components with the usual arithmetic" -- the [re]/[im]
   laws below.

   The complex operations are named c-prefixed (cadd, cmul, ...) so the laws can
   refer to the *component* (Float32) operations [add], [mul], [sub] from the
   scalar / floating typeclasses without a name clash. *)

open Kuiper.Scalars.Base   (* add, mul, zero, one, eq for the f32 component *)
open Kuiper.Floating.Base  (* sub, div for the f32 component *)
module F = Kuiper.Float32   (* the public f32 type and its floating instance *)

new
val t : Type0

(* Real / imaginary components and the constructor (make_cuFloatComplex). *)
val re : t -> F.t
val im : t -> F.t
val cmk : F.t -> F.t -> t

val czero : t
val cone  : t
val cci   : t   (* the imaginary unit i *)

val cadd : t -> t -> t
val cmul : t -> t -> t
val csub : t -> t -> t
val cdiv : t -> t -> t
val cconj : t -> t

val ceq  : t -> t -> bool
val clt  : t -> t -> bool   (* complex is unordered; always false *)
val clte : t -> t -> bool

(* ----------------------------------------------------------------------- *)
(* Model laws (right-hand sides use the f32 component operations).           *)
(* ----------------------------------------------------------------------- *)

val re_cmk : x:F.t -> y:F.t -> Lemma (re (cmk x y) == x) [SMTPat (re (cmk x y))]
val im_cmk : x:F.t -> y:F.t -> Lemma (im (cmk x y) == y) [SMTPat (im (cmk x y))]
val cmk_eta : z:t -> Lemma (cmk (re z) (im z) == z) [SMTPat (cmk (re z) (im z))]

val re_czero : squash (re czero == zero)
val im_czero : squash (im czero == zero)
val re_cone  : squash (re cone  == one)
val im_cone  : squash (im cone  == zero)
val re_cci   : squash (re cci == zero)
val im_cci   : squash (im cci == one)

val re_cadd : a:t -> b:t -> Lemma (re (cadd a b) == add (re a) (re b)) [SMTPat (re (cadd a b))]
val im_cadd : a:t -> b:t -> Lemma (im (cadd a b) == add (im a) (im b)) [SMTPat (im (cadd a b))]
val re_csub : a:t -> b:t -> Lemma (re (csub a b) == sub (re a) (re b)) [SMTPat (re (csub a b))]
val im_csub : a:t -> b:t -> Lemma (im (csub a b) == sub (im a) (im b)) [SMTPat (im (csub a b))]

(* (a+bi)(c+di) = (ac - bd) + (ad + bc) i *)
val re_cmul : a:t -> b:t ->
  Lemma (re (cmul a b) == sub (mul (re a) (re b)) (mul (im a) (im b))) [SMTPat (re (cmul a b))]
val im_cmul : a:t -> b:t ->
  Lemma (im (cmul a b) == add (mul (re a) (im b)) (mul (im a) (re b))) [SMTPat (im (cmul a b))]

val re_cconj : z:t -> Lemma (re (cconj z) == re z) [SMTPat (re (cconj z))]
val im_cconj : z:t -> Lemma (im (cconj z) == sub zero (im z)) [SMTPat (im (cconj z))]

val ceq_spec : a:t -> b:t ->
  Lemma (ensures ceq a b <==> (re a == re b /\ im a == im b)) [SMTPat (ceq a b)]

val clt_false  : a:t -> b:t -> Lemma (clt  a b == false) [SMTPat (clt a b)]
val clte_false : a:t -> b:t -> Lemma (clte a b == false) [SMTPat (clte a b)]
