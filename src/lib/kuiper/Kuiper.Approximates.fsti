module Kuiper.Approximates

open Kuiper
open FStar.Real
open Kuiper.Scalars

class real_like (a:Type) {| scalar a |} = {
  approximates : a -> real -> prop;

  a0 : squash (approximates zero 0.0R);
  a1 : squash (approximates one 1.0R);

  a_add : x:a -> y:a -> r:real -> s:real ->
          Lemma (requires approximates x r /\ approximates y s)
                (ensures approximates (x `add` y) (r +. s))
                [SMTPat (approximates x r); SMTPat (approximates y s)];

  a_mul : x:a -> y:a -> r:real -> s:real ->
          Lemma (requires approximates x r /\ approximates y s)
                (ensures approximates (x `mul` y) (r *. s))
                [SMTPat (approximates x r); SMTPat (approximates y s)];
}

(* We assume these types approximate reals *)
instance val real_like_f16 : real_like f16
instance val real_like_f32 : real_like f32
instance val real_like_f64 : real_like f64

(* For the integer types, we can actually define the relation
and prove it. BUT, we must consider overflow! So we make the relation
weaker than you may expect. *)
instance val real_like_u8 : real_like u8
instance val real_like_u16 : real_like u16
instance val real_like_u32 : real_like u32
instance val real_like_u64 : real_like u64
