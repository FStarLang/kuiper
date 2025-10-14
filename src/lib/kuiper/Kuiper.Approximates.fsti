module Kuiper.Approximates

open Kuiper
open FStar.Real
open Kuiper.Scalars
open Kuiper.Seq.Common

class real_like (a:Type) {| scalar a |} = {
  to_real : a -> real;

  approximates : a -> real -> prop;

  to_real_ok :
    x:a ->
    Lemma (ensures x `approximates` to_real x)
          [SMTPat (x `approximates` to_real x)];

  (* We assume these two values exist and approximate correctly *)

  a0 : squash (approximates zero 0.0R);
  a1 : squash (approximates one 1.0R);

  a_add : x:a -> y:a -> r:real -> s:real ->
          Lemma (requires approximates x r /\ approximates y s)
                (ensures approximates (x `add` y) (r +. s))
                [SMTPat (approximates x r); SMTPat (approximates y s)];
                // ^ Does not kick in

  a_mul : x:a -> y:a -> r:real -> s:real ->
          Lemma (requires approximates x r /\ approximates y s)
                (ensures approximates (x `mul` y) (r *. s))
                [SMTPat (approximates x r); SMTPat (approximates y s)];
                // ^ Does not kick in
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

let seq_approximates (#a:Type) {| scalar a, real_like a |}
  (s : seq a) (r : seq Real.real) : prop
  = Seq.length s == Seq.length r /\
    (forall i. i < len s ==> (s @! i) `approximates` (r @! i))

let to_real_seq (#a:Type) {| scalar a, real_like a |}
  (s : seq a) : GTot (seq Real.real)
  = Seq.init_ghost (Seq.length s) (fun i -> to_real (s @! i))

let real_seq_sum (r : seq Real.real) : Real.real
  = seq_fold_left (+.) 0.0R r

let real_seq_sum_append
  (r1 r2 : seq Real.real)
  : Lemma (real_seq_sum (r1 `Seq.append` r2) == real_seq_sum r1 +. real_seq_sum r2)
  = admit()

let seq_approximates_append (#a:Type) {| scalar a, real_like a |}
  (s1 s2 : a) (r1 r2 : seq Real.real)
  : Lemma (requires s1 `approximates` real_seq_sum r1 /\ s2 `approximates` real_seq_sum r2)
          (ensures (s1 `add` s2) `approximates` real_seq_sum (r1 `Seq.append` r2))
  = a_add s1 s2 (real_seq_sum r1) (real_seq_sum r2);
    real_seq_sum_append r1 r2
