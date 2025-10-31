module Kuiper.Approximates.Class

open Kuiper
open Kuiper.Scalars

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
