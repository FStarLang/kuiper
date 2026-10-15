module Kuiper.Test.Approximates

(* Exercise the public re-exports without opening Scalar or Core directly. *)
open Kuiper.Approximates
open Kuiper.Real
open Kuiper.Scalars

let public_relation (#a #b:Type) {| Kuiper.Approximates.can_approximate a b |}
  (x:a) (y:b)
  : Lemma (requires Kuiper.Approximates.approximates x y)
          (ensures x %~ y) = ()

(* Function contracts must work with the underlying scalar relation, including
   when that relation is written directly rather than through %~. *)
let unary #a {| scalar a, Kuiper.Approximates.real_like a |}
  (f:a -> a) (g:real -> real) (x:a) (r:real)
  : Lemma (requires f %~ g /\ v_approximates x r)
          (ensures v_approximates (f x) (g r)) = ()

let binary #a {| scalar a, real_like a |}
  (f:a -> a -> a) (g:real -> real -> real) (x y:a) (r s:real)
  : Lemma (requires f %~ g /\ x %~ r /\ y %~ s)
          (ensures f x y %~ g r s) = ()

let ternary #a {| scalar a, real_like a |}
  (f:a -> a -> a -> a) (g:real -> real -> real -> real)
  (x y z:a) (r s t:real)
  : Lemma (requires f %~ g /\ x %~ r /\ y %~ s /\ z %~ t)
          (ensures f x y z %~ g r s t) = ()

unfold let positive_square (x:real{x >. 0.0R}) : real = x *. x
unfold let quotient (x:real) (y:real{y =!= 0.0R}) : real = x /. y

let refined_domain #a {| scalar a, real_like a |}
  (f:a -> a) (x:a) (r:real{r >. 0.0R})
  : Lemma (requires f %~ positive_square /\ x %~ r)
          (ensures f x %~ (r *. r)) = ()

let refined_second_domain #a {| scalar a, real_like a |}
  (f:a -> a -> a) (x y:a) (r:real) (s:real{s =!= 0.0R})
  : Lemma (requires f %~ quotient /\ x %~ r /\ y %~ s)
          (ensures f x y %~ (r /. s)) =
  approx_apply2 #a #real #a #(s:real{s =!= 0.0R}) #a #real
    f quotient x y r s

let exact_input #a {| scalar a, real_like a |}
  (f:string -> a) (g:string -> real) (s:string)
  : Lemma (requires f %~ g) (ensures f s %~ g s) =
  approx_apply f g s s

(* A contract over a refined real domain says nothing outside that domain. *)
[@@expect_failure [19]]
let outside_domain #a {| scalar a, real_like a |}
  (f:a -> a) (x:a)
  : Lemma (requires f %~ positive_square /\ x %~ (0.0R -. 1.0R))
          (ensures f x %~ 1.0R) = ()
