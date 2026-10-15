module Kuiper.Approximates.Core

[@@FStar.Tactics.Typeclasses.fundeps [1]]
class can_approximate (c m : Type) = {
  approximates : c -> m -> prop;
}

unfold let (%~) #c #m (x:c) (y:m) {| d:can_approximate c m |}
  : prop = d.approximates x y

(* Non-floating inputs, such as literal strings and Int64 conversions. *)
instance exact_can_approximate (a:Type) : can_approximate a a = {
  approximates = (fun x y -> x == y);
}

unfold instance approx_function_can_approximate
  (dom1 dom2 cod1 cod2 : Type)
  {| can_approximate dom1 dom2, can_approximate cod1 cod2 |}
  : can_approximate (dom1 -> cod1) (dom2 -> cod2) = {
  approximates = (fun f g -> forall x y.
    {:pattern (f x); (x %~ y)} x %~ y ==> f x %~ g y);
}

(* The unary instance can logically compose through function-valued codomains:
     approx_function_can_approximate a ar (b -> c) (br -> cr)
   works when those types are supplied explicitly. F*'s typeclass resolver
   currently fails to infer this application for a binary function (before SMT).
   Give it explicit binary and ternary instances to work around that limitation. *)
unfold instance approx_function2_can_approximate (a ar b br c cr:Type)
  {| can_approximate a ar, can_approximate b br, can_approximate c cr |}
  : can_approximate (a -> b -> c) (ar -> br -> cr) = {
  approximates = (fun f g -> forall x y r s.
    {:pattern (f x y); (x %~ r); (y %~ s)}
    x %~ r /\ y %~ s ==> f x y %~ g r s);
}

unfold instance approx_function3_can_approximate (a ar b br c cr d dr:Type)
  {| can_approximate a ar, can_approximate b br,
     can_approximate c cr, can_approximate d dr |}
  : can_approximate (a -> b -> c -> d) (ar -> br -> cr -> dr) = {
  approximates = (fun f g -> forall x y z r s t.
    {:pattern (f x y z); (x %~ r); (y %~ s); (z %~ t)}
    x %~ r /\ y %~ s /\ z %~ t ==> f x y z %~ g r s t);
}

(* Explicit instantiation helpers for proofs where SMT matching is expensive. *)
let approx_apply (#a #ar #b #br:Type)
  {| can_approximate a ar, can_approximate b br |}
  (f:a -> b) (g:ar -> br) (x:a) (r:ar)
  : Lemma (requires f %~ g /\ x %~ r)
          (ensures f x %~ g r) = ()

let approx_apply2 (#a #ar #b #br #c #cr:Type)
  {| can_approximate a ar, can_approximate b br, can_approximate c cr |}
  (f:a -> b -> c) (g:ar -> br -> cr) (x:a) (y:b) (r:ar) (s:br)
  : Lemma (requires f %~ g /\ x %~ r /\ y %~ s)
          (ensures f x y %~ g r s) = ()
