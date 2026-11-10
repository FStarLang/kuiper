module Kuiper.Approximates.Core

(* The model type [m] is determined by the concrete type [c]. *)
[@@FStar.Tactics.Typeclasses.fundeps [1]]
class can_approximate (c m : Type) = {
  approximates : c -> m -> prop;
}

unfold let (%~) #c #m {| d:can_approximate c m |} (x:c) (y:m)
  : prop = d.approximates x y

(* Exact approximation. This is not an instance, since it would match every
   concrete type; declare instances for the specific types that need it. *)
unfold let exact_can_approximate (a:Type) : can_approximate a a = {
  approximates = (fun x y -> x == y);
}

(* Non-floating inputs, such as literal strings and Int64 conversions. *)
instance int64_can_approximate : can_approximate FStar.Int64.t FStar.Int64.t =
  exact_can_approximate _

instance string_can_approximate : can_approximate string string =
  exact_can_approximate _

(* Function models may have refined domains, e.g. [div %~ r_div] where [r_div]
   only accepts a nonzero divisor. The domain predicates are implicit: they are
   inferred from the model when it is known, and are trivial otherwise. *)
unfold instance approx_function_can_approximate
  (dom1 dom2 cod1 cod2 : Type) (#p : dom2 -> prop)
  {| can_approximate dom1 dom2, can_approximate cod1 cod2 |}
  : can_approximate (dom1 -> cod1) (x:dom2{p x} -> cod2) = {
  approximates = (fun f g -> forall x (y:dom2{p y}).
    {:pattern (f x); (x %~ y)} x %~ y ==> f x %~ g y);
}

(* The unary instance can logically compose through function-valued codomains:
     approx_function_can_approximate a ar (b -> c) (br -> cr)
   works when those types are supplied explicitly. F*'s typeclass resolver
   currently fails to infer this application for a binary function (before SMT).
   Give it explicit binary and ternary instances to work around that limitation. *)
unfold instance approx_function2_can_approximate (a ar b br c cr:Type)
  (#pa : ar -> prop) (#pb : br -> prop)
  {| can_approximate a ar, can_approximate b br, can_approximate c cr |}
  : can_approximate (a -> b -> c) (x:ar{pa x} -> y:br{pb y} -> cr) = {
  approximates = (fun f g -> forall x y (r:ar{pa r}) (s:br{pb s}).
    {:pattern (f x y); (x %~ r); (y %~ s)}
    x %~ r /\ y %~ s ==> f x y %~ g r s);
}

unfold instance approx_function3_can_approximate (a ar b br c cr d dr:Type)
  (#pa : ar -> prop) (#pb : br -> prop) (#pc : cr -> prop)
  {| can_approximate a ar, can_approximate b br,
     can_approximate c cr, can_approximate d dr |}
  : can_approximate (a -> b -> c -> d)
                    (x:ar{pa x} -> y:br{pb y} -> z:cr{pc z} -> dr) = {
  approximates = (fun f g -> forall x y z (r:ar{pa r}) (s:br{pb s}) (t:cr{pc t}).
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
  (f:a -> b -> c) (g:ar -> br -> cr)
  (x:a) (y:b) (r:ar) (s:br)
  : Lemma (requires f %~ g /\ x %~ r /\ y %~ s)
          (ensures f x y %~ g r s) = ()

(* Variants for models with refined domains. The domain predicates are not
   inferred from unrefined models, so the helpers above do not take them. *)
let approx_apply_dom (#a #ar #b #br:Type) (#p:ar -> prop)
  {| can_approximate a ar, can_approximate b br |}
  (f:a -> b) (g:(r:ar{p r}) -> br) (x:a) (r:ar{p r})
  : Lemma (requires f %~ g /\ x %~ r)
          (ensures f x %~ g r) = ()

let approx_apply2_dom (#a #ar #b #br #c #cr:Type) (#pa:ar -> prop) (#pb:br -> prop)
  {| can_approximate a ar, can_approximate b br, can_approximate c cr |}
  (f:a -> b -> c) (g:(r:ar{pa r}) -> (s:br{pb s}) -> cr)
  (x:a) (y:b) (r:ar{pa r}) (s:br{pb s})
  : Lemma (requires f %~ g /\ x %~ r /\ y %~ s)
          (ensures f x y %~ g r s) = ()
