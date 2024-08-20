module GPU.IsReduction

open FStar.List.Tot

let assoc (#a:Type) (f : a -> a -> a) : prop =
  forall x y z. f (f x y) z == f x (f y z)

type is_reduction (#a:Type0) (f : a -> a -> a) : (xs : list a) -> (r : a) -> Type0 =
  | Singl :
    r:a ->
    is_reduction f [r] r
  | Split :
    s1:list a -> s2:list a -> r1:a -> r2:a ->
    is_reduction f s1 r1 -> is_reduction f s2 r2 ->
    is_reduction f (s1 @ s2) (f r1 r2)
  
let assoc_uniq_reduction 
  (#a:Type) (f : a -> a -> a) (xs : list a) (r1 r2 : a)
  (pf1 : is_reduction f xs r1) (pf2 : is_reduction f xs r2)
: Lemma (requires assoc f)
        (ensures r1 == r2)
= admit()
