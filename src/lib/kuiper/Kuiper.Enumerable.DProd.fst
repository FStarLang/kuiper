module Kuiper.Enumerable.DProd

// NOTE: There a couple of admits here, but turns we no longer need
// the enumerability of dependent products, so deleting this file
// (or at least not shipping it) is totally fine.

open Kuiper.Enumerable
open Kuiper.Bijection
open Kuiper.Common

let dprod_cardinal
  (t1 : Type) (t2 : t1 -> Type)
  {| d1 : enumerable t1 |} {| d2 : (x:t1 -> enumerable (t2 x)) |}
  : GTot nat
  = sum_n #(cardinal t1 #_)
        (fun x -> let y : t1 = of_nat x in
                  cardinal (t2 y) #(d2 y))

let rec dprod_cardinal_split_aux
  (t1 : Type) (t2 : t1 -> Type)
  {| d1 : enumerable t1 |} {| d2 : (x:t1 -> enumerable (t2 x)) |}
  (b : natlt (cardinal t1 #_))
  (x : natlt (sum_n #(cardinal t1 #_ - b)
              (fun i -> let y : t1 = of_nat (b + i) in
                        cardinal (t2 y) #(d2 y))))
  : GTot (i : natlt (cardinal t1 #_) & natlt (cardinal (t2 (of_nat i)) #(d2 _)))
         (decreases cardinal t1 #_ - b)
=
  let f : nat = cardinal (t2 (of_nat b)) #(d2 _) in
  if x < f then
    (| b, x |)
  else (
    let s1 = (sum_n #(cardinal t1 #_ - b)
                  (fun i -> let y : t1 = of_nat (b + i) in
                            cardinal (t2 y) #(d2 y))) in
    let s2 = (sum_n #(cardinal t1 #_ - (b + 1))
                  (fun i -> let y : t1 = of_nat (b + 1 + i) in
                            cardinal (t2 y) #(d2 y))) in
    assume (s1 == f + s2); // fixme... but clearly ok
    dprod_cardinal_split_aux t1 t2 (b + 1) (x - f)
  )

let dprod_cardinal_split
  (t1 : Type) (t2 : t1 -> Type)
  {| d1 : enumerable t1 |} {| d2 : (x:t1 -> enumerable (t2 x)) |}
  (x : natlt (dprod_cardinal t1 t2 #_ #_))
  : GTot (i : natlt (cardinal t1 #_) & natlt (cardinal (t2 (of_nat i)) #_))
=  // again also clearly true
   assume (dprod_cardinal t1 t2 #_ #_ ==
            sum_n #(cardinal t1 #_ - 0)
              (fun i -> let y : t1 = of_nat (0 + i) in
                        cardinal (t2 y) #(d2 y)));
   dprod_cardinal_split_aux t1 t2 #d1 #d2 0 x

let dprod_cardinal_unsplit
  (t1 : Type) (t2 : t1 -> Type)
  {| d1 : enumerable t1 |} {| d2 : (x:t1 -> enumerable (t2 x)) |}
  (x : natlt (cardinal t1 #_))
  (y : natlt (cardinal (t2 (of_nat x)) #(d2 _)))
  : GTot (natlt (dprod_cardinal t1 t2 #_ #_))
=
  let offset =
    sum_n #x
      (fun k -> let z : t1 = of_nat k in
                cardinal (t2 z) #(d2 z)) in
  let r = offset + y in
  assume (r < dprod_cardinal t1 t2 #_ #_);
  r

let lemma_dprod_cardinal_unsplit_split
  (t1 : Type) (t2 : t1 -> Type)
  {| d1 : enumerable t1 |} {| d2 : (x:t1 -> enumerable (t2 x)) |}
  (x : natlt (dprod_cardinal t1 t2 #_ #_))
  : Lemma (let (| i, j |) = dprod_cardinal_split t1 t2 #d1 #d2 x in
           dprod_cardinal_unsplit t1 t2 #d1 #d2 i j == x)
  = admit()

let lemma_dprod_cardinal_split_unsplit
  (t1 : Type) (t2 : t1 -> Type)
  {| d1 : enumerable t1 |} {| d2 : (x:t1 -> enumerable (t2 x)) |}
  (i : natlt (cardinal t1 #_))
  (j : natlt (cardinal (t2 (of_nat i)) #(d2 _)))
  : Lemma (dprod_cardinal_split t1 t2 (dprod_cardinal_unsplit t1 t2 #d1 #d2 i j) == (| i, j |))
  = admit()

let dprod_bij_ff
  (t1 : Type) (t2 : t1 -> Type)
  {| d1 : enumerable t1 |} {| d2 : (x:t1 -> enumerable (t2 x)) |}
  (xy : (x:t1 & t2 x))
  : GTot (natlt (dprod_cardinal t1 t2 #_ #_))
=
  let (| x, y |) = xy in
  dprod_cardinal_unsplit t1 t2 #d1 #d2 (to_nat x) (to_nat #_ #(d2 x) y)

let dprod_bij_gg
  (t1 : Type) (t2 : t1 -> Type)
  {| d1 : enumerable t1 |} {| d2 : (x:t1 -> enumerable (t2 x)) |}
  (n : natlt (dprod_cardinal t1 t2 #_ #_))
  : GTot (x:t1 & t2 x)
=
  let (| i, j |) = dprod_cardinal_split t1 t2 #d1 #d2 n in
  let x = of_nat i in
  let y = of_nat #_ #(d2 x) j in
  (| x, y |)

let dprod_bij
  (t1 : Type) (t2 : t1 -> Type)
  {| d1 : enumerable t1 |} {| d2 : (x:t1 -> enumerable (t2 x)) |}
  : ((x:t1 & t2 x) =~ natlt (dprod_cardinal t1 t2 #_ #_))
= {
  ff = dprod_bij_ff t1 t2 #d1 #d2;
  gg = dprod_bij_gg t1 t2 #d1 #d2;
  ff_gg = (fun n ->
    let (| i, j |) = dprod_cardinal_split t1 t2 #d1 #d2 n in
    calc (==) {
      dprod_bij_ff t1 t2 #d1 #d2 (dprod_bij_gg t1 t2 #d1 #d2 n);
      == {}
      dprod_bij_ff t1 t2 #d1 #d2 (| of_nat i, of_nat #_ #(d2 (of_nat i)) j |);
      == {}
      dprod_cardinal_unsplit t1 t2 #d1 #d2 (to_nat (of_nat i)) (to_nat (of_nat #_ #(d2 (of_nat i)) j));
      == { d1.bij.ff_gg i }
      dprod_cardinal_unsplit t1 t2 #d1 #d2 i (to_nat (of_nat #_ #(d2 (of_nat i)) j));
      == { (d2 (of_nat i)).bij.ff_gg j }
      dprod_cardinal_unsplit t1 t2 #d1 #d2 i j;
      == { lemma_dprod_cardinal_unsplit_split t1 t2 #d1 #d2 n }
      n;
    };
    d1.bij.ff_gg i;
    (d2 (of_nat i)).bij.ff_gg j;
    ());
  gg_ff = (fun (xy : (x:t1 & t2 x)) ->
    let (| x, y |) = xy in
    let i = to_nat x in
    let j = to_nat #_ #(d2 x) y in
    let n = dprod_cardinal_unsplit t1 t2 #d1 #d2 i j in
    calc (==) {
      dprod_bij_gg t1 t2 #d1 #d2 (dprod_bij_ff t1 t2 #d1 #d2 xy);
      == {}
      dprod_bij_gg t1 t2 #d1 #d2 (dprod_cardinal_unsplit t1 t2 #d1 #d2 i j);
      == { lemma_dprod_cardinal_split_unsplit t1 t2 #d1 #d2 i j }
      (| of_nat i, of_nat #_ #(d2 (of_nat i)) j |);
      == { d1.bij.gg_ff x; (d2 (of_nat i)).bij.gg_ff y }
      (| x, y |);
    }
  );
}

instance enumerable_dprod (t1 : Type) (t2 : t1 -> Type)
  {| d1 : enumerable t1 |} {| d2 : (x:t1 -> enumerable (t2 x)) |}
  : enumerable (x:t1 & t2 x)
= {
  _cardinal = dprod_cardinal t1 t2 #_ #_;
  bij       = dprod_bij t1 t2 #_ #_;
}
