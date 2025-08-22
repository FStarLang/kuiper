module Kuiper.ForEvery

#lang-pulse
open Kuiper.Common
open Kuiper.Bijection
open Kuiper.Enumerable
open Pulse.Lib.BigStar
open Pulse.Lib.Trade

let ( forall+ ) #a #d p =
  bigstar 0 (cardinal a #_) (fun i -> p (of_nat i))

let forevery_ext_lem
  (#a:Type0) {| enumerable a |}
  (f : a -> slprop)
  (g : a -> slprop { forall x. f x == g x })
  : Lemma (ensures (forall+ (x:a). f x) == (forall+ (x:a). g x))
  = ()

ghost
fn forevery_ext
  (#a:Type0) {| enumerable a |}
  (f : a -> slprop)
  (g : a -> slprop { forall x. f x == g x })
  requires
    forall+ (x:a). f x
  ensures
    forall+ (x:a). g x
{
  ();
}

ghost
fn forevery_ext_2
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (f : a -> b -> slprop)
  (g : a -> b -> slprop)
  requires
    pure (forall x y. f x y == g x y)
  requires
    forall+ (x:a) (y:b). f x y
  ensures
    forall+ (x:a) (y:b). g x y
{
  ();
}

ghost
fn forevery_flatten
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (f : a -> b -> slprop)
  requires
    forevery a (fun x ->
      forevery b (fun y -> f x y))
  ensures
    forevery (a & b) (fun (x, y) -> f x y)
{
  unfold op_forall_Plus #a (fun x -> forevery b (fun y -> f x y));
  ghost
  fn aux1 (i:natlt (cardinal a #_))
    requires forevery b (fun y -> f (of_nat i) y)
    ensures  bigstar 0 (cardinal b #_) (fun j -> f (of_nat i) (of_nat j))
  {
    unfold op_forall_Plus #b (fun y -> f (of_nat i) y);
  };
  bigstar_map #_ #_ #0 #(cardinal a #_) aux1; // optional :-)
  bigstar_flatten #_ #_ #(cardinal a #_) #(cardinal b #_);
  fold op_forall_Plus #(a & b) (fun (x, y) -> f x y);
}

ghost
fn forevery_flatten'
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (f : a & b -> slprop)
  requires
    forall+ (x:a) (y:b). f (x, y)
  ensures
    forall+ (xy : a & b). f xy
{
  forevery_flatten (fun x y -> f (x, y));
}

ghost
fn forevery_unflatten
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (f : a -> b -> slprop)
  requires
    forevery (a & b) (fun (x, y) -> f x y)
  ensures
    forevery a (fun x ->
      forevery b (fun y -> f x y))
{
  unfold op_forall_Plus #(a & b) (fun (x, y) -> f x y);
  assert bigstar 0 (cardinal (a & b) #_) (fun i -> let x, y = of_nat i in f x y);
  rewrite
    bigstar 0 (cardinal (a & b) #_) (fun i -> let x, y = of_nat i in f x y)
  as
    bigstar 0 (cardinal a #_ * cardinal b #_) (fun i -> f (of_nat (i / cardinal b #_)) (of_nat (i % cardinal b #_)));
  bigstar_unflatten #0 #0 #(cardinal a #_) #(cardinal b #_) #(fun x y -> f (of_nat x) (of_nat y));
  fold op_forall_Plus #a (fun x ->
    forevery b (fun y -> f x y));
}

ghost
fn forevery_unflatten'
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (f : a & b -> slprop)
  requires
    forall+ (xy : a & b). f xy
  ensures
    forall+ (x:a) (y:b). f (x, y)
{
  forevery_unflatten (fun x y -> f (x, y));
}

let bij2perm (n:nat) (d : natlt n =~ natlt n)
  : permutation (i:nat {0 <= i /\ i < n}) = {
  f = d.ff;
  g = d.gg;
  proof = ez; (* patterns! *)
}

ghost
fn bigstar_permute'
  (#u1 : int)
  (#n : nat)
  (f : natlt n -> slprop)
  (d : bijection (natlt n) (natlt n))
  requires bigstar #u1 0 n f
  ensures  bigstar #u1 0 n (fun i -> f (d.ff i))
{
  let pp = bij2perm n d;
  bigstar_permute #u1 #0 #n #f pp;
}

ghost
fn bigstar_permute''
  (#u1 : int)
  (#n : nat)
  (f : natlt n -> slprop)
  (d : bijection (natlt n) (natlt n))
  requires bigstar #u1 0 n f
  ensures  bigstar #u1 0 n (fun i -> f (d.gg i))
{
  bigstar_permute' #u1 #n f (bij_sym d);
}

ghost
fn forevery_iso
  (#a:Type0) {| ea : enumerable a |}
  (#b:Type0) {| eb : enumerable b |}
  (bij : erased (a =~ b))
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    forall+ (y:b). p (bij.gg y)
{
  bijection_implies_equal_cardinal a b bij;
  assert (pure (cardinal a #_ == cardinal b #_));

  unfold op_forall_Plus #a (fun x -> p x);
  assert bigstar 0 (cardinal a #_) (fun i -> p (of_nat i));

  let bij_n : (natlt (cardinal a #_) =~ natlt (cardinal a #_)) =
    bij_sym ea.bij `bij_comp` bij `bij_comp` eb.bij;

  assert bigstar 0 (cardinal a #_) (fun i -> p (of_nat #a i));
  bigstar_permute'' (fun i -> p (of_nat i)) bij_n;
  assert bigstar 0 (cardinal a #_) (fun i -> p (of_nat #a (bij_n.gg i)));
  assert bigstar 0 (cardinal a #_) (fun i -> p (of_nat #a (to_nat #a (bij.gg (eb.bij.gg i)))));
  assert bigstar 0 (cardinal a #_) (fun i -> p (bij.gg (eb.bij.gg i)));
  rewrite (* rewrite each cardinal a as cardinal b fails *)
    bigstar 0 (cardinal a #_) (fun i -> p (bij.gg (eb.bij.gg i)))
  as
    bigstar 0 (cardinal b #_) (fun i -> p (bij.gg (eb.bij.gg i)));

  assert bigstar 0 (cardinal b #_) (fun i -> p (bij.gg (eb.bij.gg i)));
  fold op_forall_Plus #b (fun y -> p (bij.gg y));
}

ghost
fn forevery_iso_back
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (bij : erased (a =~ b))
  (p : a -> slprop)
  requires
    forall+ (y:b). p (bij.gg y)
  ensures
    forall+ (x:a). p x
{
  forevery_iso (bij_sym bij) _;
}

ghost
fn forevery_permute
  (#a:Type0) {| ea: enumerable a |}
  (bij : erased (a =~ a))
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    forall+ (x:a). p (bij.ff x)
{
  unfold op_forall_Plus #a (fun x -> p x);
  bigstar_permute'
    (fun i -> p (of_nat i))
    (bij_sym ea.bij `bij_comp` bij `bij_comp` ea.bij);
  fold op_forall_Plus #a (fun x -> p (of_nat (to_nat (bij.ff x))));
}
ghost
fn forevery_permute_back
  (#a:Type0) {| ea: enumerable a |}
  (bij : erased (a =~ a))
  (p : a -> slprop)
  requires
    forall+ (x:a). p (bij.ff x)
  ensures
    forall+ (x:a). p x
{
  forevery_permute (bij_sym bij) _;
}

ghost
fn forevery_tostar
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    bigstar 0 (cardinal a #_) (fun i -> p (of_nat i))
{
  unfold op_forall_Plus #a (fun x -> p x);
}

ghost
fn forevery_fromstar
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop)
  requires
    bigstar 0 (cardinal a #_) (fun i -> p (of_nat i))
  ensures
    forall+ (x:a). p x
{
  fold op_forall_Plus #a (fun x -> p x);
}

ghost
fn forevery_fromnat
  (n : nat)
  (p : natlt n -> slprop)
  requires
    bigstar 0 n (fun i -> p i)
  ensures
    forall+ (x : natlt n). p x
{
  rewrite each n as cardinal (natlt n) #_;
  forevery_fromstar p;
}

ghost
fn forevery_tonat
  (n : nat)
  (p : natlt n -> slprop)
  requires
    forall+ (x : natlt n). p x
  ensures
    bigstar 0 n (fun i -> p i)
{
  forevery_tostar p;
  rewrite each cardinal (natlt n) #_ as n;
}

ghost
fn forevery_emp_intro
  (a : Type0) {| enumerable a |}
  requires
    emp
  ensures
    forall+ (_ : a). emp
{
  bigstar_emp_intro 0 (cardinal a #_);
  fold op_forall_Plus #a (fun _ -> emp);
}

ghost
fn forevery_emp_elim
  (a : Type0) {| enumerable a |}
  requires
    forall+ (_ : a). emp
  ensures
    emp
{
  unfold op_forall_Plus #a (fun _ -> emp);
  bigstar_emp_elim #_ #0 #(cardinal a #_);
}

ghost
fn forevery_singleton_intro
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop { cardinal a #_ == 1 })
  requires
    p (of_nat 0)
  ensures
    forall+ (x:a). p x
{
  bigstar_single_intro #0 0 (fun x -> p (of_nat x));
  rewrite
    bigstar 0 1 (fun x -> p (of_nat x))
  as
    bigstar 0 (cardinal a #_) (fun x -> p (of_nat x));
  fold op_forall_Plus #a (fun x -> p x);
}

ghost
fn forevery_singleton_elim
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop { cardinal a #_ == 1 })
  requires
    forall+ (x:a). p x
  ensures
    p (of_nat 0)
{
  unfold op_forall_Plus #a (fun x -> p x);
  rewrite each cardinal a #_ as (0 + 1);
  bigstar_single_elim #0 #0 #(fun x -> p (of_nat #a x));
}

ghost
fn forevery_unit_intro
  (p : slprop)
  requires
    p
  ensures
    forevery unit (fun _ -> p)
{
  forevery_singleton_intro #unit (fun _ -> p);
}

ghost
fn forevery_unit_elim
  (p : slprop)
  requires
    forevery unit (fun _ -> p)
  ensures
    p
{
  forevery_singleton_elim #unit (fun _ -> p);
}

ghost
fn forevery_eta
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop)
  requires
    forevery a p
  ensures
    forevery a (fun x -> p x)
{
  unfold op_forall_Plus #a p;
  bigstar_eta ();
  fold op_forall_Plus #a (fun x -> p x);
  ();
}

ghost
fn forevery_uneta
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop)
  requires
    forevery a (fun x -> p x)
  ensures
    forevery a p
{
  unfold op_forall_Plus #a (fun x -> p x);
  bigstar_uneta ();
  fold op_forall_Plus #a p;
  ();
}

ghost
fn forevery_rw_type
  (a:Type0) {| d : enumerable a |}
  (b:Type{a == b})
  (f : a -> slprop)
  requires
    forall+ (x:a). f x
  ensures
    forevery b #d (fun (x:b) -> f x)
{
  ();
}

ghost
fn forevery_rw_size
  (n1 : nat)
  (n2 : nat{n1 == n2})
  (#p : natlt n1 -> slprop)
  requires
    forall+ (i : natlt n1). p i
  ensures
    forall+ (i : natlt n2). p i
{
  ()
}

ghost
fn forevery_rw_size2
  (n1 : nat)
  (n2 : nat{n1 == n2})
  (n3 : nat)
  (n4 : nat{n3 == n4})
  (#p : natlt n1 -> natlt n3 -> slprop)
  requires
    forall+ (i : natlt n1) (j : natlt n3). p i j
  ensures
    forall+ (i : natlt n2) (j : natlt n4). p i j
{
  ();
}

ghost
fn forevery_factor
  (n : nat)
  (d1 : nat) (d2 : nat { n == d1 * d2 })
  (p : natlt n -> slprop)
  requires
    forall+ (i:natlt n). p i
  ensures
    forall+ (i1:natlt d1) (i2:natlt d2). p (i1 * d2 + i2)
{
  open Kuiper.Bijection;
  forevery_rw_size n (d1 * d2);
  forevery_iso (bij_sym <| bij_nat_prod #d1 #d2) _;
  forevery_unflatten #(natlt d1) #_ #(natlt d2) (fun i1 i2 -> p (i1 * d2 + i2));
}

#push-options "--z3rlimit 20"
ghost
fn forevery_unfactor
  (n : nat)
  (d1 : nat) (d2 : nat { n == d1 * d2 })
  (p : natlt n -> slprop)
  requires
    forall+ (i1:natlt d1) (i2:natlt d2). p (i1 * d2 + i2)
  ensures
    forall+ (i:natlt n). p i
{
  open Kuiper.Bijection;
  forevery_flatten #(natlt d1) #_ #(natlt d2) (fun i1 i2 -> p (i1 * d2 + i2));
  forevery_iso (bij_nat_prod #d1 #d2) _;
  forevery_rw_size (d1 * d2) n;
  ()
}
#pop-options

ghost
fn forevery_unfactor'
  (n : nat)
  (d1 : nat) (d2 : nat { n == d1 * d2 })
  (p : natlt d1 -> natlt d2 -> slprop)
  requires
    forall+ (i1:natlt d1) (i2:natlt d2). p i1 i2
  ensures
    forall+ (i:natlt n). p (i/d2) (i%d2)
{
  forevery_unfactor n d1 d2 (fun i -> p (i/d2) (i%d2));
}

ghost
fn forevery_zip
  (#a:Type0) {| enumerable a |}
  (p1 p2 : a -> slprop)
  requires
    (forall+ (x:a). p1 x) **
    (forall+ (x:a). p2 x)
  ensures
    forall+ (x:a). p1 x ** p2 x
{
  rewrite forevery a (fun x -> p1 x)
       as bigstar #1 0 (cardinal a #_) (fun i -> p1 (of_nat i));
  unfold op_forall_Plus #a (fun x -> p2 x);
  bigstar_zip #1 0 (cardinal a #_) _ _;
  fold op_forall_Plus #a (fun x -> p1 x ** p2 x);
}

ghost
fn forevery_unzip
  (#a:Type0) {| enumerable a |}
  (p1 p2 : a -> slprop)
  requires
    forall+ (x:a). p1 x ** p2 x
  ensures
    (forall+ (x:a). p1 x) **
    (forall+ (x:a). p2 x)
{
  unfold op_forall_Plus #a (fun x -> p1 x ** p2 x);
  bigstar_unzip 0 (cardinal a #_) _ _;
  fold op_forall_Plus #a (fun x -> p1 x);
  fold op_forall_Plus #a (fun x -> p2 x);
}

ghost
fn forevery_map
  (#a:Type0) {| enumerable a |}
  (p1 p2 : a -> slprop)
  (f : (x:a -> stt_ghost unit emp_inames (p1 x) (fun _ -> p2 x)))
  requires
    forall+ (x:a). p1 x
  ensures
    forall+ (x:a). p2 x
{
  unfold op_forall_Plus #a (fun x -> p1 x);
  bigstar_map #_ #_ #0 #(cardinal a #_) (fun x -> f (of_nat x));
  fold op_forall_Plus #a (fun x -> p2 x);
}

ghost
fn forevery_map_2
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (p1 p2 : a -> b -> slprop)
  (f : (x:a -> y:b -> stt_ghost unit emp_inames (p1 x y) (fun _ -> p2 x y)))
  requires
    forall+ (x:a) (y:b). p1 x y
  ensures
    forall+ (x:a) (y:b). p2 x y
{
  forevery_map #a
    (fun x -> forevery b (fun y -> p1 x y))
    (fun x -> forevery b (fun y -> p2 x y))
    (fun x -> forevery_map (fun y -> p1 x y)
                          (fun y -> p2 x y)
                          (f x));
}

ghost
fn forevery_pad
  (n1 : nat)
  (n2 : nat{n1 <= n2})
  (p : natlt n1 -> slprop)
  requires
    forall+ (i : natlt n1). p i
  ensures
    forall+ (i : natlt n2). pad_f n2 p i
{
  forevery_tonat n1 _;
  bigstar_emp_intro n1 n2;
  bigstar_extensionality
    0 n1
    (fun i -> p i)
    (fun i -> if i < n1 then p i else emp)
    (fun i -> ());
  bigstar_extensionality
    n1 n2
    (fun i -> emp)
    (fun i -> if i < n1 then p i else emp)
    (fun i -> ());
  bigstar_paste #_ #0 #n2 n1 #(fun i -> if i < n1 then p i else emp);
  forevery_fromnat n2 (fun i -> pad_f n2 p i);
}

ghost
fn forevery_unpad
  (n1 : nat)
  (n2 : nat{n1 <= n2})
  (p : natlt n1 -> slprop)
  requires
    forall+ (i : natlt n2). pad_f n2 p i
  ensures
    forall+ (i : natlt n1). p i
{
  forevery_tonat n2 (pad_f n2 p);
  bigstar_cut #_ #0 #n2 n1 #(fun i -> if i < n1 then p i else emp);
  bigstar_extensionality
    0 n1
    (fun i -> if i < n1 then p i else emp)
    (fun i -> p i)
    (fun i -> ());
  bigstar_extensionality
    n1 n2
    (fun i -> if i < n1 then p i else emp)
    (fun i -> emp)
    (fun i -> ());
  bigstar_emp_elim #_ #n1 #n2;
  forevery_fromnat n1 p;
}

ghost
fn forevery_extract
  (#a:Type0) {| enumerable a |}
  (z : a)
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    p z ** (p z @==> forall+ (x:a). p x)
{
  ghost
  fn aux2 (x : a)
    requires (if to_nat x = to_nat z then p z else emp) ** (if to_nat x <> to_nat z then p x else emp)
    ensures  p x
  {
    let b = to_nat x = to_nat z;
    if b {
      rewrite (if to_nat x = to_nat z then p z else emp) as  p x;
      rewrite (if to_nat x <> to_nat z then p x else emp) as emp;
    } else {
      rewrite (if to_nat x = to_nat z then p z else emp) as  emp;
      rewrite (if to_nat x <> to_nat z then p x else emp) as p x;
    }
  };
  ghost
  fn aux1 (x : a)
    requires p x
    ensures  (if to_nat x = to_nat z then p z else emp) ** (if to_nat x <> to_nat z then p x else emp)
  {
    let b = to_nat x = to_nat z;
    if b {
      rewrite p x as (if to_nat x = to_nat z then p z else emp);
      rewrite emp as (if to_nat x <> to_nat z then p x else emp);
    } else {
      rewrite emp as (if to_nat x = to_nat z then p z else emp);
      rewrite p x as (if to_nat x <> to_nat z then p x else emp);
    }
  };
  forevery_map _ _ aux1;
  forevery_unzip #a _ _;
  assume (pure ((forall+ (x:a). if to_nat x = to_nat z then p z else emp) == p z));
  rewrite (forall+ (x:a). if to_nat x = to_nat z then p z else emp) as p z;

  ghost
  fn goback ()
    requires (forall+ (x:a). if to_nat x <> to_nat z then p x else emp) ** p z
    ensures  forall+ (x:a). p x
  {
    rewrite p z as (forall+ (x:a). if to_nat x = to_nat z then p x else emp);
    forevery_zip #a (fun x -> if to_nat x = to_nat z then p x else emp)
      (fun x -> if to_nat x <> to_nat z then p x else emp);
    forevery_map _ _ aux2;
  };

  intro_trade
    (p z) (forall+ (x:a). p x)
    (forall+ (x:a). if to_nat x <> to_nat z then p x else emp)
    goback;
}

ghost
fn forevery_extract_2
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (z : a) (w : b)
  (p : a -> b -> slprop)
  requires
    forall+ (x:a) (y:b). p x y
  ensures
    p z w ** (p z w @==> forall+ (x:a) (y:b). p x y)
{
  admit();
}

ghost
fn forevery_extract_if
  (#a:Type0) {| enumerable a |}
  (z : a)
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    p z **
    (forall+ (x:a).
      if Enumerable.to_nat x = Enumerable.to_nat z then emp else p x)
{
  (* Boring, but clearly provable. *)
  admit ();
}

ghost
fn forevery_extract_if_2
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (z : a) (w : b)
  (p : a -> b -> slprop)
  requires
    forall+ (x:a) (y:b). p x y
  ensures
    p z w **
    (forall+ (x:a) (y:b).
      if (Enumerable.to_nat x, Enumerable.to_nat y) = (Enumerable.to_nat z, Enumerable.to_nat w) then emp else p x y)
{
  forevery_flatten #a #_ #b _;
  forevery_extract_if (z, w) _;
  forevery_unflatten' #a #_ #b _;
  rewrite p (z,w)._1 (z,w)._2 as p z w;
}



ghost
fn forevery_boolean_split
  (#a:Type0) {| enumerable a |}
  (b : a -> bool)
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    (forall+ (x:a). if b x then emp else p x) **
    (forall+ (x:a). if b x then p x else emp)
{
  ghost
  fn aux (x:a)
    requires p x
    ensures
      (if b x then emp else p x) ** (if b x then p x else emp)
  {
    if (b x) {
      emp_unit (p x);
      rewrite p x
           as (if b x then emp else p x) ** (if b x then p x else emp);
    } else {
      emp_unit (p x);
      star_comm (p x) emp;
      rewrite p x
           as (if b x then emp else p x) ** (if b x then p x else emp);
    };
  };
  forevery_map _ _ aux;
  forevery_unzip #a _ _;
  ();
}

ghost
fn forevery_boolean_join
  (#a:Type0) {| enumerable a |}
  (b : a -> bool)
  (p1 p2 : a -> slprop)
  requires
    (forall+ (x:a). if b x then p1 x else emp) **
    (forall+ (x:a). if b x then emp else p2 x)
  ensures
    (forall+ (x:a). if b x then p1 x else p2 x)
{
  ghost
  fn aux (x:a)
    requires
      (if b x then p1 x else emp) ** (if b x then emp else p2 x)
    ensures
      (if b x then p1 x else p2 x)
  {
    if (b x) {
      rewrite p1 x as (if b x then p1 x else p2 x);
    } else {
      rewrite p2 x as (if b x then p1 x else p2 x);
    };
  };
  forevery_zip #a
    (fun x -> if b x then p1 x else emp)
    (fun x -> if b x then emp else p2 x);
  forevery_map _ _ aux;
  ();
}

ghost
fn forevery_boolean_equal_sides
  (#a:Type0) {| enumerable a |}
  (b : a -> bool)
  (p : a -> slprop)
  requires
    (forall+ (x:a). if b x then p x else p x)
  ensures
    (forall+ (x:a). p x)
{
  ghost
  fn aux (x:a)
    requires
      (if b x then p x else p x)
    ensures
      p x
  {
    rewrite (if b x then p x else p x) as p x;
    ();
  };
  forevery_map _ _ aux;
  ();
}

ghost
fn forevery_intro_if
  (#a:Type0) {| enumerable a |}
  (z : a)
  (p : a -> slprop)
  requires
    p z
  ensures
    (forall+ (x:a).
      if Enumerable.to_nat x = Enumerable.to_nat z then p x else emp)
{
  admit();
}

ghost
fn forevery_split_either
  (#a #b : Type0) {| enumerable a, enumerable b |}
  (p : either a b -> slprop)
  requires
    forall+ (x:either a b). p x
  ensures
    (forall+ (x:a). p (Inl x)) **
    (forall+ (x:b). p (Inr x))
{
  admit();
}

ghost
fn forevery_join_either
  (#a #b : Type0) {| enumerable a, enumerable b |}
  (p : either a b -> slprop)
  requires
    (forall+ (x:a). p (Inl x)) **
    (forall+ (x:b). p (Inr x))
  ensures
    forall+ (x:either a b). p x
{
  admit();
}
