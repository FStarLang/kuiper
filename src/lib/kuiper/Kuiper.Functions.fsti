module Kuiper.Functions

include FStar.Functions

(* Base properties and facts about functions. *)

let is_commutative (#a:Type) (f : a -> a -> a) : prop =
  forall x y. f x y == f y x

let is_associative (#a:Type) (f : a -> a -> a) : prop =
  forall x y z. f (f x y) z == f x (f y z)

let is_ac (#a:Type) (f : a -> a -> a) : prop =
  is_associative f /\ is_commutative f

let is_neutral_for (#a:Type) (e : a) (f : a -> a -> a) : prop =
  forall x. f e x == x /\ f x e == x

let is_semigroup (#a:Type) (e:a) (f : a -> a -> a) : prop =
  is_associative f /\ is_neutral_for e f

let is_comm_semigroup (#a:Type) (e:a) (f : a -> a -> a) : prop =
  is_semigroup e f /\ is_commutative f

let is_monoid (#a:Type) (e : a) (f : a -> a -> a) : prop =
  is_associative f /\ is_neutral_for e f

let no_overlap (f1 : 'a -> 'c) (f2 : 'b -> 'c) : prop =
  forall (x1 : 'a) (x2 : 'b). f1 x1 =!= f2 x2

let merge_either (f1 : 'a -> 'c) (f2 : 'b -> 'c) (x : either 'a 'b) : 'c =
  match x with
  | Inl x1 -> f1 x1
  | Inr x2 -> f2 x2

let left  (e : either 'a 'b{Inl? e}) : 'a = let Inl x = e in x
let right (e : either 'a 'b{Inr? e}) : 'b = let Inr x = e in x
