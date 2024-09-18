module Kuiper.Functions

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
